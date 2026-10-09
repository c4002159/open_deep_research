import asyncio

import pytest

from open_deep_research.model_fallback import ainvoke_with_fallback, is_retryable_model_error


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class HttpError(Exception):
    def __init__(self, status):
        self.status_code = status


class Fake:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result


@pytest.mark.parametrize("exc,expected", [
    (RateLimitError("429"), True),
    (HttpError(503), True),
    (HttpError(429), True),
    (TimeoutError(), True),
    (ConnectionError(), True),
    (AuthenticationError("bad key"), False),
    (HttpError(400), False),
    (ValueError("context length exceeded"), False),
])
def test_error_classification(exc, expected):
    assert is_retryable_model_error(exc) is expected


def test_primary_success_never_touches_fallback():
    p, f = Fake("A"), Fake("B")
    assert asyncio.run(ainvoke_with_fallback(p, f, [])) == "A" and f.calls == 0


def test_availability_error_switches_to_fallback():
    p, f, stats = Fake(exc=RateLimitError("slow down")), Fake("B"), {}
    assert asyncio.run(ainvoke_with_fallback(p, f, [], stats=stats)) == "B"
    assert p.calls == 1 and f.calls == 1 and stats["fallbacks"] == 1


def test_non_availability_error_is_not_masked():
    p, f = Fake(exc=AuthenticationError("bad key")), Fake("B")
    with pytest.raises(AuthenticationError):
        asyncio.run(ainvoke_with_fallback(p, f, []))
    assert f.calls == 0


def test_no_fallback_configured_reraises():
    with pytest.raises(RateLimitError):
        asyncio.run(ainvoke_with_fallback(Fake(exc=RateLimitError("x")), None, []))


def test_fallback_failure_propagates():
    with pytest.raises(HttpError):
        asyncio.run(ainvoke_with_fallback(Fake(exc=RateLimitError("x")), Fake(exc=HttpError(503)), []))
