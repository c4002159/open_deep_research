"""Shared marker for security-policy refusals.

A *fatal* denial means the request itself is not permitted (a write, DDL, shell or file
access). Rephrasing cannot help, so the agent loop must stop instead of retrying. A
non-fatal rejection (unsupported function, stray comment, disallowed library) is
something the model can fix and is returned as an ordinary error message.
"""

POLICY_DENIED_PREFIX = "[POLICY_DENIED]"

FATAL_GUIDANCE = (
    "This request is not permitted by the security policy and cannot be completed in any "
    "form. Do NOT retry, rephrase or look for a workaround. Stop and tell the user plainly "
    "that the action was refused (the data is read-only and unchanged)."
)


def fatal_denial(reason: str) -> str:
    return f"{POLICY_DENIED_PREFIX} {reason}. {FATAL_GUIDANCE}"


def is_fatal_denial(text: object) -> bool:
    return isinstance(text, str) and text.startswith(POLICY_DENIED_PREFIX)
