"""Bearer-token authentication: static API keys (service accounts) or HS256 JWTs.

The authenticated ``user_id`` is the *only* identity the rest of the service trusts. It is
never taken from a request body or header, and it is validated against a strict pattern
because it ends up inside Redis keys, LangGraph thread ids and the memory-store user id.
"""

from __future__ import annotations

import hmac
import re
from typing import Optional

import jwt

from open_deep_research.service.settings import Settings

_USER_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


class AuthError(Exception):
    pass


class Authenticator:
    def __init__(self, settings: Settings):
        self.api_keys = settings.api_keys
        self.jwt_secret = settings.jwt_secret

    def authenticate(self, authorization: Optional[str]) -> str:
        if not authorization:
            raise AuthError("missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("expected 'Authorization: Bearer <token>'")
        token = token.strip()

        user = None
        for key, uid in self.api_keys.items():           # constant-time compare against every key
            if hmac.compare_digest(token.encode(), key.encode()):
                user = uid
        if user is None and self.jwt_secret:
            try:
                claims = jwt.decode(token, self.jwt_secret, algorithms=["HS256"], options={"require": ["exp", "sub"]})
                user = str(claims["sub"])
            except jwt.PyJWTError as e:
                raise AuthError(f"invalid token: {type(e).__name__}")
        if user is None:
            raise AuthError("invalid credentials")
        if not _USER_RE.match(user):
            raise AuthError("invalid user id")
        return user
