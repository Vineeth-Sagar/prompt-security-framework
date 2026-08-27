"""Unhandled server errors must still carry CORS headers.

Reported live as "Could not reach the server." on the Playground, with
the backend demonstrably up and reachable (GET /health returned 200 from
the very same page, and the decision log recorded the request).

The mechanism, confirmed by probing response headers directly rather
than inferred: Starlette's middleware stack puts `ServerErrorMiddleware`
*outside* `CORSMiddleware`. Any exception that propagates out of a route
without a registered handler is therefore turned into a bare 500 by the
outermost middleware, after CORSMiddleware has already been passed —
so the response carries no `access-control-allow-origin` header. A
browser then refuses to expose that response to JS at all, `fetch()`
rejects with a TypeError indistinguishable from a genuine network
failure, and the frontend's catch-all reports the server as unreachable.

The user-visible effect is the worst possible one for debugging: a real,
specific server-side error is replaced with a false claim that the
server is down.

Registering a handler for `Exception` does NOT fix this — Starlette
routes `Exception`/500 handlers to `ServerErrorMiddleware` specifically
(see its `build_middleware_stack`), i.e. back outside CORSMiddleware.
The fix has to be a middleware installed *inside* the CORS layer, which
is what `app.main`'s exception-catching middleware does.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

CORS_HEADER = "access-control-allow-origin"


@pytest.fixture
def boom_route():
    """Mount a route that raises an unregistered exception type, then
    remove it — exercising the generic failure path without depending on
    any particular real route being broken."""

    @app.get("/__test_boom")
    async def boom():
        raise RuntimeError("unexpected failure inside a route")

    yield
    app.router.routes = [
        route for route in app.router.routes if getattr(route, "path", None) != "/__test_boom"
    ]


@pytest.mark.asyncio
async def test_unhandled_error_still_returns_cors_headers(boom_route):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/__test_boom", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 500
    # Without this header the browser discards the response entirely and
    # the client sees a network error instead of a server error.
    assert CORS_HEADER in {key.lower() for key in response.headers}


@pytest.mark.asyncio
async def test_unhandled_error_returns_json_detail_not_a_bare_body(boom_route):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/__test_boom", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    # Says the server failed — the opposite of "could not reach the
    # server" — so the reported symptom can't recur silently.
    assert "unexpected error" in detail.lower()
    # The exception's own text must not leak to an API client.
    assert "RuntimeError" not in detail
    assert "unexpected failure inside a route" not in detail


@pytest.mark.asyncio
async def test_successful_response_still_has_cors_headers():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    assert CORS_HEADER in {key.lower() for key in response.headers}
