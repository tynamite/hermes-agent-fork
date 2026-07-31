import asyncio
import json
import threading

import pytest

from hermes_cli import web_server
from hermes_cli.pty_session import PtySessionRegistry


class FakeBridge:
    def __init__(self):
        self.alive = True

    def read(self, timeout):
        return b""        # idle forever

    def write(self, data):
        pass

    def resize(self, cols, rows):
        pass

    def close(self):
        self.alive = False

    def is_alive(self):
        return self.alive


@pytest.mark.asyncio
async def test_concurrent_attach_key_spawns_exactly_one_bridge():
    registry = PtySessionRegistry(
        ttl=60,
        max_sessions=4,
        buffer_cap=1024,
        read_timeout=0.01,
    )
    spawn_started = threading.Event()
    release_spawn = threading.Event()
    calls = 0

    def slow_spawn():
        nonlocal calls
        calls += 1
        spawn_started.set()
        assert release_spawn.wait(timeout=2)
        return FakeBridge()

    first = asyncio.create_task(
        registry.attach_or_spawn("same", spawn=slow_spawn)
    )
    assert await asyncio.to_thread(spawn_started.wait, 1)
    second = asyncio.create_task(
        registry.attach_or_spawn("same", spawn=slow_spawn)
    )
    await asyncio.sleep(0.05)
    release_spawn.set()
    try:
        (first_session, first_created), (
            second_session,
            second_created,
        ) = await asyncio.gather(first, second)
        assert calls == 1
        assert first_session is second_session
        assert (first_created, second_created) == (True, False)
    finally:
        release_spawn.set()
        await registry.close_all()


@pytest.fixture
def pty_keepalive_harness(monkeypatch):
    spawned = []

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        spawned.append(argv)
        return b

    monkeypatch.setattr(web_server.PtyBridge, "spawn", staticmethod(fake_spawn))
    monkeypatch.setattr(web_server, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(web_server, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        resume = "child" if kw.get("resume") == "parent" else kw.get("resume")
        env = {"HERMES_TUI_RESUME": resume} if resume else {}
        return (["x", resume or "fresh"], "/tmp", env)

    monkeypatch.setattr(web_server, "_resolve_chat_argv_async", fake_argv)

    try:
        yield spawned
    finally:
        web_server.PTY_REGISTRY._sessions.clear()


@pytest.mark.asyncio
async def test_attach_token_reuses_same_session(pty_keepalive_harness):
    """Two connects with the same ?attach= token hit one spawned bridge."""
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1") as ws2:
        ws2.send_bytes(b"again")
    assert len(pty_keepalive_harness) == 1                # reattached, did not respawn


@pytest.mark.asyncio
async def test_attach_token_reuses_same_resume(pty_keepalive_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&resume=same") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1&resume=same") as ws2:
        ws2.send_bytes(b"again")
    assert pty_keepalive_harness == [["x", "same"]]




@pytest.mark.asyncio
async def test_attach_token_reuses_canonical_resume(pty_keepalive_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&resume=parent") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1&resume=child") as ws2:
        ws2.send_bytes(b"again")
    assert pty_keepalive_harness == [["x", "child"]]




@pytest.mark.asyncio
async def test_attach_token_reuses_default_chat_after_active_session_fallback(
    pty_keepalive_harness, tmp_path, monkeypatch
):
    from starlette.testclient import TestClient

    active_session_file = tmp_path / "active-session.json"
    monkeypatch.setattr(
        web_server,
        "_active_session_file_for_channel",
        lambda app, channel: active_session_file,
    )

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&channel=CHAT") as ws1:
        ws1.send_bytes(b"hi")

    active_session_file.write_text(json.dumps({"session_id": "existing"}))

    with client.websocket_connect("/api/pty?attach=TOK1&channel=CHAT") as ws2:
        ws2.send_bytes(b"again")

    assert pty_keepalive_harness == [["x", "fresh"]]
