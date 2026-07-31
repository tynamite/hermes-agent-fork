"""Tests for the external drain-control marker contract + gateway state machine.

Task 2.2/2.3. Two layers:
  * drain_control.py — the presence-based marker contract (write/clear/read,
    HERMES_HOME-scoped, never-raises).
  * GatewayRunner enter/exit/watcher + the new-turn accept gate — the
    reversible state machine driven by the marker.

Mocked tests are necessary-not-sufficient here (the HARD live-validation gate,
Q-B, exercises a real `hermes gateway run`); these lock the unit contract.
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gateway.drain_control as dc
from gateway.run import GatewayRunner
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


# ---------------------------------------------------------------------------
# Marker contract (drain_control.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestMarkerContract:
    def test_absent_by_default(self, home):
        assert dc.drain_requested() is False
        assert dc.read_drain_request() is None

    def test_write_then_present(self, home):
        payload = dc.write_drain_request(principal="nas")
        assert dc.drain_requested() is True
        assert payload["action"] == "drain"
        assert payload["principal"] == "nas"
        body = dc.read_drain_request()
        assert body is not None and body["principal"] == "nas"

    def test_compare_delete_is_serialized_against_replacement(
        self,
        home,
        monkeypatch,
    ):
        owned = dc.write_drain_request(
            principal="hermes-update",
            request_id="owned",
            owner_pid=123,
            owner_start_time=456,
        )
        marker = dc.drain_request_path()
        unlink_started = threading.Event()
        replacement_started = threading.Event()
        original_unlink = Path.unlink

        def delayed_unlink(path, *args, **kwargs):
            if path == marker:
                unlink_started.set()
                assert replacement_started.wait(timeout=1)
                time.sleep(0.05)
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", delayed_unlink)

        clear_thread = threading.Thread(
            target=lambda: dc.clear_drain_request_if_matches(owned)
        )

        def replace_marker():
            replacement_started.set()
            dc.write_drain_request(principal="operator")

        writer_thread = threading.Thread(target=replace_marker)
        clear_thread.start()
        assert unlink_started.wait(timeout=1)
        writer_thread.start()
        clear_thread.join(timeout=2)
        writer_thread.join(timeout=2)

        assert not clear_thread.is_alive()
        assert not writer_thread.is_alive()
        assert dc.read_drain_request()["principal"] == "operator"

    def test_live_updater_marker_uses_pid_fallback_without_start_time(
        self,
        monkeypatch,
    ):
        import gateway.status

        marker = {
            "principal": "hermes-update",
            "owner_pid": 123,
            "owner_start_time": None,
        }
        monkeypatch.setattr(
            gateway.status,
            "get_process_start_time",
            lambda _pid: None,
        )
        monkeypatch.setattr(gateway.status, "_pid_exists", lambda pid: pid == 123)

        assert dc._live_updater_owns_marker(marker) is True

    def test_updater_marker_rejects_reused_pid_when_start_time_is_known(
        self,
        monkeypatch,
    ):
        import gateway.status

        marker = {
            "principal": "hermes-update",
            "owner_pid": 123,
            "owner_start_time": 456,
        }
        monkeypatch.setattr(
            gateway.status,
            "get_process_start_time",
            lambda _pid: 789,
        )
        monkeypatch.setattr(gateway.status, "_pid_exists", lambda _pid: True)

        assert dc._live_updater_owns_marker(marker) is False

    def test_updater_marker_is_protected_when_liveness_is_unverifiable(
        self,
        monkeypatch,
    ):
        import gateway.status

        marker = {
            "principal": "hermes-update",
            "owner_pid": 123,
            "owner_start_time": 456,
        }
        monkeypatch.setattr(
            gateway.status,
            "get_process_start_time",
            MagicMock(side_effect=PermissionError("denied")),
        )

        assert dc._live_updater_owns_marker(marker) is True


class TestSuppressNotification:
    """The generic suppress_notification flag on the drain marker.

    Gates ONLY the gateway's home-channel shutdown broadcast (NAS auto-update
    sets it true). Default-false so legacy/operator drains behave as before.
    The reader reuses the NS-570 epoch-staleness check so an orphaned marker
    can never silence a fresh gateway.
    """

    def test_default_false(self, home):
        payload = dc.write_drain_request(principal="nas")
        assert payload["suppress_notification"] is False
        assert dc.drain_notification_suppressed() is False

    def test_flag_round_trips_true(self, home):
        payload = dc.write_drain_request(principal="nas", suppress_notification=True)
        assert payload["suppress_notification"] is True
        body = dc.read_drain_request()
        assert body is not None and body["suppress_notification"] is True
        assert dc.drain_notification_suppressed() is True


# ---------------------------------------------------------------------------
# Instantiation-epoch staleness (NS-570: orphaned marker on durable volume)
# ---------------------------------------------------------------------------


class TestInstantiationEpoch:
    def test_write_stamps_current_epoch(self, home):
        payload = dc.write_drain_request(principal="nas")
        assert payload["epoch"] == dc.current_instantiation_epoch()
        body = dc.read_drain_request()
        assert body is not None and body["epoch"] == dc.current_instantiation_epoch()


    def test_marker_from_prior_instantiation_reads_as_absent(self, home, monkeypatch):
        # THE NS-570 REGRESSION. A begin-drain marker written by a PREVIOUS
        # container/VM instantiation survives on the durable HERMES_HOME volume
        # across a machine restart. The freshly-restarted gateway (new epoch)
        # must treat it as absent, NOT re-engage drain.
        monkeypatch.setattr(dc, "current_instantiation_epoch", lambda: "epoch-OLD")
        dc.write_drain_request(principal="nas")  # stamps "epoch-OLD"
        assert dc.drain_requested() is True  # same epoch → active

        # Simulate the restart: a brand-new instantiation epoch.
        monkeypatch.setattr(dc, "current_instantiation_epoch", lambda: "epoch-NEW")
        # The marker file is still physically present on the volume…
        assert dc.drain_request_path().exists() is True
        # …but it is ignored because its epoch belongs to a prior instantiation.
        assert dc.drain_requested() is False


    def test_current_epoch_empty_when_proc_unreadable(self, monkeypatch):
        # When neither /proc identity source is readable, the epoch is "" so
        # the staleness check is disabled rather than crashing.
        from pathlib import Path as _P

        orig_read_text = _P.read_text

        def _boom(self, *a, **k):
            if str(self).startswith("/proc/"):
                raise OSError("no /proc")
            return orig_read_text(self, *a, **k)

        dc.current_instantiation_epoch.cache_clear()
        monkeypatch.setattr(_P, "read_text", _boom)
        try:
            assert dc.current_instantiation_epoch() == ""
        finally:
            dc.current_instantiation_epoch.cache_clear()


# ---------------------------------------------------------------------------
# Gateway state machine (enter / exit / idempotency)
# ---------------------------------------------------------------------------


def _drain_runner():
    runner, adapter = make_restart_runner()
    runner._external_drain_active = False
    runner._external_drain_blocks_internal = False
    # Bind the real methods under test.
    runner._enter_external_drain = GatewayRunner._enter_external_drain.__get__(
        runner, GatewayRunner
    )
    runner._exit_external_drain = GatewayRunner._exit_external_drain.__get__(
        runner, GatewayRunner
    )
    return runner, adapter


class TestDrainStateMachine:


    def test_enter_idempotent(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        runner._enter_external_drain()  # second call — no-op
        runner._update_runtime_status.assert_not_called()


    def test_exit_during_shutdown_does_not_revert_to_running(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        # A shutdown drain is now in progress — exit must NOT resurrect running.
        runner._draining = True
        runner._exit_external_drain()
        assert runner._external_drain_active is False
        runner._update_runtime_status.assert_not_called()

    def test_active_work_count_includes_async_delegations(
        self,
        monkeypatch,
    ):
        import tools.async_delegation

        runner, _ = _drain_runner()
        runner._running_agents = {}
        runner._background_tasks = set()
        monkeypatch.setattr(
            tools.async_delegation,
            "active_count",
            lambda: 1,
        )

        assert runner._active_work_count() == 1

    def test_active_work_count_excludes_supervised_lifecycle_tasks(
        self,
        monkeypatch,
    ):
        import tools.async_delegation
        import tools.process_registry

        runner, _ = _drain_runner()
        runner._running_agents = {}
        heartbeat_task = MagicMock()
        heartbeat_task.done.return_value = False
        runner._loop_heartbeat_task = heartbeat_task
        runner._background_tasks = {heartbeat_task}
        runner._supervised_tasks = {heartbeat_task}
        monkeypatch.setattr(tools.async_delegation, "active_count", lambda: 0)
        monkeypatch.setattr(
            tools.process_registry.process_registry,
            "has_any_active",
            lambda: False,
        )
        monkeypatch.setattr(
            tools.process_registry.process_registry,
            "pending_watchers",
            set(),
        )

        assert runner._active_work_count() == 0

        finite_task = MagicMock()
        finite_task.done.return_value = False
        runner._background_tasks.add(finite_task)

        assert runner._active_work_count() == 1

    @pytest.mark.parametrize(
        "registry_name",
        (
            "_deferred_agent_cleanup_tasks",
            "_fatal_handler_tasks",
        ),
    )
    def test_active_work_count_includes_detached_owned_registries(
        self,
        monkeypatch,
        registry_name,
    ):
        import tools.async_delegation
        import tools.process_registry

        runner, _ = _drain_runner()
        runner._running_agents = {}
        runner._background_tasks = set()
        runner._supervised_tasks = set()
        detached_task = MagicMock()
        detached_task.done.return_value = False
        setattr(runner, registry_name, {detached_task})
        monkeypatch.setattr(tools.async_delegation, "active_count", lambda: 0)
        monkeypatch.setattr(
            tools.process_registry.process_registry,
            "has_any_active",
            lambda: False,
        )
        monkeypatch.setattr(
            tools.process_registry.process_registry,
            "pending_watchers",
            set(),
        )

        assert runner._active_work_count() == 1

    def test_active_work_count_fails_closed_when_background_probe_raises(
        self,
        monkeypatch,
    ):
        import tools.async_delegation

        runner, _ = _drain_runner()
        runner._running_agents = {}
        runner._background_tasks = set()
        monkeypatch.setattr(
            tools.async_delegation,
            "active_count",
            MagicMock(side_effect=RuntimeError("unreadable")),
        )

        assert runner._active_work_count() == 1


# ---------------------------------------------------------------------------
# Watcher reconciliation
# ---------------------------------------------------------------------------


class TestDrainWatcher:

    @pytest.mark.asyncio
    async def test_watcher_enters_then_exits_with_marker(self, home):
        runner, _ = _drain_runner()
        runner._drain_control_watcher = GatewayRunner._drain_control_watcher.__get__(
            runner, GatewayRunner
        )
        # Drive a few ticks manually rather than spinning the loop.
        dc.write_drain_request()
        task = asyncio.create_task(runner._drain_control_watcher(interval=0.02))
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is True
        dc.clear_drain_request()
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is False
        runner._running = False
        await asyncio.sleep(0.04)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# New-turn accept gate
# ---------------------------------------------------------------------------


class TestNewTurnGate:
    @pytest.mark.asyncio
    async def test_new_turn_refused_during_external_drain(self):
        runner, _ = _drain_runner()
        runner._external_drain_active = True
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=make_restart_source(),
            message_id="m1",
        )
        result = await runner._handle_message(event)
        assert result is not None
        assert "draining" in result.lower()

    @pytest.mark.asyncio
    async def test_operator_drain_still_allows_internal_work(self, monkeypatch):
        runner, _ = _drain_runner()
        runner._external_drain_active = True
        runner._external_drain_blocks_internal = False
        event = MessageEvent(
            text="[SYSTEM: Background process completed]",
            source=make_restart_source(),
            message_id="m1",
            internal=True,
        )

        async def stop_after_gate(*_args, **_kwargs):
            raise RuntimeError("passed gate")

        monkeypatch.setattr(runner, "_handle_message_with_agent", stop_after_gate)
        with pytest.raises(RuntimeError, match="passed gate"):
            await runner._handle_message(event)

    @pytest.mark.asyncio
    async def test_updater_drain_refuses_internal_work(self):
        runner, _ = _drain_runner()
        runner._external_drain_active = True
        runner._external_drain_blocks_internal = True
        event = MessageEvent(
            text="[SYSTEM: Background process completed]",
            source=make_restart_source(),
            message_id="m1",
            internal=True,
        )

        result = await runner._handle_message(event)

        assert result is not None
        assert "draining" in result.lower()
