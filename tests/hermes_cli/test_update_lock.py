"""Cross-process update mutual exclusion (``hermes_cli.update_lock``).

Three surfaces can start an update of one install tree: a terminal ``hermes
update``, the dashboard's Update button (which spawns that same command
detached), and the desktop's Update button (Tauri updater → install-mode
bootstrap on its failure screen). Before the shared lock, two of them could run
concurrently and rewrite source under a live interpreter — observed in the wild
as an installer ``git checkout`` rewinding the checkout ~9k commits while a
dashboard-spawned ``hermes update`` was mid-``npm install``, which then failed
against the rewound tree's manifests.

These exercise the real marker file against a temp home — no mocks — because
the contract that matters is what the Rust updater and the Electron gate see on
disk.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli.update_lock import (
    HANDOFF_PID_ENV,
    UPDATE_EXIT_CONCURRENT,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    UpdateHolder,
    UpdateLock,
    describe_holder,
    read_live_update,
    update_marker_path,
)


def test_windows_pid_above_dword_range_is_not_probed(monkeypatch):
    from hermes_cli import update_lock

    monkeypatch.setattr(update_lock.os, "name", "nt")

    assert update_lock._pid_alive(0x1_0000_0000) is False


def test_windows_handoff_accepts_only_managed_launcher_ancestry(monkeypatch):
    from hermes_cli import update_lock

    monkeypatch.setenv(HANDOFF_PID_ENV, "4321")
    monkeypatch.setattr(update_lock.os, "name", "nt")
    monkeypatch.setattr(update_lock.os, "getppid", lambda: 7777)
    monkeypatch.setattr(
        update_lock.sys,
        "executable",
        r"C:\Hermes\venv\Scripts\python.exe",
    )
    monkeypatch.setattr(
        update_lock,
        "_windows_process_parent_and_image",
        lambda pid: (4321, r"c:\hermes\VENV\scripts\HERMES.EXE")
        if pid == 7777
        else None,
    )

    assert update_lock.is_verified_handoff(4321) is True


@pytest.mark.parametrize(
    ("launcher_parent", "launcher_image"),
    [
        (9999, r"C:\Hermes\venv\Scripts\hermes.exe"),
        (4321, r"C:\Other\hermes.exe"),
    ],
)
def test_windows_handoff_rejects_wrong_launcher_identity(
    monkeypatch,
    launcher_parent,
    launcher_image,
):
    from hermes_cli import update_lock

    monkeypatch.setenv(HANDOFF_PID_ENV, "4321")
    monkeypatch.setattr(update_lock.os, "name", "nt")
    monkeypatch.setattr(update_lock.os, "getppid", lambda: 7777)
    monkeypatch.setattr(
        update_lock.sys,
        "executable",
        r"C:\Hermes\venv\Scripts\python.exe",
    )
    monkeypatch.setattr(
        update_lock,
        "_windows_process_parent_and_image",
        lambda _pid: (launcher_parent, launcher_image),
    )

    assert update_lock.is_verified_handoff(4321) is False


def test_non_update_cli_launch_is_blocked_by_live_update(monkeypatch, capsys):
    import hermes_bootstrap

    holder = UpdateHolder(pid=4321, age_seconds=3)
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: holder,
    )

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(
            ["chat"], entrypoint="cli"
        )

    assert exc.value.code == UPDATE_EXIT_CONCURRENT
    assert "Another Hermes update is already running" in capsys.readouterr().out


def test_update_cli_without_holder_reaches_authoritative_lock(monkeypatch):
    import hermes_bootstrap

    read = Mock(return_value=None)
    monkeypatch.setattr("hermes_cli.update_lock.read_live_update", read)

    hermes_bootstrap.enforce_update_launch_gate(
        ["--profile", "work", "update", "--yes"],
        entrypoint="cli",
    )

    read.assert_called_once_with()


def test_foreign_update_is_blocked_before_recovery(monkeypatch):
    import hermes_bootstrap

    monkeypatch.delenv(HANDOFF_PID_ENV, raising=False)
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: UpdateHolder(pid=4321, age_seconds=3),
    )

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(
            ["update", "--yes"], entrypoint="cli"
        )

    assert exc.value.code == UPDATE_EXIT_CONCURRENT


def test_verified_update_handoff_passes_bootstrap_gate(monkeypatch):
    import hermes_bootstrap

    monkeypatch.setenv(HANDOFF_PID_ENV, "4321")
    monkeypatch.setattr(os, "getppid", lambda: 4321)
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: UpdateHolder(pid=4321, age_seconds=3),
    )

    hermes_bootstrap.enforce_update_launch_gate(
        ["update", "--yes"], entrypoint="cli"
    )


def test_verified_handoff_passes_exact_desktop_rebuild_stage(monkeypatch):
    import hermes_bootstrap

    monkeypatch.setenv(HANDOFF_PID_ENV, "4321")
    monkeypatch.setattr(os, "getppid", lambda: 4321)
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: UpdateHolder(pid=4321, age_seconds=3),
    )

    hermes_bootstrap.enforce_update_launch_gate(
        ["desktop", "--build-only"], entrypoint="cli"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["desktop"],
        ["desktop", "--build-only", "--source"],
        ["desktop", "--no-open"],
    ],
)
def test_verified_handoff_does_not_admit_other_desktop_commands(
    monkeypatch,
    argv,
):
    import hermes_bootstrap

    monkeypatch.setenv(HANDOFF_PID_ENV, "4321")
    monkeypatch.setattr(os, "getppid", lambda: 4321)
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: UpdateHolder(pid=4321, age_seconds=3),
    )

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(argv, entrypoint="cli")

    assert exc.value.code == UPDATE_EXIT_CONCURRENT


def test_matching_handoff_env_from_non_child_is_blocked(monkeypatch):
    import hermes_bootstrap

    monkeypatch.setenv(HANDOFF_PID_ENV, "4321")
    monkeypatch.setattr(os, "getppid", lambda: 9999)
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: UpdateHolder(pid=4321, age_seconds=3),
    )

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(
            ["update", "--yes"], entrypoint="cli"
        )

    assert exc.value.code == UPDATE_EXIT_CONCURRENT


@pytest.mark.parametrize(
    "argv",
    [
        ["gateway"],
        ["gateway", "run"],
        ["dashboard", "--no-open"],
        ["serve"],
    ],
)
def test_authorized_runtime_restart_passes_launch_gate(
    monkeypatch, argv
):
    import hermes_bootstrap

    holder = UpdateHolder(
        pid=4321,
        age_seconds=3,
        runtime_restarts_authorized=True,
    )
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: holder,
    )

    hermes_bootstrap.enforce_update_launch_gate(
        argv, entrypoint="cli"
    )


def test_restart_phase_does_not_admit_general_cli(monkeypatch):
    import hermes_bootstrap

    holder = UpdateHolder(
        pid=4321,
        age_seconds=3,
        runtime_restarts_authorized=True,
    )
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: holder,
    )

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(
            ["chat"], entrypoint="cli"
        )

    assert exc.value.code == UPDATE_EXIT_CONCURRENT


def test_restart_phase_admits_only_dashboard_owned_tui_gateway(monkeypatch):
    import hermes_bootstrap

    holder = UpdateHolder(
        pid=4321,
        age_seconds=3,
        runtime_restarts_authorized=True,
    )
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: holder,
    )
    monkeypatch.delenv("HERMES_TUI_DASHBOARD", raising=False)

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(
            [], entrypoint="dashboard"
        )
    assert exc.value.code == UPDATE_EXIT_CONCURRENT

    monkeypatch.setenv("HERMES_TUI_DASHBOARD", "1")
    hermes_bootstrap.enforce_update_launch_gate([], entrypoint="dashboard")


@pytest.mark.parametrize(
    "argv",
    [
        ["dashboard", "--status"],
        ["serve", "--stop"],
        ["dashboard", "register"],
    ],
)
def test_restart_phase_does_not_admit_runtime_management(
    monkeypatch, argv
):
    import hermes_bootstrap

    holder = UpdateHolder(
        pid=4321,
        age_seconds=3,
        runtime_restarts_authorized=True,
    )
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: holder,
    )

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(
            argv, entrypoint="cli"
        )

    assert exc.value.code == UPDATE_EXIT_CONCURRENT


@pytest.mark.parametrize(
    "entrypoint",
    ["other", "agent", "acp"],
)
def test_restart_phase_does_not_admit_sibling_entrypoints(
    monkeypatch, entrypoint
):
    import hermes_bootstrap

    holder = UpdateHolder(
        pid=4321,
        age_seconds=3,
        runtime_restarts_authorized=True,
    )
    monkeypatch.setattr(
        "hermes_cli.update_lock.read_live_update",
        lambda: holder,
    )

    with pytest.raises(SystemExit) as exc:
        hermes_bootstrap.enforce_update_launch_gate(
            ["update"], entrypoint=entrypoint
        )

    assert exc.value.code == UPDATE_EXIT_CONCURRENT


@pytest.mark.parametrize(
    "module",
    [
        "hermes_cli.main",
        "run_agent",
        "acp_adapter.entry",
        "cron.scheduler",
    ],
)
def test_fresh_entrypoint_import_is_blocked_in_bootstrap(
    tmp_path, module
):
    marker = tmp_path / ".hermes-update-in-progress"
    marker.write_text(
        f"{os.getpid()}\n{int(time.time())}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == UPDATE_EXIT_CONCURRENT
    assert "Another Hermes update is already running" in result.stdout


# A pid no live process owns. os.kill(pid, 0) must report it dead so a crashed
# updater can never wedge every future update. Deliberately larger than any
# platform's pid_t so it also covers the corrupt-marker path (OverflowError).
DEAD_PID = 4294967294


@pytest.fixture
def marker(tmp_path):
    return tmp_path / ".hermes-update-in-progress"


def test_marker_path_is_shared_across_profiles(tmp_path, monkeypatch):
    """Named profiles sharing one install must observe the same lock."""
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert update_marker_path() == root / ".hermes-update-in-progress"


def test_marker_path_preserves_arbitrary_nested_custom_home(
    tmp_path,
    monkeypatch,
):
    """Only the explicit profiles/<name> shape is install-wide."""
    import hermes_constants

    native = tmp_path / ".hermes"
    custom = native / "team"
    monkeypatch.setattr(
        hermes_constants,
        "_get_platform_default_hermes_home",
        lambda: native,
    )
    monkeypatch.setenv("HERMES_HOME", str(custom))

    assert update_marker_path() == custom / ".hermes-update-in-progress"


def test_runtime_restart_phase_is_live_claim_metadata(marker):
    lock = UpdateLock(path=marker)
    assert lock.acquire() is True

    assert lock.authorize_runtime_restarts() is True
    holder = read_live_update(path=marker)
    assert holder is not None
    assert holder.runtime_restarts_authorized is True

    assert lock.deauthorize_runtime_restarts() is True
    holder = read_live_update(path=marker)
    assert holder is not None
    assert holder.runtime_restarts_authorized is False
    lock.release()


def test_acquire_writes_pid_and_start_time(marker):
    lock = UpdateLock(path=marker)

    assert lock.acquire() is True
    assert lock.acquired is True

    lines = marker.read_text(encoding="utf-8").splitlines()
    assert int(lines[0]) == os.getpid(), "the Electron gate probes this pid for liveness"
    assert int(lines[1]) == pytest.approx(time.time(), abs=5)
    assert len(lines) == 2, "wire format is exactly pid + started_at"


def test_second_acquire_is_refused_while_the_first_is_live(marker):
    """The bug: two updaters mutating one checkout at the same time."""
    first = UpdateLock(path=marker)
    assert first.acquire() is True

    second = UpdateLock(path=marker)
    assert second.acquire() is False
    assert second.holder is not None
    assert second.holder.pid == os.getpid()
    assert second.acquired is False


def test_refused_lock_does_not_delete_the_live_owners_marker(marker):
    first = UpdateLock(path=marker)
    first.acquire()

    second = UpdateLock(path=marker)
    second.acquire()
    second.release()

    assert marker.exists(), "a refused claimant must never clear the live owner's lock"

    first.release()
    assert not marker.exists()


def test_release_leaves_a_marker_a_handoff_partner_now_owns(marker):
    """The desktop writes the marker, then the Tauri updater takes ownership.

    Releasing must not delete a marker whose pid is no longer ours — that would
    reopen the gate while the partner is still mid-update.
    """
    lock = UpdateLock(path=marker)
    lock.acquire()

    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")
    lock.release()

    assert marker.exists(), "the partner's marker is not ours to remove"


def test_dead_owner_is_reclaimed_not_honored(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True
    assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


def test_owner_past_the_age_ceiling_is_reclaimed(marker):
    """A live-but-wedged updater must not hold the lock forever."""
    long_ago = int(time.time()) - UPDATE_MARKER_MAX_AGE_SECONDS - 60
    marker.write_text(f"{os.getpid()}\n{long_ago}\n", encoding="utf-8")

    lock = UpdateLock(path=marker)
    assert lock.acquire() is True


@pytest.mark.parametrize(
    "body",
    ["", "not-a-pid\n123\n", "\n\n", "12345"],
    ids=["empty", "garbage-pid", "blank-lines", "no-start-time"],
)
def test_malformed_markers_never_block_an_update(marker, body):
    marker.write_text(body, encoding="utf-8")

    assert read_live_update(path=marker) is None
    assert UpdateLock(path=marker).acquire() is True


def test_stale_marker_is_removed_on_read(marker):
    marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

    assert read_live_update(path=marker) is None
    assert not marker.exists(), "whoever notices a stale marker clears it"


def test_absent_marker_reports_no_live_update(marker):
    assert read_live_update(path=marker) is None


def test_context_manager_releases_even_on_exception(marker):
    with pytest.raises(RuntimeError):
        with UpdateLock(path=marker) as lock:
            assert lock.acquired is True
            raise RuntimeError("update blew up mid-flight")

    assert not marker.exists(), "a crashed update must not strand the lock"


def test_describe_holder_names_the_pid_and_elapsed_time(marker):
    lock = UpdateLock(path=marker)
    lock.acquire()

    holder = read_live_update(path=marker)
    assert holder is not None
    message = describe_holder(holder)

    assert str(os.getpid()) in message, "the user needs the pid to find the other update"
    assert "already running" in message


def test_unwritable_marker_location_does_not_block_the_update(tmp_path):
    """Degrade to pre-lock behavior rather than refusing to update at all.

    An unwritable marker path is a worse reason to block an update than the
    race the lock prevents.
    """
    lock = UpdateLock(path=tmp_path / "nonexistent-file" / "marker")
    (tmp_path / "nonexistent-file").write_text("i am a file, not a dir", encoding="utf-8")

    assert lock.acquire() is True
    assert lock.acquired is False, "nothing was written, so there is nothing to release"


class TestHandoffFromOrchestratingUpdater:
    """The Tauri updater holds the marker, then spawns ``hermes update``.

    The regression: the child saw its own parent's live marker and exited 2,
    so every GUI update failed with "Hermes is still running" and retrying
    just re-ran the same self-deadlock. The parent names its pid in
    HANDOFF_PID_ENV; the live holder must also be the child's actual parent.
    """

    def test_child_runs_under_the_parents_live_claim(self, marker, monkeypatch):
        # Stand in for the parent updater with our own (live) pid.
        marker.write_text(
            f"{os.getpid()}\n{int(time.time())}\nruntime-restarts\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))
        monkeypatch.setattr(os, "getppid", lambda: os.getpid())

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        assert lock.acquired is False, "the parent's claim is not ours to own"
        assert read_live_update(
            path=marker
        ).runtime_restarts_authorized is False
        assert lock.authorize_runtime_restarts() is True
        assert read_live_update(
            path=marker
        ).runtime_restarts_authorized is True

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()
        assert read_live_update(
            path=marker
        ).runtime_restarts_authorized is False

    def test_matching_holder_without_parent_relationship_is_refused(
        self,
        marker,
        monkeypatch,
    ):
        marker.write_text(
            f"{os.getpid()}\n{int(time.time())}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))
        monkeypatch.setattr(os, "getppid", lambda: os.getpid() + 1)

        assert UpdateLock(path=marker).acquire() is False

    def test_handoff_pid_that_is_not_the_live_holder_grants_nothing(self, marker, monkeypatch):
        """The env var alone must not bypass the lock."""
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid() + 1))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None

    @pytest.mark.parametrize("value", ["", "not-a-pid", "-1", "0"], ids=["empty", "garbage", "negative", "zero"])
    def test_malformed_handoff_values_fall_back_to_refusal(self, marker, monkeypatch, value):
        marker.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        monkeypatch.setenv(HANDOFF_PID_ENV, value)

        assert UpdateLock(path=marker).acquire() is False

    def test_handoff_env_with_no_marker_claims_normally(self, marker, monkeypatch):
        """A handoff pid must not stop us writing our own claim when unlocked."""
        monkeypatch.setenv(HANDOFF_PID_ENV, str(os.getpid()))

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True
        assert lock.acquired is True
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getpid()


class TestAncestryHandoff:
    """Staged updaters older than the HANDOFF_PID_ENV export never send it.

    ``hermes-setup`` under ``~/.hermes`` is only refreshed by a full installer
    run, so an updated checkout (new lock) driven by a pre-handoff staged
    updater (old parent) deadlocks on exit 2 forever unless the child also
    recognizes a live holder that is its own process ancestor.

    ``_pid_alive`` is pinned True here because the hermetic conftest guards
    ``os.kill`` probes of pids outside the test subtree (our ppid included);
    liveness has its own coverage above — ancestry is what's under test.
    """

    @pytest.fixture(autouse=True)
    def _liveness_pinned_true(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.update_lock._pid_alive", lambda pid: True)

    def test_marker_owned_by_our_parent_process_is_our_orchestrator(self, marker):
        marker.write_text(f"{os.getppid()}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is True, "a live ancestor's claim is the one we run under"
        assert lock.acquired is False, "the parent's claim is not ours to own"

        lock.release()
        assert marker.exists(), "the parent still needs its marker after our stage ends"
        assert int(marker.read_text(encoding="utf-8").splitlines()[0]) == os.getppid()

    def test_live_non_ancestor_holder_is_still_refused(self, marker):
        """Ancestry must not open the lock to unrelated concurrent updaters."""
        marker.write_text(f"{DEAD_PID}\n{int(time.time())}\n", encoding="utf-8")

        lock = UpdateLock(path=marker)
        assert lock.acquire() is False
        assert lock.holder is not None
        assert lock.holder.pid == DEAD_PID
