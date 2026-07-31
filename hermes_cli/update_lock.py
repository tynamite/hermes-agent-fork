"""Cross-process mutual exclusion for in-flight Hermes updates.

Three different surfaces can start an update of the same install tree:

* ``hermes update`` from a terminal,
* the dashboard's Update button (``POST /api/hermes/update`` →
  ``_spawn_hermes_action(["update"])``, detached),
* the desktop's Update button, which hands off to the Tauri
  ``hermes-setup --update`` and, on its failure screen, to install-mode
  bootstrap (``install.ps1`` / ``install.sh``).

Until now only the Tauri updater published an "update in progress" marker
(``UpdateMarkerGuard`` in ``apps/bootstrap-installer/src-tauri/src/update.rs``),
and only the Electron desktop consumed it (``electron/update-marker.ts``, to
gate local backend startup). Nothing stopped two *updaters* from running at
once — so a dashboard-spawned ``hermes update`` and an installer-driven
``git checkout`` could mutate the same checkout concurrently, rewriting source
under a live interpreter and leaving the tree half-updated.

This module makes that same marker the single lock for **all** update
entrypoints instead of adding a fourth mechanism. Its wire format remains
backward-compatible with the Rust and Electron readers:

    <HERMES_ROOT>/.hermes-update-in-progress
        body: "<pid>\\n<started_at_unix>[\\nruntime-restarts]"

A marker only counts as a live update when its pid is alive AND it is younger
than :data:`UPDATE_MARKER_MAX_AGE_MS` — mirroring ``readLiveUpdateMarker`` so a
crashed updater self-heals instead of wedging every future update. A stale
marker is removed on read by whoever notices it first.

One layering wrinkle: the Tauri updater holds this marker for its WHOLE run and
then spawns ``hermes update`` as a child stage. Without a handoff the child
sees its own parent's live marker and refuses — the GUI update deadlocks
against itself on every attempt ("Hermes is still running", retry forever).
The updater therefore exports :data:`HANDOFF_PID_ENV` naming its own pid, and
``acquire`` treats a live holder matching that pid as the lock we are already
running under. The env var alone grants nothing: the pid must also be the live
marker owner and this process's actual parent, so a copied or forged value
cannot bypass the lock.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in
# apps/desktop/electron/update-marker.ts — the same marker is read by both, and
# a shorter ceiling here would let Python steal a lock Electron still considers
# live. A full update (git pull + uv sync + desktop rebuild) is minutes.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

MARKER_NAME = ".hermes-update-in-progress"

# Set by an orchestrating updater (the Tauri `hermes-setup --update` flow) to
# its own pid before spawning `hermes update` as a child stage. The parent
# holds the marker for its whole run, so without this the child refuses its
# own parent's lock and the GUI update can never complete. See update_child_env
# in apps/bootstrap-installer/src-tauri/src/update.rs — keep the name in sync.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Exit code meaning "another updater/instance owns this install right now".
# Already the de-facto contract: the Windows shim + venv-holder guards in
# _cmd_update_impl exit 2, and the Tauri updater matches on it
# (UPDATE_EXIT_CONCURRENT in apps/bootstrap-installer/src-tauri/src/update.rs)
# to show "Hermes is still running" instead of a generic failure. Naming it
# here keeps the concurrent-update refusal on that same understood contract.
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Normalizes only the explicit ``<root>/profiles/<name>`` layout. Arbitrary
    nested custom homes remain intact, matching Electron's
    ``normalizeHermesHomeRoot`` and the HERMES_HOME passed to the Rust updater.
    Collapsing every path beneath the platform-native home would split Python
    from those readers for valid homes such as ``~/.hermes/team``.
    """
    from hermes_constants import get_process_hermes_home

    root = get_process_hermes_home()
    if root.parent.name.lower() == "profiles":
        root = root.parent.parent
    return root / MARKER_NAME


def _pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` currently exists.

    This must remain stdlib-only because the launch gate calls it from
    :mod:`hermes_bootstrap`, before interrupted-install recovery can repair
    third-party packages. POSIX ``kill(pid, 0)`` is a non-signalling probe.
    Windows must use ``OpenProcess`` instead: CPython's ``os.kill(pid, 0)``
    routes to ``GenerateConsoleCtrlEvent`` there and can interrupt the updater
    it is meant to inspect (bpo-14484).
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL(  # type: ignore[attr-defined]
                "kernel32", use_last_error=True
            )
            kernel32.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            process = kernel32.OpenProcess(
                0x1000,  # PROCESS_QUERY_LIMITED_INFORMATION
                False,
                pid,
            )
            if process:
                kernel32.CloseHandle(process)
                return True
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED => alive
        except (AttributeError, OSError, OverflowError):
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — reached only after os.name != "nt"
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us, if any.

    Read from :data:`HANDOFF_PID_ENV`. Malformed values count as absent —
    a broken handoff must fall back to the normal refusal, never crash.
    """
    raw = os.environ.get(HANDOFF_PID_ENV, "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def is_verified_handoff(holder_pid: int) -> bool:
    """Whether the live marker owner is this process's declared parent."""
    handoff_pid = _handoff_pid()
    if handoff_pid is None or handoff_pid != holder_pid:
        return False
    try:
        return os.getppid() == holder_pid
    except (AttributeError, OSError):
        return False


@dataclass(frozen=True)
class UpdateHolder:
    """A confirmed-live update currently holding the lock."""

    pid: int
    age_seconds: float
    runtime_restarts_authorized: bool = False


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Return the live update holding the lock, or ``None``.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``: absent,
    unreadable, malformed, dead-pid, and past-the-ceiling all mean "no live
    update", and a stale marker file is deleted so it can't strand future runs.
    Never raises.
    """
    marker = path or update_marker_path()
    try:
        raw = marker.read_text(encoding="utf-8")
    except OSError:
        return None  # absent or unreadable => no live update

    lines = raw.splitlines()
    try:
        pid = int(lines[0].strip())
    except (IndexError, ValueError):
        pid = -1
    try:
        started_at = float(lines[1].strip())
    except (IndexError, ValueError):
        started_at = float("-inf")

    age = time.time() - started_at
    if not _pid_alive(pid) or age > UPDATE_MARKER_MAX_AGE_SECONDS:
        try:
            marker.unlink()
        except OSError:
            pass
        return None

    runtime_restarts_authorized = (
        len(lines) > 2 and lines[2].strip() == "runtime-restarts"
    )
    return UpdateHolder(
        pid=pid,
        age_seconds=age,
        runtime_restarts_authorized=runtime_restarts_authorized,
    )


def describe_holder(holder: UpdateHolder) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    minutes, seconds = divmod(int(max(holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"✗ Another Hermes update is already running (PID {holder.pid}, "
        f"started {elapsed} ago).\n"
        "\n"
        "  Two updates mutating the same checkout corrupt it: one rewrites\n"
        "  source while the other is mid-install. Wait for it to finish, or\n"
        "  close the window/dashboard tab that started it, then retry."
    )


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update already holds it — callers
    decide whether that's a hard refusal (CLI/dashboard) or a wait. Releasing
    only removes the marker when *we* still own it, so a marker rewritten by a
    handoff partner (the Tauri updater overwrites it with its own pid) is never
    deleted out from under its new owner.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self._claim_pid: int | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` and the OS
        parent pid is our own orchestrating parent (the Tauri updater spawning
        `hermes update` as a stage): we run under ITS claim rather than
        refusing or re-writing the marker, and ``release`` leaves the parent's
        marker untouched.
        """
        existing = read_live_update(path=self.path)
        if existing is not None:
            if is_verified_handoff(existing.pid):
                self.holder = existing
                self._claim_pid = existing.pid
                # A previous child stage may have crashed during the narrow
                # restart phase. Close that phase before this retry performs
                # any new mutation under the parent's still-live claim.
                if not self.deauthorize_runtime_restarts():
                    self._claim_pid = None
                    return False
                return True
            self.holder = existing
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8"
            )
        except OSError as exc:
            # Best-effort, exactly like the Rust guard: an unwritable marker
            # must not block the update itself (that would be a worse failure
            # than the race it prevents). Degrade to the pre-lock behavior.
            logger.debug("Could not write update marker %s: %s", self.path, exc)
            return True
        self.acquired = True
        self._claim_pid = os.getpid()
        return True

    def _set_runtime_restarts_authorized(self, authorized: bool) -> bool:
        """Compare-and-rewrite our live claim's restart phase."""
        if self._claim_pid is None:
            # Marker creation is best-effort. If no claim exists, there is no
            # launch gate to bypass.
            return True
        try:
            raw = self.path.read_text(encoding="utf-8")
            lines = raw.splitlines()
            owner = int(lines[0].strip())
            started_at = lines[1].strip()
        except (OSError, IndexError, ValueError):
            return False
        if owner != self._claim_pid:
            return False
        body = f"{owner}\n{started_at}\n"
        if authorized:
            body += "runtime-restarts\n"
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(body, encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return True

    def authorize_runtime_restarts(self) -> bool:
        """Allow only managed runtime entrypoints after mutation completes."""
        return self._set_runtime_restarts_authorized(True)

    def deauthorize_runtime_restarts(self) -> bool:
        """Close the restart phase while preserving an orchestrator's claim."""
        return self._set_runtime_restarts_authorized(False)

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            self.deauthorize_runtime_restarts()
            self._claim_pid = None
            return
        self.acquired = False
        try:
            raw = self.path.read_text(encoding="utf-8")
            owner = int(raw.splitlines()[0].strip())
        except (OSError, IndexError, ValueError):
            self._claim_pid = None
            return
        if owner != os.getpid():
            # A handoff partner took ownership (e.g. the Tauri updater wrote
            # its own pid). Leave it alone — it's still a live update.
            self._claim_pid = None
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self._claim_pid = None

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
