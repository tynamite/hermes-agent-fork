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
        body: "<pid>\\n<started_at_unix>[\\n<phase>][\\n<process-start-identity>]"

A legacy marker only counts as a live update when its pid is alive AND it is
younger than :data:`UPDATE_MARKER_MAX_AGE_MS`. New markers also carry a
process-start identity, so a verified live updater remains authoritative past
that ceiling while a recycled pid is rejected. A stale marker is removed on
read by whoever notices it first. Marker operations use a short-lived
``.hermes-update-in-progress.lock`` sidecar acquired with atomic directory
creation; stale sidecars are first atomically renamed to a unique tombstone so
two reclaimers cannot unlink a replacement owner's lock.
The sidecar owner file carries ``pid``, creation time, and a process-start
identity; a matching live identity remains authoritative even if the updater
is suspended past the short-operation age fallback.

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
import ntpath
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in
# apps/desktop/electron/update-marker.ts — the same marker is read by both, and
# a shorter ceiling here would let Python steal a lock Electron still considers
# live. A full update (git pull + uv sync + desktop rebuild) is minutes.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

MARKER_NAME = ".hermes-update-in-progress"
MARKER_OPERATION_LOCK_NAME = ".hermes-update-in-progress.lock"
MARKER_OPERATION_LOCK_STALE_SECONDS = 30

# Set by an orchestrating updater (the Tauri `hermes-setup --update` flow) to
# its own pid before spawning `hermes update` as a child stage. The parent
# holds the marker for its whole run, so without this the child refuses its
# own parent's lock and the GUI update can never complete. See update_child_env
# in apps/bootstrap-installer/src-tauri/src/update.rs — keep the name in sync.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Set only on the short-lived ``python -c`` child used by the updater's
# post-update cross-module import probe.  The bootstrap gate admits that child
# only when this names its actual parent and the attested marker-owner pid;
# an arbitrary process cannot bypass the gate by setting the variable on its
# own.
IMPORT_PROBE_PARENT_PID_ENV = "HERMES_UPDATE_IMPORT_PROBE_PARENT_PID"
# In the desktop handoff chain the Python updater is the probe's direct
# parent, while the Tauri process one level above it owns the marker. This
# second attestation carries that already-verified marker-owner identity to
# the probe child.
IMPORT_PROBE_MARKER_PID_ENV = "HERMES_UPDATE_IMPORT_PROBE_MARKER_PID"

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


def _posix_pid_is_zombie(pid: int) -> bool:
    """Return whether a POSIX process is defunct, without third-party imports."""
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        # macOS/BSD have no procfs, and Linux may run with procfs hidden or
        # unmounted (for example in a restricted container or chroot). Ask the
        # standard ``ps`` utility when procfs cannot provide the process state.
        try:
            import subprocess

            result = subprocess.run(
                ["ps", "-o", "state=", "-p", str(pid)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            return result.returncode == 0 and result.stdout.strip().startswith("Z")
        except Exception:
            return False
    except (PermissionError, OSError):
        return False

    # /proc/<pid>/stat wraps the command name in parentheses; the name itself
    # may contain spaces or closing parentheses, so split after the final one.
    _command, separator, fields = raw_stat.rpartition(")")
    state_fields = fields.split()
    return bool(separator and state_fields and state_fields[0] == "Z")


def _pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` currently exists.

    This must remain stdlib-only because the launch gate calls it from
    :mod:`hermes_bootstrap`, before interrupted-install recovery can repair
    third-party packages. POSIX ``kill(pid, 0)`` is a non-signalling probe,
    but it still succeeds for zombies, which are already dead and cannot own
    a useful update claim. A stdlib-only procfs/``ps`` check rejects those.
    Windows must use ``OpenProcess`` instead: CPython's ``os.kill(pid, 0)``
    routes to ``GenerateConsoleCtrlEvent`` there and can interrupt the updater
    it is meant to inspect (bpo-14484).
    """
    if pid <= 0 or (os.name == "nt" and pid > 0xFFFFFFFF):
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
            kernel32.WaitForSingleObject.argtypes = (
                wintypes.HANDLE,
                wintypes.DWORD,
            )
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            process = kernel32.OpenProcess(
                0x1000 | 0x100000,  # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
                False,
                pid,
            )
            if process:
                try:
                    # A successfully opened process object can already be
                    # signaled (exited) while another handle keeps it alive.
                    # WAIT_TIMEOUT is the only result that proves execution is
                    # still in progress.
                    return kernel32.WaitForSingleObject(process, 0) == 0x00000102
                finally:
                    kernel32.CloseHandle(process)
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED => alive
        except (AttributeError, OSError, OverflowError):
            return False
    if _posix_pid_is_zombie(pid):
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


def _process_start_identity(pid: int) -> str | None:
    """Return a stable per-process start identity, when the OS exposes one.

    The marker-operation sidecar must remain authoritative while its owner is
    suspended. Pairing its PID with this identity lets readers distinguish a
    recycled PID without expiring a genuinely live owner by age. Keep this
    helper stdlib-only because the bootstrap launch gate imports this module
    before third-party packages are repairable.
    """
    if pid <= 0:
        return None

    stat_path = Path(f"/proc/{pid}/stat")
    try:
        _command, separator, fields = stat_path.read_text(encoding="utf-8").rpartition(")")
        values = fields.split()
        # After the command name, field 22 (starttime) is index 19.
        if separator and len(values) > 19:
            return values[19]
    except (FileNotFoundError, IndexError, PermissionError, OSError):
        pass

    # Linux has no portable process-start source once procfs is unavailable;
    # keep the sidecar's conservative age fallback rather than spawning a
    # locale-dependent probe for every marker read.
    if sys.platform.startswith("linux"):
        return None

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
            kernel32.GetProcessTimes.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
            )
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            process = kernel32.OpenProcess(0x1000, False, pid)
            if process:
                try:
                    creation = wintypes.FILETIME()
                    exit_time = wintypes.FILETIME()
                    kernel_time = wintypes.FILETIME()
                    user_time = wintypes.FILETIME()
                    if kernel32.GetProcessTimes(
                        process,
                        ctypes.byref(creation),
                        ctypes.byref(exit_time),
                        ctypes.byref(kernel_time),
                        ctypes.byref(user_time),
                    ):
                        value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                        return str(value)
                finally:
                    kernel32.CloseHandle(process)
        except (AttributeError, OSError, OverflowError, TypeError):
            return None
        return None

    # macOS/BSD do not expose procfs. `ps -o lstart=` is available on the
    # supported hosts and is stable for the lifetime of a process.
    try:
        import subprocess

        env = {
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=5,
            check=False,
        )
        identity = result.stdout.strip()
        if result.returncode == 0 and identity:
            return identity
    except Exception:
        pass
    return None


def _marker_process_start_identity(lines: list[str]) -> str | None:
    """Read the optional fourth-line process identity from a marker."""
    if len(lines) <= 3:
        return None
    identity = lines[3].strip()
    return identity or None


def _marker_owner_is_live(
    pid: int,
    age_seconds: float,
    process_identity: str | None,
) -> bool:
    """Validate a marker owner, retaining verified live claims past the age ceiling."""
    if not _pid_alive(pid):
        return False
    if process_identity:
        current_identity = _process_start_identity(pid)
        # An unavailable probe is not evidence of PID reuse. Retain the claim
        # conservatively rather than opening a concurrent-update window.
        return current_identity is None or current_identity == process_identity
    return age_seconds <= UPDATE_MARKER_MAX_AGE_SECONDS


def _marker_body(
    pid: int,
    started_at: str,
    process_identity: str | None,
    *,
    runtime_restarts: bool = False,
) -> str:
    """Render a marker while keeping legacy two-line claims readable."""
    phase = "runtime-restarts" if runtime_restarts else ""
    if process_identity:
        return f"{pid}\n{started_at}\n{phase}\n{process_identity}\n"
    if runtime_restarts:
        return f"{pid}\n{started_at}\nruntime-restarts\n"
    return f"{pid}\n{started_at}\n"


def _ensure_marker_process_identity_locked(
    marker: Path,
    pid: int,
    started_at: str,
) -> None:
    """Upgrade a legacy/pre-claim marker while its operation sidecar is held."""
    process_identity = _process_start_identity(pid)
    if not process_identity:
        return
    try:
        with marker.open("r+b") as handle:
            raw = handle.read().decode("utf-8")
            lines = raw.splitlines()
            if (
                not lines
                or int(lines[0].strip()) != pid
                or len(lines) < 2
                or lines[1].strip() != started_at
            ):
                return
            if _marker_process_start_identity(lines) == process_identity:
                return
            body = _marker_body(
                pid,
                started_at,
                process_identity,
                runtime_restarts=(
                    len(lines) > 2 and lines[2].strip() == "runtime-restarts"
                ),
            ).encode("utf-8")
            handle.seek(0)
            handle.write(body)
            handle.truncate()
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except (OSError, IndexError, ValueError, UnicodeDecodeError):
        return


def _marker_operation_lock_path(marker: Path) -> Path:
    """Stable sidecar path used to serialize marker check/claim/reclaim."""
    return marker.with_name(MARKER_OPERATION_LOCK_NAME)


def _live_marker_operation_holder(marker: Path) -> tuple[int, float] | None:
    """Return a live sidecar owner while marker publication is in progress.

    The sidecar is deliberately part of the reader protocol as well as the
    writer protocol.  A writer creates the directory before publishing the
    marker, so treating an absent marker as clear during that interval would
    let a runtime start in the middle of an update.  A missing/malformed owner
    file is considered active for the short stale-lock window: that covers the
    tiny create-directory → owner-file write interval without trusting a
    crashed lock forever. A well-formed owner with a matching process-start
    identity remains active regardless of age, so a suspended updater cannot
    lose its claim while a recycled PID is still rejected.
    """
    lock_dir = _marker_operation_lock_path(marker)
    try:
        stat = lock_dir.stat()
    except OSError:
        return None

    owner_pid: int | None = None
    owner_identity: str | None = None
    try:
        lines = (lock_dir / "owner").read_text(encoding="utf-8").splitlines()
        parsed = int(lines[0].strip())
        if parsed > 0:
            owner_pid = parsed
            if len(lines) > 2 and lines[2].strip():
                owner_identity = lines[2].strip()
    except (FileNotFoundError, IndexError, ValueError, OSError):
        pass

    age = max(0.0, time.time() - stat.st_mtime)
    if owner_pid is not None:
        if not _pid_alive(owner_pid):
            return None
        if owner_identity:
            current_identity = _process_start_identity(owner_pid)
            # If identity inspection is temporarily unavailable, retain the
            # live owner rather than opening a concurrent-update window.
            if current_identity is None or current_identity == owner_identity:
                return owner_pid, age
            return None
        if age >= MARKER_OPERATION_LOCK_STALE_SECONDS:
            return None
        return owner_pid, age
    if age < MARKER_OPERATION_LOCK_STALE_SECONDS:
        return -1, age
    return None


def _reap_stale_marker_operation_lock(lock_dir: Path) -> bool:
    """Remove a crashed sidecar lock only when its owner is not alive.

    Reaping first renames the sidecar to a unique sibling tombstone.  The
    rename is atomic, so only one competing reaper can take ownership; a new
    claimant can then create the original path without a delayed reaper being
    able to unlink its replacement owner file.
    """
    owner_file = lock_dir / "owner"
    owner_identity = ""
    try:
        raw = owner_file.read_text(encoding="utf-8")
        lines = raw.splitlines()
        owner_pid = int(lines[0].strip())
        owner_identity = lines[2].strip() if len(lines) > 2 else ""
    except (FileNotFoundError, IndexError, ValueError, OSError):
        try:
            age = time.time() - lock_dir.stat().st_mtime
        except OSError:
            return True
        if age < MARKER_OPERATION_LOCK_STALE_SECONDS:
            return False
        owner_pid = 0

    try:
        age = time.time() - lock_dir.stat().st_mtime
    except OSError:
        return True
    if owner_pid > 0 and _pid_alive(owner_pid):
        if owner_identity:
            current_identity = _process_start_identity(owner_pid)
            # A matching identity (or an unavailable probe) means the owner is
            # still authoritative. Only a known mismatch proves PID reuse.
            if current_identity is None or current_identity == owner_identity:
                return False
        elif age < MARKER_OPERATION_LOCK_STALE_SECONDS:
            return False
    reclaim_dir = lock_dir.with_name(
        f"{lock_dir.name}.reaping-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        lock_dir.rename(reclaim_dir)
    except FileNotFoundError:
        return True
    except OSError:
        return False

    try:
        (reclaim_dir / "owner").unlink(missing_ok=True)
        reclaim_dir.rmdir()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def _marker_operation_lock(marker: Path):
    """Serialize a marker operation with an atomic sidecar directory claim."""
    lock_dir = _marker_operation_lock_path(marker)
    owner_file = lock_dir / "owner"
    deadline = time.monotonic() + 5.0
    while True:
        try:
            lock_dir.mkdir()
        except FileExistsError:
            _reap_stale_marker_operation_lock(lock_dir)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring marker operation lock {lock_dir}")
            time.sleep(0.01)
            continue
        except OSError:
            raise
        break

    try:
        identity = _process_start_identity(os.getpid()) or ""
        _write_marker_exclusive(
            owner_file,
            f"{os.getpid()}\n{int(time.time())}\n{identity}\n",
        )
    except BaseException:
        owner_file.unlink(missing_ok=True)
        lock_dir.rmdir()
        raise
    try:
        yield
    finally:
        try:
            owner_file.unlink(missing_ok=True)
            lock_dir.rmdir()
        except OSError:
            pass


def _reclaim_stale_marker_locked(marker: Path, expected_raw: str) -> bool:
    """Remove a stale marker after the caller acquired the sidecar lock."""
    try:
        if marker.read_text(encoding="utf-8") != expected_raw:
            return False
    except FileNotFoundError:
        return True
    except OSError:
        return False

    try:
        marker.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _reclaim_stale_marker(marker: Path, expected_raw: str) -> bool:
    """Compare-and-delete a stale marker under the shared sidecar lock."""
    try:
        with _marker_operation_lock(marker):
            return _reclaim_stale_marker_locked(marker, expected_raw)
    except (OSError, TimeoutError):
        return False


def _write_marker_exclusive(marker: Path, body: str) -> None:
    """Create and publish a complete marker only if the path is absent."""
    fd = os.open(
        marker,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except BaseException:
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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


def _windows_process_parent_and_image(pid: int) -> tuple[int, str] | None:
    """Return a Windows process's parent pid and executable path, fail-closed."""
    if pid <= 0 or pid > 0xFFFFFFFF:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            ]

        kernel32 = ctypes.WinDLL(  # type: ignore[attr-defined]
            "kernel32", use_last_error=True
        )
        kernel32.CreateToolhelp32Snapshot.argtypes = (
            wintypes.DWORD,
            wintypes.DWORD,
        )
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        )
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        )
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == wintypes.HANDLE(-1).value:
            return None
        parent_pid = None
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while has_entry:
                if int(entry.th32ProcessID) == pid:
                    parent_pid = int(entry.th32ParentProcessID)
                    break
                has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
        if not parent_pid:
            return None

        process = kernel32.OpenProcess(
            0x1000,  # PROCESS_QUERY_LIMITED_INFORMATION
            False,
            pid,
        )
        if not process:
            return None
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(
                process,
                0,
                buffer,
                ctypes.byref(size),
            ):
                return None
            return parent_pid, buffer.value
        finally:
            kernel32.CloseHandle(process)
    except (AttributeError, OSError, OverflowError, TypeError, ValueError):
        return None


def _is_windows_launcher_handoff(holder_pid: int, launcher_pid: int) -> bool:
    """Verify Tauri -> managed hermes.exe -> Python without trusting names."""
    process_info = _windows_process_parent_and_image(launcher_pid)
    if process_info is None:
        return False
    launcher_parent_pid, launcher_image = process_info
    expected_launcher = ntpath.join(
        ntpath.dirname(sys.executable),
        "hermes.exe",
    )
    return (
        launcher_parent_pid == holder_pid
        and ntpath.normcase(ntpath.abspath(launcher_image))
        == ntpath.normcase(ntpath.abspath(expected_launcher))
    )


def is_verified_handoff(holder_pid: int) -> bool:
    """Whether the live marker owner is our declared updater ancestor."""
    handoff_pid = _handoff_pid()
    if handoff_pid is None or handoff_pid != holder_pid:
        return False
    try:
        parent_pid = os.getppid()
    except (AttributeError, OSError):
        return False
    if parent_pid == holder_pid:
        return True
    return os.name == "nt" and _is_windows_launcher_handoff(
        holder_pid,
        parent_pid,
    )


def is_verified_import_probe(holder_pid: int) -> bool:
    """Whether a ``python -c`` import probe is our updater's child.

    The caller additionally requires the exact ``-c`` interpreter shape.  The
    parent-pid attestation here prevents a copied environment variable from
    opening the launch gate for an unrelated process.
    """
    raw = os.environ.get(IMPORT_PROBE_PARENT_PID_ENV, "").strip()
    try:
        parent_pid = int(raw)
    except ValueError:
        return False
    marker_raw = os.environ.get(IMPORT_PROBE_MARKER_PID_ENV, "").strip()
    try:
        marker_pid = int(marker_raw) if marker_raw else parent_pid
    except ValueError:
        return False
    if (
        parent_pid <= 0
        or marker_pid <= 0
        or marker_pid != holder_pid
    ):
        return False
    try:
        if os.getppid() != parent_pid or not _pid_alive(parent_pid):
            return False
        # In a direct CLI update the probe's parent owns the marker. In the
        # Tauri handoff chain, the inherited HANDOFF_PID_ENV names the live
        # marker owner one level above the Python updater.
        return holder_pid == parent_pid or _handoff_pid() == holder_pid
    except (AttributeError, OSError):
        return False


@dataclass(frozen=True)
class UpdateHolder:
    """A confirmed-live update currently holding the lock."""

    pid: int
    age_seconds: float
    runtime_restarts_authorized: bool = False
    operation_lock: bool = False


def read_live_update(
    *, path: Path | None = None, _lock_held: bool = False, _retries: int = 0
) -> UpdateHolder | None:
    """Return the live update holding the lock, or ``None``.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``:
    absent/unreadable state means "no live update" only when the marker
    operation sidecar is also clear. A live sidecar is reported as an active
    operation while the replacement marker is being published. Malformed,
    dead-pid, recycled-identity, and legacy past-the-ceiling markers are
    reclaimed so they can't strand future runs. Never raises.
    """
    marker = path or update_marker_path()
    try:
        raw = marker.read_text(encoding="utf-8")
    except OSError:
        if not _lock_held:
            operation = _live_marker_operation_holder(marker)
            if operation is not None:
                pid, age = operation
                return UpdateHolder(
                    pid=pid,
                    age_seconds=age,
                    operation_lock=True,
                )
            # The sidecar may have been released just before the first
            # marker read completed. Re-read once after it appears clear so a
            # newly published live marker cannot be mistaken for absence.
            if _retries < 1:
                return read_live_update(
                    path=marker,
                    _lock_held=False,
                    _retries=_retries + 1,
                )
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
    process_identity = _marker_process_start_identity(lines)
    if not _marker_owner_is_live(pid, age, process_identity):
        reclaimed = (
            _reclaim_stale_marker_locked(marker, raw)
            if _lock_held
            else _reclaim_stale_marker(marker, raw)
        )
        if not reclaimed:
            try:
                replacement = marker.read_text(encoding="utf-8")
            except OSError:
                replacement = raw
            if replacement != raw and _retries < 2:
                return read_live_update(
                    path=marker,
                    _lock_held=_lock_held,
                    _retries=_retries + 1,
                )
            if not _lock_held:
                operation = _live_marker_operation_holder(marker)
                if operation is not None:
                    operation_pid, operation_age = operation
                    return UpdateHolder(
                        pid=operation_pid,
                        age_seconds=operation_age,
                        operation_lock=True,
                    )
                if _retries < 1:
                    return read_live_update(
                        path=marker,
                        _lock_held=False,
                        _retries=_retries + 1,
                    )
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
    if holder.operation_lock:
        return (
            "✗ Another Hermes update is already starting (the update-operation "
            "lock is held).\n"
            "\n"
            "  Wait for that update to finish publishing its lock, then retry."
        )
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
    handoff partner (the Tauri updater adopts the desktop's pre-claim) is never
    deleted out from under its new owner.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self._claim_pid: int | None = None
        self._claim_started_at: str | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` and the OS
        parent pid is our own orchestrating parent (the Tauri updater spawning
        `hermes update` as a stage): we run under ITS claim rather than
        refusing or re-writing the marker, and ``release`` leaves the parent's
        marker untouched.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("Could not prepare update marker %s: %s", self.path, exc)
            return True

        pid = os.getpid()
        started_at = str(int(time.time()))
        body = _marker_body(
            pid,
            started_at,
            _process_start_identity(pid),
        )
        for _attempt in range(8):
            try:
                with _marker_operation_lock(self.path):
                    # Keep the sidecar across stale-marker reclamation and the
                    # replacement publication.  Leaving this context between
                    # those two operations creates a marker-free window in
                    # which readers can start a runtime before this claimant
                    # has published its lock.
                    for _claim_attempt in range(8):
                        try:
                            _write_marker_exclusive(self.path, body)
                        except FileExistsError:
                            existing = read_live_update(
                                path=self.path,
                                _lock_held=True,
                            )
                            if existing is not None:
                                if is_verified_handoff(existing.pid):
                                    self.holder = existing
                                    self._claim_pid = existing.pid
                                    try:
                                        self._claim_started_at = (
                                            self.path.read_text(
                                                encoding="utf-8"
                                            )
                                            .splitlines()[1]
                                            .strip()
                                        )
                                    except (OSError, IndexError):
                                        self._claim_started_at = None
                                    if self._claim_started_at:
                                        _ensure_marker_process_identity_locked(
                                            self.path,
                                            existing.pid,
                                            self._claim_started_at,
                                        )
                                    # A previous child stage may have crashed during
                                    # the narrow restart phase. Close that phase
                                    # before this retry performs mutation under the
                                    # parent's claim.
                                    if not self.deauthorize_runtime_restarts(
                                        _lock_held=True
                                    ):
                                        self._claim_pid = None
                                        return False
                                    return True
                                self.holder = existing
                                return False
                            # The stale/malformed marker was atomically reclaimed
                            # under this same sidecar. Retry the exclusive create
                            # before releasing it, so no sibling reader sees an
                            # unlocked install.
                            continue
                        except OSError as exc:
                            # Best-effort, exactly like the Rust guard: an unwritable
                            # marker must not block the update itself. Degrade to
                            # pre-lock behavior rather than claiming a path we could
                            # not publish.
                            logger.debug(
                                "Could not write update marker %s: %s",
                                self.path,
                                exc,
                            )
                            return True
                        self.acquired = True
                        self._claim_pid = pid
                        self._claim_started_at = started_at
                        return True
            except TimeoutError:
                logger.debug(
                    "Could not acquire marker operation lock %s",
                    self.path,
                )
                # A live sidecar is itself an in-flight update operation. Do
                # not fall through to the historical best-effort success path:
                # the marker may not have been published yet, but mutation can
                # already be imminent.
                self.holder = read_live_update(path=self.path)
                if self.holder is None:
                    self.holder = UpdateHolder(
                        pid=-1,
                        age_seconds=0,
                        operation_lock=True,
                    )
                return False
            except OSError as exc:
                # Marker locking is best-effort when the parent filesystem
                # cannot create the sidecar (permissions, read-only mount,
                # or an unsupported directory operation).  Preserve the
                # historical update behavior instead of aborting before the
                # mutation path starts.
                logger.debug(
                    "Could not create marker operation lock %s: %s",
                    self.path,
                    exc,
                )
                return True

        existing = read_live_update(path=self.path)
        if existing is not None:
            if is_verified_handoff(existing.pid):
                self.holder = existing
                self._claim_pid = existing.pid
                if self.deauthorize_runtime_restarts():
                    return True
                self._claim_pid = None
                return False
            self.holder = existing
            return False
        logger.debug("Could not atomically claim update marker %s", self.path)
        return True

    def _set_runtime_restarts_authorized(
        self, authorized: bool, *, _lock_held: bool = False
    ) -> bool:
        if _lock_held:
            return self._set_runtime_restarts_authorized_locked(authorized)
        try:
            with _marker_operation_lock(self.path):
                return self._set_runtime_restarts_authorized_locked(authorized)
        except (OSError, TimeoutError):
            return False

    def _set_runtime_restarts_authorized_locked(self, authorized: bool) -> bool:
        """Compare-and-rewrite our live claim's restart phase."""
        if self._claim_pid is None:
            # Marker creation is best-effort. If no claim exists, there is no
            # launch gate to bypass.
            return True
        try:
            # Keep the descriptor open while checking and updating. If another
            # process replaces the pathname after this open, writes still land
            # on the old inode rather than clobbering the replacement claim.
            with self.path.open("r+b") as handle:
                raw_bytes = handle.read()
                raw = raw_bytes.decode("utf-8")
                lines = raw.splitlines()
                owner = int(lines[0].strip())
                started_at = lines[1].strip()
                if owner != self._claim_pid or (
                    self._claim_started_at is not None
                    and started_at != self._claim_started_at
                ):
                    return False

                process_identity = _marker_process_start_identity(lines)
                if process_identity is None:
                    process_identity = _process_start_identity(owner)
                body = _marker_body(
                    owner,
                    started_at,
                    process_identity,
                    runtime_restarts=authorized,
                ).encode("utf-8")
                handle.seek(0)
                handle.write(body)
                handle.truncate()
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
        except (OSError, IndexError, ValueError, UnicodeDecodeError):
            return False
        return True

    def authorize_runtime_restarts(self) -> bool:
        """Allow only managed runtime entrypoints after mutation completes."""
        return self._set_runtime_restarts_authorized(True)

    def deauthorize_runtime_restarts(self, *, _lock_held: bool = False) -> bool:
        """Close the restart phase while preserving an orchestrator's claim."""
        return self._set_runtime_restarts_authorized(
            False,
            _lock_held=_lock_held,
        )

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            self.deauthorize_runtime_restarts()
            self._claim_pid = None
            return
        self.acquired = False
        try:
            raw = self.path.read_text(encoding="utf-8")
            lines = raw.splitlines()
            owner = int(lines[0].strip())
            started_at = lines[1].strip()
        except (OSError, IndexError, ValueError):
            self._claim_pid = None
            return
        if owner != os.getpid() or (
            self._claim_started_at is not None
            and started_at != self._claim_started_at
        ):
            # A handoff partner took ownership (e.g. the Tauri updater wrote
            # its own pid). Leave it alone — it's still a live update.
            self._claim_pid = None
            return
        _reclaim_stale_marker(self.path, raw)
        self._claim_pid = None

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
