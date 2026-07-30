"""Tests for half-updated-venv hardening.

Covers three additions to ``hermes update``:

1. ``_venv_core_imports_healthy`` — the venv health probe that lets an
   "Already up to date" checkout still repair a broken dependency install.
2. ``_detect_venv_python_processes`` — the venv-interpreter process guard
   that refuses to mutate the venv while another Python process is using it.
3. The commit_count == 0 repair branch wiring in ``_cmd_update_impl``.

Platform-specific paths are exercised via ``_is_windows`` patching so they
run on any host (same approach as test_update_concurrent_quarantine).
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


# ---------------------------------------------------------------------------
# _venv_core_imports_healthy
# ---------------------------------------------------------------------------




def _fake_venv_python(tmp_path, *, windows: bool = False):
    bin_dir = tmp_path / "venv" / ("Scripts" if windows else "bin")
    bin_dir.mkdir(parents=True)
    py = bin_dir / ("python.exe" if windows else "python")
    py.write_bytes(b"")
    return py




# ---------------------------------------------------------------------------
# _detect_venv_python_processes
# ---------------------------------------------------------------------------


def _proc(
    pid: int,
    exe: str | None,
    name: str,
    cmdline: list[str] | None = None,
    cwd: str = "",
):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "exe": exe,
        "name": name,
        "cmdline": cmdline or [],
        "cwd": cwd,
    }
    proc.environ.return_value = {}
    return proc


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_finds_posix_launchers(_winp, tmp_path):
    venv_python = str(tmp_path / "venv" / "bin" / "python")
    dot_venv_python = str(tmp_path / ".venv" / "bin" / "python3.13")
    base_python = "/usr/bin/python3.13"
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(101, base_python, "python3.13", [venv_python, "-m", "worker"]),
                _proc(102, dot_venv_python, "python3.13"),
                _proc(
                    103,
                    base_python,
                    "python3.13",
                    ["venv/bin/python", "-m", "worker"],
                    cwd=str(tmp_path),
                ),
                _proc(104, base_python, "python3.13", [base_python, "worker.py"]),
                _proc(
                    105,
                    "/usr/bin/pypy3.10",
                    "pypy3.10",
                    [str(tmp_path / "venv" / "bin" / "pypy3.10"), "worker.py"],
                ),
                _proc(
                    106,
                    str(tmp_path / "venv-other" / "bin" / "python3.13"),
                    "python3.13",
                ),
                _proc(
                    107,
                    "/usr/bin/python3.14t",
                    "python3.14t",
                    [str(tmp_path / "venv" / "bin" / "python3.14t"), "worker.py"],
                ),
            ]
        ),
        Process=MagicMock(),
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [101, 102, 103, 105, 107]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_matches_resolved_symlinked_project_root(
    _winp, tmp_path
):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    logical_root = tmp_path / "logical-root"
    logical_root.symlink_to(real_root, target_is_directory=True)
    venv_python = str(real_root / "venv" / "bin" / "python")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [_proc(108, "/usr/bin/python3", "python3", [venv_python, "worker.py"])]
        ),
        Process=MagicMock(),
    )

    with patch.object(cli_main, "PROJECT_ROOT", logical_root), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [108]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_ignores_argument_mentions(_winp, tmp_path):
    venv_python = str(tmp_path / "venv" / "bin" / "python")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(201, "/bin/bash", "bash", ["bash", "-c", f"{venv_python} worker"]),
                _proc(
                    202,
                    "/usr/bin/python3",
                    "python3",
                    ["/usr/bin/python3", "-c", f"print({venv_python!r})"],
                ),
                _proc(
                    203,
                    "/usr/bin/python3",
                    "python3",
                    ["venv/bin/python", "-m", "worker"],
                    cwd="",
                ),
            ]
        ),
        Process=MagicMock(),
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_resolves_bare_argv0_from_target_path(
    _winp, tmp_path
):
    outside_bin = tmp_path / "outside-bin"
    venv_bin = tmp_path / "venv" / "bin"
    outside_bin.mkdir()
    venv_bin.mkdir(parents=True)
    for python_path in (outside_bin / "python", venv_bin / "python"):
        python_path.write_text("#!/bin/sh\n")
        python_path.chmod(0o755)

    found = _proc(
        301,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "worker"],
        cwd=str(tmp_path),
    )
    found.environ.return_value = {
        "PATH": os.pathsep.join([str(venv_bin), str(outside_bin)])
    }
    shadowed = _proc(
        302,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "worker"],
        cwd=str(tmp_path),
    )
    shadowed.environ.return_value = {
        "PATH": os.pathsep.join([str(outside_bin), str(venv_bin)])
    }
    denied = _proc(
        303,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "worker"],
        cwd=str(tmp_path),
    )
    denied.environ.side_effect = PermissionError("environment denied")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([found, shadowed, denied]),
        Process=MagicMock(),
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [301]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_uses_install_mapping_for_retitled_hermes(
    _winp, tmp_path
):
    mapped = _proc(401, "/usr/bin/python3", "hermes", ["hermes"], cwd="/tmp")
    mapped.memory_maps.return_value = [
        SimpleNamespace(
            path=str(
                tmp_path
                / "venv"
                / "lib"
                / "python3.13"
                / "site-packages"
                / "setproctitle.cpython-313.so"
            )
        )
    ]
    unrelated = _proc(402, "/usr/bin/python3", "hermes", ["hermes"], cwd="/tmp")
    unrelated.memory_maps.return_value = [
        SimpleNamespace(path="/opt/other/lib/setproctitle.so")
    ]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([mapped, unrelated]),
        Process=MagicMock(),
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [401]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_excludes_only_self(_winp, tmp_path):
    venv_python = str(tmp_path / "venv" / "bin" / "python")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(os.getpid(), "/usr/bin/python3", "python3", [venv_python]),
                _proc(555, "/usr/bin/python3", "python3", [venv_python]),
            ]
        ),
        Process=lambda *args, **kwargs: SimpleNamespace(
            parents=lambda: [SimpleNamespace(pid=555)]
        ),
    )

    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [555]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_iteration_error_keeps_matches(_winp, tmp_path):
    venv_python = str(tmp_path / "venv" / "bin" / "python")

    def processes():
        yield _proc(501, "/usr/bin/python3", "python3", [venv_python])
        raise PermissionError("process table denied")

    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: processes(),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [501]




@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_excludes_self_and_ancestors(_winp, tmp_path):
    import os as _os

    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    parent = MagicMock()
    parent.pid = 555
    me = MagicMock()
    me.parents.return_value = [parent]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(_os.getpid(), venv_py, "python.exe"),
                _proc(555, venv_py, "hermes.exe"),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_venv_python_windows_ignores_missing_exe(_winp, tmp_path):
    venv_python = str(tmp_path / "venv" / "Scripts" / "python.exe")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [_proc(601, None, "python.exe", [venv_python, "-m", "hermes_cli.main"])]
        ),
        Process=lambda *args, **kwargs: SimpleNamespace(parents=lambda: []),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_format_venv_holders_message_explains_posix_runtime_mixing(_winp):
    message = cli_main._format_venv_python_holders_message(
        [(101, "python3", "venv/bin/python -m hermes_cli.main serve")]
    )
    assert "already-loaded modules" in message
    assert "newly-written package files" in message




# ---------------------------------------------------------------------------
# --force vs --force-venv gating of the venv-holder guard
# ---------------------------------------------------------------------------


def _update_args(**overrides):
    defaults = dict(
        gateway=False,
        check=False,
        no_backup=True,
        backup=False,
        yes=True,
        branch=None,
        force=False,
        force_venv=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_update_until_guard(args, *, is_windows=True):
    """Drive _cmd_update_impl just far enough to hit the venv-holder guard.

    Everything before the guard is stubbed; the guard firing is observed via
    SystemExit(2). The first statement AFTER the guard is
    ``git_dir = PROJECT_ROOT / ".git"`` — a PROJECT_ROOT sentinel whose
    ``__truediv__`` raises marks 'guard passed'."""

    class _PastGuard(Exception):
        pass

    class _RootSentinel:
        def __truediv__(self, _other):
            raise _PastGuard

    with patch.object(cli_main, "_is_windows", return_value=is_windows), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main,
        "_detect_venv_python_processes",
        return_value=[(101, "python.exe", "python.exe -m hermes_cli.main serve")],
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ):
        try:
            cli_main._cmd_update_impl(args, gateway_mode=False)
        except _PastGuard:
            return "past_guard"
        except SystemExit as exc:
            return f"exit_{exc.code}"
    return "returned"


@pytest.mark.parametrize(
    "force,force_venv,expected",
    [
        (False, False, "exit_2"),   # guard fires
        (True, False, "exit_2"),    # plain --force does NOT bypass the venv guard
        (False, True, "past_guard"),  # --force-venv is the explicit escape hatch
        (True, True, "past_guard"),
    ],
)
def test_venv_holder_guard_force_semantics(force, force_venv, expected, capsys):
    result = _run_update_until_guard(_update_args(force=force, force_venv=force_venv))
    assert result == expected, capsys.readouterr().out


def test_venv_holder_guard_runs_on_posix(capsys):
    result = _run_update_until_guard(
        _update_args(force=False, force_venv=False),
        is_windows=False,
    )
    assert result == "exit_2", capsys.readouterr().out
