"""Tests for the Windows half-updated-venv hardening (July 2026 incident).

Covers three additions to ``hermes update``:

1. ``_venv_core_imports_healthy`` — the venv health probe that lets an
   "Already up to date" checkout still repair a broken dependency install.
2. ``_detect_venv_python_processes`` — the venv-interpreter process guard
   that refuses to mutate the venv while a desktop backend / stray python
   holds .pyd files mapped.
3. The commit_count == 0 repair branch wiring in ``_cmd_update_impl``.

All Windows-specific paths are exercised via ``_is_windows`` patching so
they run on any host (same approach as test_update_concurrent_quarantine).
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
def test_detect_venv_python_finds_posix_venv_launcher(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "bin" / "python")
    venv_py_versioned = str(tmp_path / "venv" / "bin" / "python3.13")
    base_py = "/usr/bin/python3.13"
    me = MagicMock()
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(101, base_py, "python3.13", [venv_py, "-m", "hermes_cli.main", "serve"]),
                _proc(102, base_py, "python3.13", [base_py, "somescript.py"]),
                _proc(103, venv_py_versioned, "python3.13"),
                _proc(
                    104,
                    str(tmp_path / "venv-other" / "bin" / "python3.13"),
                    "python3.13",
                ),
                _proc(
                    105,
                    base_py,
                    "python3.13",
                    ["venv/bin/python", "-m", "worker"],
                    cwd=str(tmp_path),
                ),
                _proc(
                    107,
                    "/usr/bin/python3.14t",
                    "python3.14t",
                    [str(tmp_path / "venv" / "bin" / "python3.14t"), "worker.py"],
                ),
                _proc(
                    109,
                    "/usr/bin/python3.14td",
                    "python3.14td",
                    [str(tmp_path / "venv" / "bin" / "python3.14td"), "worker.py"],
                ),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [m[0] for m in matches] == [101, 103, 105, 107, 109]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_matches_logical_launcher_under_canonical_root(
    _winp, tmp_path
):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    logical_root = tmp_path / "logical-root"
    logical_root.symlink_to(real_root, target_is_directory=True)
    logical_venv_python = str(
        logical_root / "venv" / "bin" / "python"
    )
    venv_versioned_python = str(real_root / "venv" / "bin" / "python3.13")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(
                    108,
                    "/usr/bin/python3",
                    "python3",
                    [logical_venv_python, "worker.py"],
                ),
                _proc(110, venv_versioned_python, "python3.13"),
            ]
        ),
        Process=MagicMock(),
    )

    # Production canonicalizes PROJECT_ROOT with Path.resolve(), while a
    # process launched through the symlink can retain the logical argv[0].
    with patch.object(cli_main, "PROJECT_ROOT", real_root), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [108, 110]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_finds_posix_dot_venv_launcher(_winp, tmp_path):
    dot_venv_python = str(tmp_path / ".venv" / "bin" / "python")
    proc = _proc(
        106,
        "/usr/bin/python3",
        "python3",
        [dot_venv_python, "-m", "hermes_cli.main", "serve"],
    )
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [106]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_ignores_non_python_mentions(_winp, tmp_path):
    venv_py = str(tmp_path / "venv" / "bin" / "python")
    me = MagicMock()
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(201, "/bin/bash", "bash", ["bash", "-c", f"{venv_py} -m worker"]),
                _proc(
                    202,
                    "/bin/sh",
                    "sh",
                    ["sh", "-c", "python -m hermes_cli.main"],
                    cwd=str(tmp_path),
                ),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_relative_argv0_requires_target_cwd(
    _winp, tmp_path
):
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(
                    202,
                    "/usr/bin/python3",
                    "python3",
                    ["venv/bin/python", "-m", "worker"],
                    cwd="",
                )
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
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("#!/bin/sh\n")
    venv_python.chmod(0o755)
    proc = _proc(
        203,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "hermes_cli.main", "serve"],
        cwd=str(tmp_path),
    )
    proc.environ.return_value = {
        "PATH": os.pathsep.join(
            [str(venv_bin), "/usr/local/bin", "/usr/bin"]
        )
    }
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [203]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_bare_argv0_path_denied_is_ignored(
    _winp, tmp_path
):
    proc = _proc(
        204,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "hermes_cli.main", "serve"],
        cwd=str(tmp_path),
    )
    proc.environ.side_effect = PermissionError("environment denied")
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_bare_argv0_honors_path_order(_winp, tmp_path):
    outside_bin = tmp_path / "outside-bin"
    venv_bin = tmp_path / "venv" / "bin"
    outside_bin.mkdir()
    venv_bin.mkdir(parents=True)
    for python_path in (outside_bin / "python", venv_bin / "python"):
        python_path.write_text("#!/bin/sh\n")
        python_path.chmod(0o755)
    proc = _proc(
        205,
        "/usr/bin/python3",
        "python3",
        ["python", "-m", "worker"],
        cwd=str(tmp_path),
    )
    proc.environ.return_value = {
        "PATH": os.pathsep.join([str(outside_bin), str(venv_bin)])
    }
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_finds_retitled_hermes_venv_map(
    _winp, tmp_path
):
    proc = _proc(
        206,
        "/usr/bin/python3",
        "hermes",
        ["hermes"],
        cwd="/tmp",
    )
    proc.environ.return_value = {"PATH": "/usr/local/bin:/usr/bin"}
    proc.memory_maps.return_value = [
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
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [match[0] for match in matches] == [206]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_ignores_unrelated_retitled_hermes_map(
    _winp, tmp_path
):
    proc = _proc(
        207,
        "/usr/bin/python3",
        "hermes",
        ["hermes"],
        cwd="/tmp",
    )
    proc.environ.return_value = {"PATH": "/usr/local/bin:/usr/bin"}
    proc.memory_maps.return_value = [
        SimpleNamespace(path="/opt/other/lib/setproctitle.so")
    ]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([proc]),
        Process=MagicMock(),
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_posix_excludes_only_self(_winp, tmp_path):
    import os as _os

    venv_py = str(tmp_path / "venv" / "bin" / "python")
    parent = MagicMock()
    parent.pid = 555
    me = MagicMock()
    me.parents.return_value = [parent]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(_os.getpid(), "/usr/bin/python3", "python3", [venv_py, "update"]),
                _proc(555, "/usr/bin/python3", "python3", [venv_py, "gateway"]),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = cli_main._detect_venv_python_processes()

    assert [m[0] for m in matches] == [555]


@patch.object(cli_main, "_is_windows", return_value=False)
def test_detect_venv_python_process_iteration_error_is_empty(_winp, tmp_path):
    def denied_process_iter(_attrs):
        raise PermissionError("process table denied")
        yield  # pragma: no cover

    fake_psutil = types.SimpleNamespace(process_iter=denied_process_iter)
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


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
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [_proc(101, None, "python.exe", [venv_py, "-m", "hermes_cli.main"])]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert cli_main._detect_venv_python_processes() == []


@patch.object(cli_main, "_is_windows", return_value=False)
def test_format_venv_holders_message_explains_posix_runtime_mixing(_winp):
    msg = cli_main._format_venv_python_holders_message(
        [(101, "python3", "venv/bin/python -m hermes_cli.main serve")]
    )
    assert "already-loaded modules" in msg
    assert "newly-written package files" in msg


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


def _run_update_until_guard(
    args,
    *,
    is_windows=True,
    detector=None,
    quiesce_token=None,
    quiesce=None,
    profile_gateways=(),
):
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

    detector_effect = detector or (
        lambda **_kw: [
            (101, "python.exe", "python.exe -m hermes_cli.main serve")
        ]
    )
    quiesce_effect = (
        quiesce if quiesce is not None else lambda _pids: quiesce_token
    )

    with patch.object(cli_main, "_is_windows", return_value=is_windows), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ), patch.object(
        cli_main,
        "_quiesce_posix_gateways_for_update",
        side_effect=quiesce_effect,
    ), patch.object(
        cli_main, "_release_posix_gateway_quiesce"
    ), patch.object(
        cli_main,
        "_detect_venv_python_processes",
        side_effect=detector_effect,
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ), patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=list(profile_gateways),
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


def test_direct_posix_update_quiesces_mapped_gateways_without_supervisor(
    monkeypatch, capsys
):
    monkeypatch.delenv("_HERMES_UPDATE_SUPERVISOR_PID", raising=False)
    seen_holders = []
    quiesce = MagicMock(
        return_value={"pids": {666}, "created_markers": []}
    )

    def detect(*, exclude_pids=None):
        seen_holders.append(exclude_pids)
        return []

    result = _run_update_until_guard(
        _update_args(force=False, force_venv=False),
        is_windows=False,
        detector=detect,
        quiesce=quiesce,
        profile_gateways=[
            SimpleNamespace(profile="default", path="/tmp/hermes", pid=666)
        ],
    )

    assert result == "past_guard", capsys.readouterr().out
    quiesce.assert_called_once_with({666})
    assert seen_holders == [{666}]


@pytest.mark.parametrize(
    "quiesce_token",
    [
        None,
        {"pids": {666}, "created_markers": []},
    ],
)
def test_direct_posix_update_aborts_when_any_mapped_gateway_cannot_quiesce(
    monkeypatch, capsys, quiesce_token
):
    monkeypatch.delenv("_HERMES_UPDATE_SUPERVISOR_PID", raising=False)
    detector = MagicMock(return_value=[])

    result = _run_update_until_guard(
        _update_args(force=False, force_venv=False),
        is_windows=False,
        detector=detector,
        quiesce=MagicMock(return_value=quiesce_token),
        profile_gateways=[
            SimpleNamespace(profile="default", path="/tmp/hermes", pid=666),
            SimpleNamespace(profile="work", path="/tmp/work", pid=777),
        ],
    )

    assert result == "exit_2"
    detector.assert_not_called()
    assert "Could not establish" in capsys.readouterr().out


def test_venv_holder_guard_excludes_explicit_supervisor(monkeypatch, capsys):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "555")
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_QUIESCED", "dashboard")
    seen = []
    supervisor = SimpleNamespace(pid=555)
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(parents=lambda: [supervisor])
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        return []

    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids", return_value=[555, 666]
    ):
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
            quiesce_token={"pids": {555, 666}, "created_markers": []},
        )

    assert result == "past_guard", capsys.readouterr().out
    assert seen == [{555, 666}]


def test_venv_holder_guard_quiesces_mapped_foreground_gateway_supervisor(
    monkeypatch, capsys
):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "555")
    seen = []
    supervisor = SimpleNamespace(pid=555)
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(parents=lambda: [supervisor])
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        return []

    # find_gateway_pids deliberately omits ancestors, matching the real
    # foreground `/update` topology. The independently validated profile PID
    # mapping must restore the gateway to the drain set before exclusion.
    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids", return_value=[]
    ):
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
            quiesce_token={"pids": {555}, "created_markers": []},
            profile_gateways=[
                SimpleNamespace(profile="default", path="/tmp/hermes", pid=555)
            ],
        )

    assert result == "past_guard", capsys.readouterr().out
    assert seen == [{555}]


def test_quiesced_dashboard_does_not_exclude_detached_venv_worker(
    monkeypatch, capsys
):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "555")
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_QUIESCED", "dashboard")
    seen = []
    supervisor = SimpleNamespace(pid=555)
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(parents=lambda: [supervisor])
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        # A profile-scoped dashboard TUI can have a slash worker or compute
        # host that intentionally called setsid(). It is not the attested
        # dashboard supervisor or a drained gateway and must still block.
        return [
            (
                777,
                "python",
                "venv/bin/python -m tui_gateway.slash_worker",
            )
        ]

    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids", return_value=[]
    ):
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
            quiesce_token=None,
        )

    assert result == "exit_2", capsys.readouterr().out
    assert seen == [{555}]


def test_venv_holder_guard_does_not_trust_unquiesced_dashboard_supervisor(
    monkeypatch, capsys
):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "555")
    monkeypatch.delenv("_HERMES_UPDATE_SUPERVISOR_QUIESCED", raising=False)
    seen = []
    supervisor = SimpleNamespace(pid=555)
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(parents=lambda: [supervisor])
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        return [(555, "python", "venv/bin/python -m hermes_cli.main serve")]

    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids", return_value=[]
    ):
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
        )

    assert result == "exit_2", capsys.readouterr().out
    assert seen == [set()]


def test_venv_holder_guard_does_not_exclude_unquiesced_gateway_supervisor(
    monkeypatch, capsys
):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "555")
    seen = []
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(
            parents=lambda: [SimpleNamespace(pid=555)]
        )
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        return [(555, "python", "venv/bin/python -m hermes_cli.main gateway run")]

    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids", return_value=[555]
    ):
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
            quiesce_token=None,
        )

    assert result == "exit_2", capsys.readouterr().out
    assert seen == [set()]


def test_venv_holder_guard_rejects_non_ancestor_supervisor(monkeypatch, capsys):
    monkeypatch.setenv("_HERMES_UPDATE_SUPERVISOR_PID", "777")
    seen = []
    fake_psutil = types.SimpleNamespace(
        Process=lambda: SimpleNamespace(
            parents=lambda: [SimpleNamespace(pid=555)]
        )
    )

    def detect(*, exclude_pids=None):
        seen.append(exclude_pids)
        return []

    with patch.dict(sys.modules, {"psutil": fake_psutil}), patch(
        "hermes_cli.gateway.find_gateway_pids"
    ) as find_gateways:
        result = _run_update_until_guard(
            _update_args(force=False, force_venv=False),
            is_windows=False,
            detector=detect,
        )

    assert result == "past_guard", capsys.readouterr().out
    assert seen == [set()]
    find_gateways.assert_not_called()


@patch.object(cli_main, "_is_windows", return_value=False)
def test_quiesce_posix_gateway_confirms_live_drain_state_before_exclusion(
    _winp, tmp_path
):
    from gateway.drain_control import drain_request_path

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    proc = SimpleNamespace(profile="jasper", path=profile_home, pid=555)
    marker = drain_request_path(profile_home)

    def read_live_state(_path):
        # The real marker write must happen before the PID becomes eligible
        # for exclusion from the process guard.
        assert marker.exists()
        return {
            "pid": 555,
            "gateway_state": "draining",
            "active_agents": 0,
        }

    with patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=[proc],
    ), patch(
        "hermes_cli.gateway._get_restart_drain_timeout",
        return_value=1,
    ), patch(
        "gateway.status.read_runtime_status",
        side_effect=read_live_state,
    ):
        token = cli_main._quiesce_posix_gateways_for_update({555})

    assert token is not None
    assert token["pids"] == {555}
    assert marker.exists()

    cli_main._release_posix_gateway_quiesce(token)

    assert not marker.exists()


@patch.object(cli_main, "_is_windows", return_value=False)
def test_quiesce_posix_gateway_refreshes_stale_drain_marker(
    _winp, tmp_path
):
    from gateway.drain_control import drain_request_path

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    proc = SimpleNamespace(profile="jasper", path=profile_home, pid=555)
    marker = drain_request_path(profile_home)
    marker.write_text('{"epoch":"stale"}', encoding="utf-8")

    def read_live_state(_path):
        assert '"principal": "hermes-update"' in marker.read_text(
            encoding="utf-8"
        )
        return {
            "pid": 555,
            "gateway_state": "draining",
            "active_agents": 0,
        }

    with patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=[proc],
    ), patch(
        "hermes_cli.gateway._get_restart_drain_timeout",
        return_value=1,
    ), patch(
        "gateway.drain_control.drain_requested",
        return_value=False,
    ), patch(
        "gateway.status.read_runtime_status",
        side_effect=read_live_state,
    ):
        token = cli_main._quiesce_posix_gateways_for_update({555})

    assert token is not None
    assert [entry["home"] for entry in token["created_markers"]] == [
        profile_home
    ]
    cli_main._release_posix_gateway_quiesce(token)
    assert not marker.exists()


@patch.object(cli_main, "_is_windows", return_value=False)
def test_quiesce_posix_gateway_rejects_active_operator_drain(
    _winp, tmp_path
):
    from gateway.drain_control import drain_request_path

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    proc = SimpleNamespace(profile="jasper", path=profile_home, pid=555)
    marker = drain_request_path(profile_home)
    marker.write_text('{"principal":"operator"}', encoding="utf-8")

    with patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=[proc],
    ), patch(
        "hermes_cli.gateway._get_restart_drain_timeout",
        return_value=1,
    ), patch(
        "gateway.drain_control.drain_requested",
        return_value=True,
    ), patch(
        "gateway.status.read_runtime_status",
        return_value={
            "pid": 555,
            "gateway_state": "draining",
            "active_agents": 0,
        },
    ):
        token = cli_main._quiesce_posix_gateways_for_update({555})

    assert token is None
    assert marker.read_text(encoding="utf-8") == '{"principal":"operator"}'


@patch.object(cli_main, "_is_windows", return_value=False)
def test_quiesce_posix_gateway_reclaims_orphaned_update_marker(
    _winp, tmp_path
):
    from gateway.drain_control import (
        read_drain_request,
        write_drain_request,
    )

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    proc = SimpleNamespace(profile="jasper", path=profile_home, pid=555)
    write_drain_request(
        principal="hermes-update",
        home=profile_home,
        request_id="orphaned-update",
        owner_pid=999999,
    )

    with patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=[proc],
    ), patch(
        "hermes_cli.gateway._get_restart_drain_timeout",
        return_value=1,
    ), patch(
        "gateway.status.get_process_start_time",
        return_value=None,
    ), patch(
        "gateway.status.read_runtime_status",
        return_value={
            "pid": 555,
            "gateway_state": "draining",
            "active_agents": 0,
        },
    ):
        token = cli_main._quiesce_posix_gateways_for_update({555})

    assert token is not None
    body = read_drain_request(home=profile_home)
    assert body is not None
    assert body["request_id"] != "orphaned-update"
    assert body["owner_pid"] == os.getpid()

    cli_main._release_posix_gateway_quiesce(token)
    assert read_drain_request(home=profile_home) is None


@patch.object(cli_main, "_is_windows", return_value=False)
def test_quiesce_posix_gateway_reclaims_reused_owner_pid(
    _winp, tmp_path
):
    from gateway.drain_control import (
        read_drain_request,
        write_drain_request,
    )

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    proc = SimpleNamespace(profile="jasper", path=profile_home, pid=555)
    write_drain_request(
        principal="hermes-update",
        home=profile_home,
        request_id="stale-owner",
        owner_pid=4242,
        owner_start_time=111,
    )

    def process_start_time(pid):
        return 222 if pid == 4242 else 333

    with patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=[proc],
    ), patch(
        "hermes_cli.gateway._get_restart_drain_timeout",
        return_value=1,
    ), patch(
        "gateway.status.get_process_start_time",
        side_effect=process_start_time,
    ), patch(
        "gateway.status.read_runtime_status",
        return_value={
            "pid": 555,
            "gateway_state": "draining",
            "active_agents": 0,
        },
    ):
        token = cli_main._quiesce_posix_gateways_for_update({555})

    assert token is not None
    body = read_drain_request(home=profile_home)
    assert body is not None
    assert body["request_id"] != "stale-owner"
    assert body["owner_pid"] == os.getpid()
    assert body["owner_start_time"] == 333

    cli_main._release_posix_gateway_quiesce(token)
    assert read_drain_request(home=profile_home) is None


@patch.object(cli_main, "_is_windows", return_value=False)
def test_quiesce_posix_gateway_rejects_live_foreign_updater(
    _winp, tmp_path
):
    from gateway.drain_control import (
        read_drain_request,
        write_drain_request,
    )

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    proc = SimpleNamespace(profile="jasper", path=profile_home, pid=555)
    original = write_drain_request(
        principal="hermes-update",
        home=profile_home,
        request_id="live-owner",
        owner_pid=4242,
        owner_start_time=111,
    )

    with patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=[proc],
    ), patch(
        "hermes_cli.gateway._get_restart_drain_timeout",
        return_value=1,
    ), patch(
        "gateway.status.get_process_start_time",
        return_value=111,
    ), patch(
        "gateway.status.read_runtime_status",
        return_value={
            "pid": 555,
            "gateway_state": "draining",
            "active_agents": 0,
        },
    ):
        token = cli_main._quiesce_posix_gateways_for_update({555})

    assert token is None
    assert read_drain_request(home=profile_home) == original


@patch.object(cli_main, "_is_windows", return_value=False)
def test_release_posix_gateway_preserves_replacement_drain(
    _winp, tmp_path
):
    from gateway.drain_control import (
        read_drain_request,
        write_drain_request,
    )

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    proc = SimpleNamespace(profile="jasper", path=profile_home, pid=555)

    with patch(
        "hermes_cli.gateway.find_profile_gateway_processes",
        return_value=[proc],
    ), patch(
        "hermes_cli.gateway._get_restart_drain_timeout",
        return_value=1,
    ), patch(
        "gateway.status.read_runtime_status",
        return_value={
            "pid": 555,
            "gateway_state": "draining",
            "active_agents": 0,
        },
    ):
        token = cli_main._quiesce_posix_gateways_for_update({555})

    assert token is not None
    write_drain_request(principal="operator", home=profile_home)

    cli_main._release_posix_gateway_quiesce(token)

    body = read_drain_request(home=profile_home)
    assert body is not None
    assert body["principal"] == "operator"


@patch.object(cli_main, "_is_windows", return_value=False)
def test_finish_posix_quiesce_retains_drain_for_live_old_gateway(
    _winp, tmp_path
):
    from gateway.drain_control import (
        drain_request_path,
        write_drain_request,
    )

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    marker = write_drain_request(
        principal="hermes-update",
        home=profile_home,
        request_id="owned",
        owner_pid=os.getpid(),
    )
    token = {
        "pids": {555},
        "process_start_times": {555: 111},
        "created_markers": [
            {"home": profile_home, "marker": marker, "pid": 555}
        ],
    }

    with patch("gateway.status._pid_exists", return_value=True), patch(
        "gateway.status.get_process_start_time", return_value=111
    ):
        surviving = cli_main._finish_posix_gateway_quiesce(token)

    assert surviving == {555}
    assert token["retain_on_exit"] is True
    assert drain_request_path(profile_home).exists()

    cli_main._release_posix_gateway_quiesce_at_exit(token)
    assert drain_request_path(profile_home).exists()


@patch.object(cli_main, "_is_windows", return_value=False)
def test_finish_posix_quiesce_releases_drain_after_pid_reuse(
    _winp, tmp_path
):
    from gateway.drain_control import (
        drain_request_path,
        write_drain_request,
    )

    profile_home = tmp_path / "profiles" / "jasper"
    profile_home.mkdir(parents=True)
    marker = write_drain_request(
        principal="hermes-update",
        home=profile_home,
        request_id="owned",
        owner_pid=os.getpid(),
    )
    token = {
        "pids": {555},
        "process_start_times": {555: 111},
        "created_markers": [
            {"home": profile_home, "marker": marker, "pid": 555}
        ],
    }

    with patch("gateway.status._pid_exists", return_value=True), patch(
        "gateway.status.get_process_start_time", return_value=222
    ):
        surviving = cli_main._finish_posix_gateway_quiesce(token)

    assert surviving == set()
    assert not drain_request_path(profile_home).exists()
