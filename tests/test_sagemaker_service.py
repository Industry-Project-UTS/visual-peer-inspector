"""Service control uses disposable processes and ports, never hosted state."""
import importlib.util
from pathlib import Path
import socket
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sagemaker_service", ROOT / "apps/web/sagemaker/service.py")
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)


def test_existing_owned_server_is_not_launched_twice(tmp_path, monkeypatch):
    service = SERVICE.Service(tmp_path)
    monkeypatch.setattr(service, "owned_pid", lambda: 123)
    monkeypatch.setattr(service, "healthy", lambda: True)
    launch = Mock(side_effect=AssertionError("Duplicate server"))
    monkeypatch.setattr(SERVICE.subprocess, "Popen", launch)
    assert service.start() == {"running": True, "pid": 123, "http_ready": True,
                              "model_inference_verified": False, "already_running": True}
    launch.assert_not_called()


def test_foreign_listener_is_preserved(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        service = SERVICE.Service(tmp_path, listener.getsockname()[1])
        monkeypatch.setattr(service, "owned_pid", lambda: None)
        launch = Mock(side_effect=AssertionError("Must not launch"))
        monkeypatch.setattr(SERVICE.subprocess, "Popen", launch)
        with pytest.raises(OSError):
            service.start()
        launch.assert_not_called()


@pytest.mark.parametrize("content", ["bad", "1", "-20", "999999999"])
def test_invalid_or_stale_pid_cannot_kill_a_process(tmp_path, monkeypatch, content):
    service = SERVICE.Service(tmp_path)
    service.pid_file.write_text(content)
    kill = Mock(side_effect=AssertionError("Foreign process"))
    monkeypatch.setattr(SERVICE.os, "kill", kill)
    assert service.stop()["stopped"] is False
    kill.assert_not_called()


def test_failed_start_keeps_diagnostic_pid_and_log(tmp_path, monkeypatch):
    launch = tmp_path / "apps/web/sagemaker/launch_sagemaker.sh"
    launch.parent.mkdir(parents=True)
    launch.write_text("exit 1\n")
    service = SERVICE.Service(tmp_path, 0)
    monkeypatch.setattr(service, "owned_pid", lambda: None)
    process = Mock(pid=98765)
    process.poll.return_value = 1
    monkeypatch.setattr(SERVICE.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="exited during startup"):
        service.start()
    assert service.pid_file.read_text().strip() == "98765"
    assert (service.logs / "app.log").exists()


def test_active_inspection_cannot_be_stopped(tmp_path, monkeypatch):
    service = SERVICE.Service(tmp_path)
    monkeypatch.setattr(service, "owned_pid", lambda: 123)
    proc = tmp_path / "proc"
    children = proc / "123/task/123/children"
    children.parent.mkdir(parents=True)
    children.write_text("456")
    command = proc / "456/cmdline"
    command.parent.mkdir()
    command.write_bytes(b"python\0-m\0visual_peer_inspector.web.worker\0")
    real_path = Path
    monkeypatch.setattr(SERVICE, "Path", lambda p: proc / str(p).removeprefix("/proc/")
                        if str(p).startswith("/proc/") else real_path(p))
    kill = Mock(side_effect=AssertionError("Active worker must be preserved"))
    monkeypatch.setattr(SERVICE.os, "kill", kill)
    with pytest.raises(RuntimeError, match="Cancel it in the app"):
        service.stop()
    kill.assert_not_called()


def test_dependency_exception_is_narrow():
    spec = importlib.util.spec_from_file_location("sagemaker_environment", ROOT / "apps/web/sagemaker/check_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    warning = module.KNOWN_WARNING
    assert module.accepted_result(1, warning)
    assert module.accepted_result(0, "No broken requirements found.")
    assert not module.accepted_result(1, warning + "\nother dependency failure")
    assert not module.accepted_result(2, warning)
    assert not module.accepted_result(1, "")
