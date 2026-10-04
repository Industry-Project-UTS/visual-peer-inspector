from pathlib import Path
import sys
from unittest.mock import Mock

from traitlets.config.loader import PyFileConfigLoader

DIRECTORY = Path(__file__).resolve().parents[1] / 'apps/web/sagemaker'
sys.path.insert(0, str(DIRECTORY))
import autostart
import vpi_space_startup


def test_persistent_config_is_idempotent_and_preserves_existing_settings(tmp_path):
    config = tmp_path / 'jupyter_server_config.py'
    config.write_text('c.ServerApp.port = 8888\n')
    autostart.install(DIRECTORY.parents[2], tmp_path)
    first = config.read_text()
    autostart.install(DIRECTORY.parents[2], tmp_path)
    assert config.read_text() == first
    loaded = PyFileConfigLoader(config.name, str(tmp_path)).load_config()
    assert loaded.ServerApp.port == 8888
    assert loaded.ServerApp.jpserver_extensions.get_value({}) == {'vpi_space_startup': True}
    assert config.with_suffix('.py.pre-vpi').read_text() == 'c.ServerApp.port = 8888\n'


def test_startup_failure_does_not_prevent_jupyter_starting(monkeypatch, tmp_path):
    monkeypatch.setattr(vpi_space_startup, "__file__", str(tmp_path / "apps/web/sagemaker/vpi_space_startup.py"))
    monkeypatch.setattr(vpi_space_startup.subprocess, 'Popen', Mock(side_effect=OSError))
    server = Mock()
    vpi_space_startup._load_jupyter_server_extension(server)
    server.log.warning.assert_called_once()


def test_recycled_or_missing_tunnel_pid_is_not_accepted(tmp_path):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/ngrok.pid').write_text(str(__import__('os').getpid()))
    assert autostart.tunnel_pid(tmp_path, {'public_url':'https://example.ngrok-free.dev'}) is None
