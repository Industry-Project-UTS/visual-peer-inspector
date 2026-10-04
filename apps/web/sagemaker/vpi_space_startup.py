"""Jupyter startup hook: detach the bounded app supervisor without delaying Studio."""
from pathlib import Path
import subprocess
import sys


def _jupyter_server_extension_points():
    return [{'module': 'vpi_space_startup'}]


def _load_jupyter_server_extension(serverapp):
    root = Path(__file__).resolve().parents[3]
    try:
        (root / 'logs').mkdir(exist_ok=True)
        with (root / 'logs/autostart.log').open('ab') as log:
            subprocess.Popen([sys.executable, str(root / 'apps/web/sagemaker/autostart.py'), 'supervise'],
                             cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        serverapp.log.info('Visual Peer Inspector startup requested; see logs/autostart-status.json.')
    except OSError:
        # The demo must never prevent the user from opening their JupyterLab Space.
        serverapp.log.warning('Visual Peer Inspector startup unavailable; JupyterLab will continue.')
