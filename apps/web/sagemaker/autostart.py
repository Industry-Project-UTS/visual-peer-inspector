"""Own one app and fixed ngrok tunnel for the lifetime of this running Space."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fixed_demo import app_service, read_settings, verify_login

ROOT = Path(__file__).resolve().parents[3]


def tunnel_pid(root, settings):
    """A stale or recycled PID never authorizes stopping another process."""
    try:
        pid = int((root / 'logs/ngrok.pid').read_text())
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        expected = str(root / '.runtime/ngrok/ngrok').encode()
        return pid if (pid > 1 and args[0] == expected and b'http://127.0.0.1:7860' in args
                       and settings['public_url'].encode() in args and b'--inspect=false' in args) else None
    except (OSError, ValueError):
        return None


def save_status(root, **fields):
    path = root / 'logs/autostart-status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'time': time.time(), 'supervisor_pid': os.getpid(), **fields}, indent=2))
    tmp.replace(path)


def ensure_services(root):
    settings = read_settings(root / '.runtime/ngrok/demo.json')
    service = app_service(root)
    with (root / 'logs/service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not service.owned_pid():
            os.environ.update(VPI_AUTH_USERNAME=settings['username'],
                              VPI_AUTH_PASSWORD=settings['password'],
                              GRADIO_ROOT_PATH=settings['public_url'])
            service.start()
        verify_login(settings)
        pid = tunnel_pid(root, settings)
        if pid is None:
            with (root / 'logs/ngrok.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, str(root / 'apps/web/sagemaker/fixed_demo.py'),
                                          'start-tunnel', '--root', str(root), '--settings',
                                          str(root / '.runtime/ngrok/demo.json')], cwd=root,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            (root / 'logs/ngrok.pid').write_text(str(child.pid))
            # The helper execs the official agent. A launch is not yet a live endpoint.
            for _ in range(20):
                if child.poll() is not None:
                    raise RuntimeError('Tunnel exited during startup. Check logs/ngrok.log.')
                if tunnel_pid(root, settings):
                    break
                time.sleep(.25)
            pid = tunnel_pid(root, settings)
            if pid is None:
                raise RuntimeError('Tunnel did not start within five seconds.')
        return {'app_pid': service.owned_pid(), 'ngrok_pid': pid,
                'authentication_verified': True, 'public_url': settings['public_url']}


def supervise(root):
    (root / 'logs').mkdir(exist_ok=True)
    with (root / 'logs/autostart.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # Jupyter can load config more than once; never duplicate services.
        (root / 'logs/autostart.pid').write_text(str(os.getpid()))
        failures = 0
        while True:
            try:
                status = ensure_services(root)
                save_status(root, state='running', **status)
                failures = 0
            except Exception as error:
                failures += 1
                # Do not copy settings, credentials, or exception payloads into logs.
                save_status(root, state='retrying' if failures < 3 else 'failed',
                            error_type=type(error).__name__, consecutive_failures=failures)
                if failures >= 3:
                    return
            time.sleep(15)


def install(root, config_dir):
    """Enable the standard Jupyter server extension in persistent user config."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / 'jupyter_server_config.py'
    marker = '# BEGIN VISUAL PEER INSPECTOR AUTOSTART'
    end = '# END VISUAL PEER INSPECTOR AUTOSTART'
    content = config.read_text() if config.exists() else ''
    if marker in content:
        start, stop = content.index(marker), content.index(end) + len(end)
        content = content[:start] + content[stop:].lstrip('\n')
    block = (f'{marker}\nimport sys\nsys.path.insert(0, {str(root / "apps/web/sagemaker")!r})\n'
             'c.ServerApp.jpserver_extensions.update({"vpi_space_startup": True})\n' + end + '\n')
    compile(content + '\n' + block, str(config), 'exec')
    if config.exists():
        backup = config.with_suffix('.py.pre-vpi')
        if not backup.exists():
            backup.write_text(config.read_text())
    tmp = config.with_suffix('.tmp')
    tmp.write_text(content.rstrip() + '\n\n' + block)
    tmp.replace(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'supervise'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'install':
        read_settings(root / '.runtime/ngrok/demo.json')
        config = install(root, Path.home() / '.jupyter')
        print(json.dumps({'installed': True, 'config': str(config), 'restart_verified': False}))
    else:
        supervise(root)


if __name__ == '__main__':
    main()
