"""Prepare a password-protected loopback app, then explicitly start its fixed tunnel."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import stat
import urllib.error
import urllib.parse
import urllib.request


def read_settings(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError('Use a private regular settings file.')
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) & 0o077 or metadata.st_uid != os.getuid():
        raise ValueError('The settings file must be owned by this user and have mode 600.')
    settings = json.loads(path.read_text())
    url = settings.get('public_url', '')
    parts = urllib.parse.urlsplit(url)
    if (parts.scheme != 'https' or not parts.hostname
        or not parts.hostname.endswith(('.ngrok-free.app', '.ngrok-free.dev'))
        or parts.username or parts.password or parts.port
        or parts.path not in ('', '/') or parts.query or parts.fragment):
        raise ValueError('Use the exact HTTPS development domain assigned by ngrok.')
    username, password = settings.get('username'), settings.get('password')
    if not isinstance(username, str) or not username.strip() or '\n' in username:
        raise ValueError('Set a nonempty demo username.')
    if not isinstance(password, str) or len(password) < 4:
        raise ValueError('Set the separate demo password (at least four characters).')
    return {**settings, 'public_url': url.rstrip('/')}


def app_service(root: Path):
    spec = importlib.util.spec_from_file_location('vpi_existing_service', root / 'apps/web/sagemaker/service.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Service(root)


def verify_login(settings: dict, port: int = 7860) -> None:
    """Only use loopback; never send credentials to the external URL as a check."""
    import http.cookiejar
    jar = http.cookiejar.CookieJar()
    client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    address = f'http://127.0.0.1:{port}'
    try:
        with client.open(address + '/config', timeout=5):
            raise RuntimeError('The app is unprotected. Refusing to open a tunnel.')
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise RuntimeError('The app authentication preflight failed.') from None
    data = urllib.parse.urlencode({'username': settings['username'], 'password': settings['password']}).encode()
    with client.open(address + '/login', data=data, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError('The app login check failed.')
    with client.open(address + '/config', timeout=5) as response:
        config = json.load(response)
    if config.get('root', '').rstrip('/') != settings['public_url']:
        raise RuntimeError('The app download URL does not match the fixed external address.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start-app', 'start-tunnel'))
    parser.add_argument('--root', type=Path, default=Path('/home/sagemaker-user/visual-peer-inspector'))
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    settings = read_settings(args.settings)
    if args.action == 'check':
        print(json.dumps({'configuration_valid': True, 'public_url': settings['public_url'], 'password_configured': True, 'tunnel_started': False}))
        return
    if args.action == 'start-app':
        service = app_service(root)
        if service.owned_pid():
            raise RuntimeError('Cancel any inspection, then stop the private app with service.py before switching launch modes.')
        os.environ.update(VPI_AUTH_USERNAME=settings['username'], VPI_AUTH_PASSWORD=settings['password'], GRADIO_ROOT_PATH=settings['public_url'])
        result = service.start()
        verify_login(settings)
        print(json.dumps({**result, 'authentication_verified': True, 'tunnel_started': False}))
        return
    verify_login(settings)
    executable = root / '.runtime/ngrok/ngrok'
    agent_config = root / '.runtime/ngrok/ngrok.yml'
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError('Install the official Linux ngrok agent first.')
    if agent_config.is_symlink() or not agent_config.is_file() or stat.S_IMODE(agent_config.stat().st_mode) & 0o077:
        raise RuntimeError('The ngrok agent configuration must be a private regular file (mode 600).')
    # This is the only action that opens external access. Run after approval.
    os.execv(str(executable), [str(executable), 'http', 'http://127.0.0.1:7860', '--url', settings['public_url'], '--inspect=false', '--config', str(agent_config), '--log=stdout', '--log-format=json'])


if __name__ == '__main__':
    main()
