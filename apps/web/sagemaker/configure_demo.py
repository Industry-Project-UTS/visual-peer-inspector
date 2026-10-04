"""Prompt privately for the demo login; never include credentials in shell history."""
import argparse
import getpass
import json
import os
from pathlib import Path
import tempfile

from fixed_demo import read_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Assigned HTTPS ngrok development domain')
    parser.add_argument('--username', default='demo')
    parser.add_argument('--output', type=Path, default=Path('.runtime/ngrok/demo.json'))
    args = parser.parse_args()
    password = getpass.getpass('Demo password: ')
    repeated = getpass.getpass('Repeat demo password: ')
    if password != repeated:
        raise SystemExit('Passwords must match. No settings changed.')
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix='.demo-', dir=args.output.parent)
    candidate = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump({'public_url': args.url, 'username': args.username, 'password': password}, stream)
        read_settings(candidate)
        candidate.replace(args.output)
    finally:
        candidate.unlink(missing_ok=True)
    print('Demo login configured')


if __name__ == '__main__':
    main()
