"""Keep dependency failures fatal, with one verified decord wheel-tag exception."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

KNOWN_WARNING = 'decord 0.6.0 is not supported on this platform'


def accepted_result(returncode, output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return returncode == 0 or (returncode == 1 and lines == [KNOWN_WARNING])


def main():
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=60)
    output = result.stdout + result.stderr
    if not accepted_result(result.returncode, output):
        raise RuntimeError(output.strip() or 'Dependency check failed')
    exception = None
    if result.returncode:
        distribution = importlib.metadata.distribution('decord')
        wheel = distribution.read_text('WHEEL') or ''
        assert distribution.version == '0.6.0'
        assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
        assert 'Tag: cp36-cp36m-manylinux2010_x86_64' in wheel
        import decord
        assert decord.__version__ == '0.6.0'
        # The ctypes backend actually loads libdecord here; no decoding is used
        # by the RGB-only app. No other pip conflict is accepted.
        from decord._ffi.base import _LIB
        assert _LIB is not None
        exception = {'package': 'decord', 'version': '0.6.0', 'warning': KNOWN_WARNING,
                     'wheel_metadata': wheel, 'native_library_loaded': True,
                     'scope': 'RGB-only app; video decoding not verified'}
    receipt = {'dependency_check': 'passed' if exception is None else 'passed_with_recorded_wheel_tag_exception',
               'exception': exception}
    Path('logs/environment-check.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
