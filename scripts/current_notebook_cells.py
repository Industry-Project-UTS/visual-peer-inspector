"""Builder-only cells: locate a verified source before scientific imports."""
import nbformat
from textwrap import indent

LOCATOR = '''from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile

# Bounded input lookup supports flat, namespaced and automatically expanded mounts.
def _children(path):
    if not path.is_dir():
        return []
    children = sorted(p for p in path.iterdir() if p.is_dir())
    if len(children) > 250:
        raise ValueError('Source discovery bound exceeded; set VPI_CODE_ROOT explicitly')
    return children

def _source_candidates():
    explicit = os.getenv('VPI_CODE_ROOT')
    if explicit:
        return [Path(explicit).expanduser()]
    mount = Path('/kaggle/input')
    roots = [Path.cwd(), *Path.cwd().parents]
    roots += [p for p in _children(mount) if p.name not in {'datasets', 'notebooks', 'models'}]
    for category, depth in (('datasets', 2), ('notebooks', 2), ('models', 4)):
        current = [mount / category]
        for _ in range(depth):
            current = [child for parent in current for child in _children(parent)]
            if len(current) > 1000:
                raise ValueError('Source mount inventory exceeded its bound')
        for parent in current:
            roots += _children(parent / 'versions') if (parent / 'versions').is_dir() else [parent]
    return [root / suffix for root in roots for suffix in ('', 'code', 'repository-files', 'source')]

def _archive_source():
    configured = os.getenv('VPI_SOURCE_ARCHIVE')
    if not configured:
        raise FileNotFoundError('Mount the prepared source package or set VPI_CODE_ROOT explicitly')
    archive = Path(configured)
    expected = os.getenv('VPI_SOURCE_ARCHIVE_SHA256')
    if not expected or archive.stat().st_size > 256 * 1024 * 1024 or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError('Explicit source archive checksum or size check failed')
    destination = Path(os.getenv('VPI_SOURCE_DEST', '/kaggle/working/current-source')).resolve()
    if Path('/kaggle').is_dir() and not destination.is_relative_to(Path('/kaggle/working')):
        raise ValueError('Extracted source must be under /kaggle/working')
    if destination.exists():
        raise ValueError('Source destination exists; verify it with VPI_CODE_ROOT instead of overwriting')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='current-source.partial-', dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            names = [i.filename for i in infos]
            if len(names) != len(set(names)) or sum(i.file_size for i in infos) > 256 * 1024 * 1024 or bundle.testzip():
                raise ValueError('Source archive inventory, size or CRC is invalid')
            for info in infos:
                name = info.filename
                path = PurePosixPath(name)
                if info.is_dir() or path.is_absolute() or '\\\\' in name or any(p in {'', '.', '..'} for p in name.split('/')) or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Unsafe source archive member')
                output = temporary / name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(bundle.read(info))
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination

_matches = sorted({p.resolve() for p in _source_candidates() if (p / 'prepared_source_manifest.json').is_file()})
if len(_matches) > 1:
    raise ValueError('Multiple prepared source roots found; set VPI_CODE_ROOT explicitly')
CODE_ROOT = _matches[0] if _matches else _archive_source()
_manifest = json.loads((CODE_ROOT / 'prepared_source_manifest.json').read_text())
# Verify the bootstrap itself before executing any code from the mounted package.
_helper = CODE_ROOT / 'scripts/current_notebook_runtime.py'
if _helper.is_symlink() or hashlib.sha256(_helper.read_bytes()).hexdigest() != _manifest.get('files', {}).get('scripts/current_notebook_runtime.py'):
    raise ValueError('Prepared bootstrap source checksum differs')
_spec = importlib.util.spec_from_file_location('current_notebook_runtime', _helper)
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)
BOOTSTRAP = _runtime.bootstrap(CODE_ROOT, NOTEBOOK_RELATIVE, runtime_profile=RUNTIME_PROFILE)
CODE_PROVENANCE = BOOTSTRAP['provenance']
ROOT = CODE_ROOT
# Scientific imports are allowed only after the exact runtime/source preflight.
from IPython.display import display, Image
import pandas as pd
display({'source': CODE_PROVENANCE, 'runtime': BOOTSTRAP['runtime'],
         'preparation_only': 'Verified source/runtime does not establish protected-resource readiness'})
'''


def bootstrap_cell(notebook_relative, runtime_profile=None):
    guarded = "from pathlib import Path\nimport os\nimport json\nRUNTIME_PROFILE = " + repr(runtime_profile) + "\nNOTEBOOK_RELATIVE = " + repr(notebook_relative) + "\ntry:\n" + indent(LOCATOR.strip(), '    ')
    guarded += "\nexcept BaseException as _bootstrap_error:\n    _failure_root = Path('/kaggle/working/runtime-receipts') if Path('/kaggle').is_dir() else Path.cwd()/'.local/runtime-receipts'\n    _failure_root.mkdir(parents=True, exist_ok=True)\n    _failure = {'status':'bootstrap_failed','error_type':type(_bootstrap_error).__name__,'scientific_run_started':False}\n    _temporary = _failure_root/'bootstrap-failure.json.partial'\n    _temporary.write_text(json.dumps(_failure,sort_keys=True)+'\\n')\n    os.replace(_temporary,_failure_root/'bootstrap-failure.json')\n    raise"
    return nbformat.v4.new_code_cell(guarded)
