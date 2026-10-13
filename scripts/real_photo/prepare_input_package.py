"""Rebuild the frozen batch from byte-identical resized photos and Git source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / 'reports/real-photo-evaluation/input-manifest.json'
BINDING = 'c9e3b280fbccf6dc0fbe0b0972b52a936efb038733b999a35951fe72eaaf9d7a'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-root', type=Path, required=True, help='Directory containing S001/v01.jpg through S029/v08.jpg')
    parser.add_argument('--output', type=Path, help='New output directory for payload/ and batch.zip')
    parser.add_argument('--verify-only', action='store_true', help='Validate input bytes and frozen source without writing a package')
    args = parser.parse_args()
    if not args.verify_only and args.output is None:
        parser.error('--output is required unless --verify-only is selected')
    assert sha(MANIFEST) == BINDING, 'The evaluated input manifest changed'
    manifest = json.loads(MANIFEST.read_text())
    assert (manifest['scene_count'], manifest['image_count'], manifest['case_count']) == (29, 231, 58)
    expected = {Path(relative).relative_to('rgb').as_posix(): expected for relative, expected in manifest['image_files'].items()}
    actual = {p.relative_to(args.rgb_root).as_posix() for p in args.rgb_root.rglob('*') if p.is_file() and p.name != '.DS_Store'}
    assert actual == set(expected), 'Missing, extra or differently named resized photos'
    for relative, expected_hash in expected.items():
        path = args.rgb_root / relative
        assert not path.is_symlink() and sha(path) == expected_hash, relative
    runner = REPO / 'scripts/real_photo/run_batch.py'
    assert sha(runner) == manifest['runner_sha256']
    frozen_source = {}
    for relative, expected_hash in manifest['source_files'].items():
        path = Path(relative)
        assert not path.is_absolute() and '..' not in path.parts
        content = subprocess.check_output(['git', 'show', manifest['source_commit'] + ':' + relative], cwd=REPO)
        assert hashlib.sha256(content).hexdigest() == expected_hash, relative
        frozen_source[relative] = content
    if args.verify_only:
        print('Verified 231 exact resized photos, the runner and 158 frozen source files; no files written.')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    payload = args.output / 'payload'
    payload.mkdir()
    shutil.copy2(MANIFEST, payload / 'batch-manifest.json')
    shutil.copy2(runner, payload / 'run_batch.py')
    for relative in manifest['image_files']:
        destination = payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.rgb_root / Path(relative).relative_to('rgb'), destination)
    for relative, content in frozen_source.items():
        destination = payload / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    archive_path = args.output / 'batch.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(payload).as_posix())
    receipt = {'manifest_sha256': BINDING, 'archive_sha256': sha(archive_path),
        'archive_bytes': archive_path.stat().st_size, 'source_commit': manifest['source_commit'],
        'scene_count': 29, 'image_count': 231, 'case_count': 58,
        'note': 'ZIP timestamps can change the archive hash; evaluated input, runner and source bytes must match the frozen manifest.'}
    (args.output / 'package-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
