#!/usr/bin/env python3
"""Explicit verified checkpoint transport; export and retain both files before a lost session."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from current_notebook_runtime import digest, safe_relative, writable_output


def export_state(state_root, bundle_path, manifest_path, *, stage, provenance):
    from visual_peer_inspector.workflows.kaggle_recovery import create_recovery_bundle
    root = Path(state_root).resolve(strict=True)
    bundle, manifest = writable_output(bundle_path), writable_output(manifest_path)
    if '/kaggle/input/' in str(root) + '/' or root == Path('/kaggle/input'):
        raise ValueError('Recovery exports cannot read mounted input resources')
    if not (root / 'run_binding.json').is_file():
        raise ValueError('Recovery requires scientific/report run binding; input packages are not exportable')
    if root in bundle.parents or root in manifest.parents:
        raise ValueError('Recovery files must be outside the exported state tree')
    if bundle.exists() or manifest.exists():
        raise ValueError('Recovery transport files already exist; choose a fresh checkpoint version')
    binding = {'stage': stage, 'run_binding_sha256': digest(root / 'run_binding.json'),
               'source_snapshot_sha256': provenance['snapshot_sha256'], 'schema': 'vpi-current-stage-recovery/v1'}
    temporary = Path(tempfile.mkdtemp(prefix='vpi-current-recovery-'))
    try:
        staging = temporary / 'state'
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError('Recovery state contains a symbolic link')
            if not path.is_file() or path.name.endswith('.partial'):
                continue
            relative = path.relative_to(root); safe_relative(relative.as_posix())
            target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        result = create_recovery_bundle(source_root=temporary, bundle_path=bundle, manifest_path=manifest,
                    binding=binding, allowed_top_levels=['state'])
        return {'status': 'recovery_export_verified', 'binding': binding, 'artifact_count': result['artifact_count'],
                'retention_requirement': 'Retain archive and sidecar; export completion is not a remote backup'}
    finally:
        shutil.rmtree(temporary)


def restore_state(bundle_path, manifest_path, destination_root, *, expected_binding):
    from visual_peer_inspector.workflows.kaggle_recovery import restore_recovery_bundle
    destination = writable_output(destination_root)
    if destination.exists():
        raise ValueError('Restore into a new directory; never overwrite existing scene checkpoints')
    result = restore_recovery_bundle(bundle_path=Path(bundle_path), manifest_path=Path(manifest_path),
                destination_root=destination, expected_binding=expected_binding, allowed_top_levels=['state'])
    restored = destination / 'state'
    if digest(restored / 'run_binding.json') != expected_binding['run_binding_sha256']:
        raise ValueError('Restored scientific binding differs')
    return {'status': 'recovery_restored_verified', 'state_directory': str(restored),
            'artifact_count': result['artifact_count'], 'resume_requires': 'Normal stage input/source/runtime and scene seal checks'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'restore'])
    parser.add_argument('--bundle', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--state-root'); parser.add_argument('--destination-root')
    parser.add_argument('--binding-json'); parser.add_argument('--stage'); parser.add_argument('--provenance-json')
    args = parser.parse_args()
    if args.action == 'export':
        result = export_state(args.state_root,args.bundle,args.manifest,stage=args.stage,
                              provenance=json.loads(Path(args.provenance_json).read_text()))
    else:
        result = restore_state(args.bundle,args.manifest,args.destination_root,
                               expected_binding=json.loads(Path(args.binding_json).read_text()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
