"""Adapt official scene IDs to the fixed six-digit vision-worker interface.

The worker ID is transport metadata only. Region checkpoints and output rows
keep the original, sealed official scene ID.
"""
from contextlib import contextmanager
import re

from vpi_next_evaluation import native_regions


def worker_scene_id(scene_id, role):
    prefix = {'seen': '1', 'unseen': '2'}.get(role)
    if prefix is None or not isinstance(scene_id, str):
        raise ValueError('Official region role or scene ID is invalid')
    if not re.fullmatch(r'S' + prefix + r'\d{6}', scene_id):
        raise ValueError('Official scene ID does not match its sealed role')
    return 'S' + scene_id[2:]


@contextmanager
def worker_id_transport(role):
    original_proposal = native_regions.ProposalJob
    original_sam = native_regions.SamImageJob

    def proposal_job(*args, **kwargs):
        if args or 'opaque_scene_id' not in kwargs:
            raise ValueError('Unexpected LocateAnything job constructor')
        return original_proposal(**{**kwargs,
            'opaque_scene_id': worker_scene_id(kwargs['opaque_scene_id'], role)})

    def sam_job(*args, **kwargs):
        if args or 'opaque_scene_id' not in kwargs:
            raise ValueError('Unexpected SAM job constructor')
        return original_sam(**{**kwargs,
            'opaque_scene_id': worker_scene_id(kwargs['opaque_scene_id'], role)})

    native_regions.ProposalJob = proposal_job
    native_regions.SamImageJob = sam_job
    try:
        yield
    finally:
        native_regions.ProposalJob = original_proposal
        native_regions.SamImageJob = original_sam


def extract_regions(config, source_root):
    role = config.get('role')
    if role not in {'seen', 'unseen'}:
        return native_regions.extract_regions(config, source_root)
    with worker_id_transport(role):
        return native_regions.extract_regions(config, source_root)
