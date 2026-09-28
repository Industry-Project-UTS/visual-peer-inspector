"""Official ID transport leaves the sealed scene identity in region outputs."""
import pytest

from current_official_region_ids import worker_id_transport, worker_scene_id
from test_current_native_regions import package
from vpi_next_evaluation import native_regions as region
from vpi_object_bank.storage import digest_file
from visual_peer_inspector.models.sam2 import ContractSamWorker


@pytest.mark.parametrize('role,official_id', [('seen', 'S1000001'), ('unseen', 'S2002437')])
def test_worker_alias_and_official_output_identity(tmp_path, role, official_id):
    rgb, contract = package(tmp_path)
    image = rgb/'media/S000001/rgb/V00.png'
    row = {'scene': official_id, 'view': 'V00', 'rgb': f'media/{official_id}/rgb/V00.png',
           'rgb_sha256': digest_file(image), '_rgb_path': str(image)}
    binding = {'model_input_sha256': 'a'*64, 'attempt_timeout_seconds': 180}
    observed = []

    class LA:
        def run_job(self, job, attempt):
            observed.append(job.opaque_scene_id)
            return {'image_sha256': job.image_sha256,
                    'configuration_sha256': region._configs()[0].sha256,
                    'status': 'ok', 'error_type': None, 'boxes_xyxy': [[10,10,40,40]],
                    'runtime_seconds': 0.1}

    class SAM(ContractSamWorker):
        def process(self, job, attempt):
            observed.append(job.opaque_scene_id)
            return super().process(job, attempt)

    original_proposal, original_sam = region.ProposalJob, region.SamImageJob
    with worker_id_transport(role):
        proposal = region._proposal(LA(), row, contract, binding)
        result = region._region(SAM(region._configs()[1]), row, proposal, contract, binding)
    assert observed == [worker_scene_id(official_id, role)]*2
    assert result['scene'] == official_id and result['objects']
    assert region.ProposalJob is original_proposal and region.SamImageJob is original_sam


def test_aliases_are_injective_within_each_official_cohort():
    seen = {worker_scene_id(f'S1{i:06d}', 'seen') for i in range(1, 716)}
    unseen = {worker_scene_id(f'S2{i:06d}', 'unseen') for i in range(1, 2438)}
    assert len(seen) == 715 and len(unseen) == 2437
    with pytest.raises(ValueError, match='sealed role'):
        worker_scene_id('S2000001', 'seen')
    with pytest.raises(ValueError, match='sealed role'):
        worker_scene_id('S000001', 'unseen')
