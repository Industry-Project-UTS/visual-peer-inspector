"""Test-only worker: assisted cache replay raises if any vision stage is called."""
import argparse
import json
from pathlib import Path
from tests.test_web_two_mode import CountingModels
from visual_peer_inspector.web.pipeline import UploadPipeline
from visual_peer_inspector.web.transport import read_request, write_result

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=Path, required=True)
root = parser.parse_args().job
request = read_request(root)
models = CountingModels(forbidden=request.corrections is not None)
result = UploadPipeline(models).inspect(
    request, lambda stage: print(json.dumps({'stage': stage}), flush=True))
result.provenance['test_model_calls'] = models.calls
write_result(root, result)
