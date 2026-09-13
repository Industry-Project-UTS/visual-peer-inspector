"""Test-only subprocess protocol fixture; never included in deployment packages."""

import argparse
import json
import os
import time
from pathlib import Path

from visual_peer_inspector.web.core import InspectionResult
from visual_peer_inspector.web.transport import read_request, write_result

parser = argparse.ArgumentParser()
parser.add_argument("--job", type=Path, required=True)
root = parser.parse_args().job
request = read_request(root)
print(json.dumps({"stage": "Finding objects"}), flush=True)
mode = os.getenv("VPI_TEST_WORKER_MODE", "complete")
if mode == "hang":
    time.sleep(30)
elif mode == "fail":
    print(json.dumps({"error": "cuda_unavailable"}), flush=True)
    raise SystemExit(1)
elif mode == "pipeline":
    from tests.test_web_pipeline import PixelModels
    from visual_peer_inspector.web.pipeline import UploadPipeline

    result = UploadPipeline(PixelModels()).inspect(
        request, lambda stage: print(json.dumps({"stage": stage}), flush=True)
    )
    result.provenance["test_only"] = True
    write_result(root, result)
else:
    result = InspectionResult(
        (),
        (),
        "insufficient_evidence",
        ("Test-only empty evidence.",),
        "test-only",
        "test-only",
        "assisted" if request.corrections is not None else "automatic",
    )
    write_result(root, result)
