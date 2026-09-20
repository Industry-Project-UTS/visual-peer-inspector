"""Static and pure contract checks for the isolated CUDA FP32 pilot."""
from pathlib import Path
import json
import sys

import pytest

from vpi_next_evaluation import gpu_compatibility as gpu
from vpi_object_bank.storage import atomic_json, digest_file, seal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_cuda_profile_preserves_cpu_userspace_and_changes_only_torch_build():
    profiles = ROOT / "requirements/profiles"
    cpu = json.loads((profiles / "linux-dino-fp32-cpu.json").read_text())
    cuda = json.loads((profiles / "linux-dino-fp32-cuda126.json").read_text())
    cpu_packages = dict(cpu["runtime_contract"]["packages"])
    cuda_packages = dict(cuda["runtime_contract"]["packages"])
    assert cpu_packages.pop("torch") == "2.13.0+cpu"
    assert cuda_packages.pop("torch") == "2.13.0+cu126"
    assert cuda_packages == cpu_packages
    assert cuda["runtime_contract"]["device"] == "cuda:0"
    assert cuda["torch_index"] == "https://download.pytorch.org/whl/cu126"
    lock = profiles / cuda["requirements_lock"]
    assert digest_file(lock) == cuda["requirements_lock_sha256"]
    assert "torch==2.13.0+cu126" in lock.read_text()


def test_runtime_and_sample_selection_are_exact_and_label_free():
    assert gpu.compatible_gpu_runtime(versions=gpu.GPU_PINNED, cuda_available=True)
    assert not gpu.compatible_gpu_runtime(
        versions={**gpu.GPU_PINNED, "torch": "2.13.0+cpu"}, cuda_available=True
    )
    assert not gpu.compatible_gpu_runtime(versions=gpu.GPU_PINNED, cuda_available=False)
    scenes = [f"S{number:06d}" for number in range(100)]
    first = gpu._selected_scenes(scenes)
    assert len(first) == gpu.SAMPLE_SCENES and first == gpu._selected_scenes(reversed(scenes))
    with pytest.raises(ValueError, match="100-scene"):
        gpu._selected_scenes(scenes[:-1])
    source = Path(gpu.__file__).read_text()
    assert "ground_truth" not in source and "evaluator_root" not in source


def test_cuda_encoder_is_strict_fp32_without_autocast():
    source = Path(gpu.__file__).read_text()
    assert 'device_type="cuda", enabled=False' in source
    assert "torch.float16" not in source
    assert 'dtype=torch.float32' in source
    assert "allow_tf32 = False" in source


def test_compatibility_notebook_is_clean_gpu_only_and_label_free():
    path = ROOT / "notebooks/input-preparation/current-dino-gpu-compatibility.ipynb"
    notebook = json.loads(path.read_text())
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            compile("".join(cell["source"]), path.name, "exec")
    assert "linux-dino-fp32-cuda126.json" in code
    assert "run_gpu_compatibility" in code
    assert "evaluator_root" not in code and "ground_truth" not in code
    metadata = json.loads((path.parent / "kernel-metadata-dino-gpu-compatibility.json").read_text())
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False


def test_standalone_queue_has_only_setup_and_one_compatibility_job(tmp_path):
    from build_gpu_compatibility_queue import plan
    from current_kaggle_queue import preview
    from current_launch import build_kernels, validate_plan

    value = plan()
    assert validate_plan(value)["jobs"] == 2
    assert [job["id"] for job in value["jobs"]] == [
        "setup-dino-gpu", "dino-gpu-compatibility"
    ]
    job = value["jobs"][1]
    assert job["runtime"] == "dino-gpu" and job["input_boundary"] == "model_only"
    assert job["config"]["action"] == "gpu_compatibility"
    assert "feature_root" not in job["config"] and "feature_roots" not in job["config"]
    assert job["config"]["reference_feature_root"] == {"$resource": "reuse100"}
    built = build_kernels(value, ROOT, tmp_path / "queue")
    assert built == {"status": "prepared", "jobs": 2, "submitted": False}
    metadata = json.loads((tmp_path / "queue/dino-gpu-compatibility/kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True and metadata["enable_internet"] is False
    assert preview(tmp_path / "queue")["next_ready"] == ["setup-dino-gpu"]


def test_production_gate_requires_exact_passing_seal(tmp_path):
    model_files = {"model.safetensors": "a" * 64, "config.json": "b" * 64}
    receipt = {
        "schema": gpu.SCHEMA,
        "status": "compatible",
        "production_gpu_release_allowed": True,
        "cpu_gpu_feature_root_mixing_allowed": False,
        "region_lock_sha256": "c" * 64,
        "model_files": model_files,
        "thresholds": gpu.THRESHOLDS,
    }
    atomic_json(tmp_path / "gpu_compatibility_receipt.json", receipt)
    seal(tmp_path, [tmp_path / "gpu_compatibility_receipt.json"])
    assert gpu.require_compatible_gpu(
        tmp_path, region_hash="c" * 64, model_files=model_files
    )["status"] == "compatible"
    receipt["cpu_gpu_feature_root_mixing_allowed"] = True
    other = tmp_path / "bad"
    atomic_json(other / "gpu_compatibility_receipt.json", receipt)
    seal(other, [other / "gpu_compatibility_receipt.json"])
    with pytest.raises(ValueError, match="not covered"):
        gpu.require_compatible_gpu(other, region_hash="c" * 64, model_files=model_files)


def test_decision_signature_ignores_scores_but_preserves_order_and_membership():
    result = {
        "ranking": [
            {"score": 1.0, "members": [{"view_id": "V07", "candidate_id": "B"}]},
            {"score": 0.2, "members": [{"view_id": "V00", "candidate_id": "A"}]},
        ],
        "tracks": [
            {"members": [{"view_id": "V00", "candidate_id": "A"}]},
            {"members": [{"view_id": "V07", "candidate_id": "B"}]},
        ],
    }
    changed_scores = json.loads(json.dumps(result))
    changed_scores["ranking"][0]["score"] = 9.0
    assert gpu._decision_signature(result) == gpu._decision_signature(changed_scores)
    reversed_ranking = json.loads(json.dumps(result))
    reversed_ranking["ranking"].reverse()
    assert gpu._decision_signature(result) != gpu._decision_signature(reversed_ranking)
