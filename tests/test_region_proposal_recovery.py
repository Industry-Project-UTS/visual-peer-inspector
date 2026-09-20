"""Recovery transport tests use fixture workers only; no neural execution."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from test_current_native_regions import package
from vpi_next_evaluation import native_regions as region
from vpi_object_bank.storage import read_json
from visual_peer_inspector.models.sam2 import ContractSamWorker


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "region_proposal_recovery", ROOT / "scripts/region_proposal_recovery.py"
)
recovery = module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    rgb, contract = package(tmp_path)
    monkeypatch.setattr(
        region, "preparation_gates", lambda cfg, source: (contract, {"fixture_only": True})
    )
    cfg = {
        "role": "development",
        "rgb_root": rgb,
        "output_root": tmp_path / "failed",
        "minimum_free_bytes": 1,
        "maximum_new_views": 2,
    }
    calls = {"proposals": 0, "regions": 0}
    fail_sam = {"value": True}

    class LA:
        def run_job(self, job, attempt):
            calls["proposals"] += 1
            return {
                "image_sha256": job.image_sha256,
                "configuration_sha256": region._configs()[0].sha256,
                "status": "ok",
                "error_type": None,
                "boxes_xyxy": [[10, 10, 40, 40]],
                "runtime_seconds": 0.1,
            }

    class SAM(ContractSamWorker):
        def process(self, job, attempt):
            calls["regions"] += 1
            if fail_sam["value"]:
                raise PermissionError("fixture")
            return super().process(job, attempt)

    def factory(phase, config, source):
        return LA() if phase == "proposals" else SAM(region._configs()[1])

    with pytest.raises(PermissionError):
        region.extract_regions(
            cfg,
            ROOT,
            worker_factory=factory,
            model_verifier=lambda *args: {"fixture": "no weights loaded"},
        )
    assert calls == {"proposals": 2, "regions": 1}
    binding = region._preflight(cfg, ROOT)[3]
    return cfg, binding, calls, fail_sam, factory


def test_sealed_recovery_seeds_exact_binding_and_skips_locateanything(interrupted, tmp_path):
    cfg, binding, calls, fail_sam, factory = interrupted
    package_root = tmp_path / "proposal-recovery"
    exported = recovery.export_recovery(cfg["output_root"], package_root, 2)
    assert exported["completed_proposal_views"] == 2

    output = tmp_path / "resumed"
    receipt = recovery.seed_recovery(
        package_root,
        output,
        exported["proposal_recovery_lock_sha256"],
        binding,
    )
    assert receipt["seeded_views_this_call"] == 2
    assert receipt["locateanything_calls"] == 0
    assert read_json(output / "run_binding.json") == binding
    for view in ("V00", "V01"):
        assert read_json(output / f"checkpoints/proposals/S000001/{view}/proposal.json") == read_json(
            cfg["output_root"] / f"checkpoints/proposals/S000001/{view}/proposal.json"
        )

    fail_sam["value"] = False
    result = region.extract_regions(
        {**cfg, "output_root": output},
        ROOT,
        worker_factory=factory,
        model_verifier=lambda *args: {"fixture": "no weights loaded"},
    )
    assert result["completed_views"] == 2
    assert calls == {"proposals": 2, "regions": 3}


def test_recovery_is_idempotent_and_fails_closed_before_target_copy(interrupted, tmp_path):
    cfg, binding, calls, _, _ = interrupted
    package_root = tmp_path / "proposal-recovery"
    with pytest.raises(ValueError, match="incomplete"):
        recovery.export_recovery(cfg["output_root"], package_root, 20)
    assert not package_root.exists()
    exported = recovery.export_recovery(cfg["output_root"], package_root, 2)
    output = tmp_path / "resumed"
    recovery.seed_recovery(
        package_root, output, exported["proposal_recovery_lock_sha256"], binding
    )
    second = recovery.seed_recovery(
        package_root, output, exported["proposal_recovery_lock_sha256"], binding
    )
    assert second["seeded_views_this_call"] == 0
    assert second["already_seeded_views"] == 2
    assert second["locateanything_calls"] == 0

    wrong = {**binding, "attempt_timeout_seconds": 50}
    rejected = tmp_path / "wrong-binding"
    with pytest.raises(ValueError, match="input, model, configuration, code or partition"):
        recovery.seed_recovery(
            package_root,
            rejected,
            exported["proposal_recovery_lock_sha256"],
            wrong,
        )
    assert wrong["attempt_timeout_seconds"] == 50
    assert not rejected.exists()

    corrupt = package_root / "checkpoints/proposals/S000001/V00/proposal.json"
    corrupt.write_text("{}")
    rejected = tmp_path / "corrupt-package"
    with pytest.raises(ValueError):
        recovery.seed_recovery(
            package_root,
            rejected,
            exported["proposal_recovery_lock_sha256"],
            binding,
        )
    assert not rejected.exists()
    assert calls == {"proposals": 2, "regions": 1}
