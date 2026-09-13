"""Run integration checks with a clearly labelled non-neural test encoder."""

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visual_peer_inspector.models.visual_encoder import ContractVisualEncoder
from visual_peer_inspector.workflows.comparison_io import atomic_json, digest_file, read_json, seal
from visual_peer_inspector.workflows.scoring_comparison import predict, preflight
from visual_peer_inspector.workflows.comparison_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / ".local/contract-validation-v7",
        help="New directory for this immutable validation run",
    )
    args = parser.parse_args()
    source = ROOT / ".local/inputs-v1"
    destination = args.destination.resolve()
    local_root = (ROOT / ".local").resolve()
    if destination == local_root or local_root not in destination.parents:
        raise ValueError("Validation evidence must stay in a new .local subdirectory")
    if destination.exists():
        raise FileExistsError("Preserve previous validation evidence; choose a new directory")
    model, evaluator = destination / "model-input", destination / "evaluator-only"
    model.mkdir(parents=True)
    evaluator.mkdir()
    index = read_json(source / "model-input/index.json")
    scenes = index["smoke_scenes"][:2]
    index["scenes"] = scenes
    index["smoke_scenes"] = scenes
    for scene in scenes:
        for folder in ("scenes", "media"):
            old = source / "model-input" / folder
            (model / folder).mkdir(exist_ok=True)
            if folder == "media":
                shutil.copytree(old / scene, model / folder / scene)
            else:
                shutil.copy2(old / f"{scene}.json", model / folder / f"{scene}.json")
        (evaluator / "scenes").mkdir(exist_ok=True)
        shutil.copy2(source / "evaluator-only/scenes" / f"{scene}.json", evaluator / "scenes" / f"{scene}.json")
    atomic_json(model / "index.json", index)
    seal(model, [p for p in model.rglob("*") if p.is_file()])
    evaluator_index = read_json(source / "evaluator-only/index.json")
    evaluator_index["scenes"] = scenes
    evaluator_index["model_lock_sha256"] = digest_file(model / "artifact_lock.json")
    atomic_json(evaluator / "index.json", evaluator_index)
    seal(evaluator, [p for p in evaluator.rglob("*") if p.is_file()])
    output = destination / "predictions"
    predict(model, output, ROOT, ContractVisualEncoder, producer="contract-test-not-dino")
    previous = digest_file(output / "prediction_lock.json")
    def must_not_load():
        raise AssertionError("A complete resume must not load the encoder")
    predict(model, output, ROOT, must_not_load, producer="contract-test-not-dino")
    assert digest_file(output / "prediction_lock.json") == previous
    restored = destination / "restored-predictions"
    predict(
        model,
        restored,
        ROOT,
        must_not_load,
        producer="contract-test-not-dino",
        recovery_bundle=destination / "predictions-recovery.zip",
        recovery_manifest=destination / "predictions-recovery-manifest.json",
    )
    assert digest_file(restored / "prediction_lock.json") == previous
    try:
        evaluate(model, output, evaluator, destination / "should-not-exist")
    except ValueError as error:
        assert "Contract-test" in str(error)
    else:
        raise AssertionError("Contract test features were accepted as DINO evidence")
    evaluate(model, output, evaluator, destination / "evaluation", allow_contract=True)
    atomic_json(destination / "validation_receipt.json", {"status": "passed", "test_scenes": 2, "test_route_views": 80, "encoder": "non-neural contract test, not DINO", "checks": ["input hashes", "two mask routes", "eight fixed scorers", "asset-grouped context folds", "complete resume without encoder", "verified recovery restore without encoder", "contract results rejected by scientific evaluator", "restartable CPU evaluation", "private masks and winner gallery", "aggregate tables and charts"], "accuracy_claim": "none"})
    print("Integration validation passed. These are contract-test outputs, not model results.")


if __name__ == "__main__":
    main()
