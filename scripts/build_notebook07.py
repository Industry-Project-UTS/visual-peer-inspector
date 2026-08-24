#!/usr/bin/env python3
"""Build the clean Notebook 07 source from reviewed Markdown and code cells."""

from __future__ import annotations

from pathlib import Path

import nbformat


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "notebooks"
    / "representation-and-scoring"
    / "toysad8k-ds0-peer-scores"
    / "07-toysad8k-ds0-representation-and-peer-scores.ipynb"
)


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


cells = [
    markdown(
        """
# ToysAD-8K DS0 Representation and Peer Scores

**Notebook file:** `07-toysad8k-ds0-representation-and-peer-scores.ipynb`

This notebook tests the complete frozen-representation and peer-scoring stage on DS0. It uses all 20 scenes, all 20 views, and the three frozen mask routes from Notebook 06.

The main questions are:

1. Can the same object preparation work reliably for every encoder?
2. Do B0, B1, and P1 return stable, label-free rankings?
3. How much do encoder choice and mask route change downstream ranking?
4. What should move to the 100-scene DS1 encoder-selection run?

DS0 is an engineering and method-validation cohort. It is not large enough to choose the final encoder.
"""
    ),
    markdown(
        """
## Experiment map and claim boundary

The notebook has two separate stages.

- **Model-facing stage:** RGB images, masks, crop transforms, frozen encoders, and B0/B1/P1 scores. No anomaly labels, categories, asset groups, or evaluator crosswalks are opened here.
- **Evaluator-only stage:** the completed label-free score package is hash-locked first. Only then are private labels and crosswalks opened to calculate ranking metrics.

The DINO comparison in this notebook is provisional. The final model choice will use the DS1 100-scene supplied-mask screen. If DS1 does not show a clear and efficient gain, DINOv2-small remains the anchor.
"""
    ),
    markdown(
        """
## Encoder screen

The screen compares DINOv2-small, DINOv2-small with registers, DINOv2-base, DINOv3-small, DINOv3-small-plus, and a ResNet50 control. DINOv2-base is a capacity check, not the default deployment choice. DINOv3-base is optional because it adds another gated capacity run without answering the first lightweight-model question.

DINOv3 access is gated by Meta's licence on Hugging Face. This notebook never accepts that licence automatically. If the account has not accepted it or the `HF_TOKEN` Kaggle Secret is unavailable, DINOv3 is recorded as unavailable and the DINOv2 run continues.

Every encoder receives the same 224 by 224 masked crop. Patch grids may differ by model, so P1 heatmaps are projected through each model's recorded patch grid rather than compared as raw arrays.

The anchor runs on all three mask routes. Challenger encoders run on supplied masks during the screen. This isolates representation quality and avoids repeating the same downstream route experiment for every model. After the encoder is frozen on DS1, the selected encoder will run on all routes.
"""
    ),
    code(
        """
from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from IPython.display import Image as DisplayImage, display
import pandas as pd

NOTEBOOK_NAME = "07-toysad8k-ds0-representation-and-peer-scores.ipynb"
EXECUTION_ENVIRONMENT = (
    "kaggle" if Path("/kaggle/working").is_dir() else "local"
)


def find_code_root() -> Path:
    # Locate the attached repository snapshot without relying on one Kaggle slug.
    configured = os.getenv("VPI_CODE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path.parents[3] for path in kaggle_input.rglob("visual_encoder.py"))
    for candidate in candidates:
        marker = candidate / "src" / "visual_peer_inspector" / "models" / "visual_encoder.py"
        if marker.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the Notebook 07 code snapshot.")


CODE_ROOT = find_code_root()
sys.path.insert(0, str(CODE_ROOT / "src"))
print(f"Environment: {EXECUTION_ENVIRONMENT}")
print(f"Notebook: {NOTEBOOK_NAME}")
print(f"Code snapshot folder: {CODE_ROOT.name}")
"""
    ),
    markdown(
        """
## Runtime dependencies

Kaggle already provides PyTorch. The setup cell installs only the reviewed notebook dependencies that are missing. The installed versions are recorded in the run outputs. Model weights are downloaded from their official Hugging Face repositories and are never written into GitHub outputs.
"""
    ),
    code(
        """
if EXECUTION_ENVIRONMENT == "kaggle":
    # Avoid replacing Kaggle's CUDA-enabled PyTorch build.
    required = {
        "transformers": "transformers==5.14.1",
        "huggingface_hub": "huggingface-hub==1.26.1",
        "matplotlib": "matplotlib==3.10.8",
        "clearml": "clearml==2.1.10",
    }
    missing = [requirement for module, requirement in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import matplotlib
import numpy as np
torch = None
transformers = None
if importlib.util.find_spec("torch") is not None:
    import torch
if importlib.util.find_spec("transformers") is not None:
    import transformers

runtime_versions = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "torch": "not installed" if torch is None else torch.__version__,
    "transformers": "not installed" if transformers is None else transformers.__version__,
    "matplotlib": matplotlib.__version__,
}
display(pd.DataFrame(runtime_versions.items(), columns=["package", "version"]))
"""
    ),
    markdown(
        """
## Configuration and private Secrets

The normal Kaggle run uses one GPU. A second T4 does not help unless the notebook starts independent model workers, which this reproducible version does not do.

`HF_TOKEN` is used only for gated DINOv3 access. ClearML is disabled during the GPU run by default. After the output is downloaded and verified, the aggregate-only finaliser can publish the same evidence from the local ClearML configuration. This prevents tracking from delaying inference.
"""
    ),
    code(
        """
def read_kaggle_secret(name: str) -> str | None:
    if EXECUTION_ENVIRONMENT != "kaggle":
        return os.environ.get(name)
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        return value if value else None
    except Exception:
        return None


RUN_MODE = os.getenv("VPI_RUN_MODE", "model_evidence")
if RUN_MODE not in {"model_evidence", "contract_test"}:
    raise ValueError("VPI_RUN_MODE must be model_evidence or contract_test")

default_encoders = [
    "dinov2_small",
    "dinov2_small_registers",
    "dinov3_small",
    "dinov3_small_plus",
    "dinov2_base",
    "resnet50_control",
]
if os.getenv("VPI_RUN_CAPACITY_CONFIRMATION", "0") == "1":
    default_encoders.append("dinov3_base")
ENCODER_KEYS = (
    ["contract_encoder"]
    if RUN_MODE == "contract_test"
    else [value.strip() for value in os.getenv("VPI_ENCODERS", ",".join(default_encoders)).split(",") if value.strip()]
)
DEVICE = (
    "cuda"
    if torch is not None and torch.cuda.is_available() and RUN_MODE == "model_evidence"
    else "cpu"
)
HF_TOKEN = read_kaggle_secret("HF_TOKEN")

if RUN_MODE == "model_evidence" and (torch is None or transformers is None):
    raise RuntimeError("The model-evidence run requires torch and transformers.")
if RUN_MODE == "model_evidence" and DEVICE != "cuda":
    raise RuntimeError("The model-evidence run requires a Kaggle GPU.")

configuration_table = pd.DataFrame(
    {
        "setting": ["run mode", "device", "encoders", "DINOv3 token available"],
        "value": [RUN_MODE, DEVICE, ", ".join(ENCODER_KEYS), bool(HF_TOKEN)],
    }
)
display(configuration_table)
"""
    ),
    markdown(
        """
## Locate and verify the three inputs

The path resolver looks for package records, not account-specific Kaggle folder names. It requires:

- the Notebook 03 opaque RGB package;
- the Notebook 06 label-free mask-route package; and
- the evaluator-only DS0 package.

Every RGB hash and every mask package checksum is checked before inference.
"""
    ),
    code(
        """
def locate_package(variable: str, predicate) -> Path:
    configured = os.getenv(variable)
    candidates = [Path(configured).expanduser()] if configured else []
    if Path("/kaggle/input").is_dir():
        candidates.extend(path.parent for path in Path("/kaggle/input").rglob("package_record.json"))
    for candidate in candidates:
        record_path = candidate / "package_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if predicate(record):
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate required package: {variable}")


RGB_ROOT = locate_package(
    "VPI_RGB_ROOT",
    lambda record: record.get("release_id") == "toysad8k-train-2000-v2"
    and record.get("image_count") == 400,
)
MASK_ROUTE_ROOT = locate_package(
    "VPI_MASK_ROUTE_ROOT",
    lambda record: record.get("package_kind") == "ds0-model-safe-mask-routes",
)
EVALUATOR_ROOT = locate_package(
    "VPI_EVALUATOR_ROOT",
    lambda record: record.get("package_kind") == "ds0-sam-mask-evaluator",
)
default_output = (
    Path("/kaggle/working/toysad8k_ds0_representation_peer_scores")
    if EXECUTION_ENVIRONMENT == "kaggle"
    else CODE_ROOT / ".local" / "runs" / "notebook07"
)
OUTPUT_ROOT = Path(os.getenv("VPI_OUTPUT_ROOT", str(default_output))).expanduser().resolve()

from visual_peer_inspector.data.representation_package import load_representation_candidates

candidate_groups, input_audit = load_representation_candidates(RGB_ROOT, MASK_ROUTE_ROOT)
display(pd.DataFrame([input_audit.to_dict()]))
print(f"RGB package folder: {RGB_ROOT.name}")
print(f"Mask-route package folder: {MASK_ROUTE_ROOT.name}")
print(f"Evaluator package folder: {EVALUATOR_ROOT.name}")
print(f"Output folder: {OUTPUT_ROOT}")
"""
    ),
    markdown(
        """
## Coverage before scoring

Peer comparison requires at least three candidates in a view. The supplied and reference routes have complete object coverage. The automatic route contains a small number of one-object or two-object views. Those views are recorded as `insufficient_peers`; they are not dropped and they are not given invented scores.
"""
    ),
    code(
        """
coverage_rows = []
for route in ("supplied_mask", "reference_box", "automatic_box"):
    counts = [len(values) for (_scene, _view, current_route), values in candidate_groups.items() if current_route == route]
    coverage_rows.append(
        {
            "route": route,
            "images": len(counts),
            "candidates": sum(counts),
            "minimum candidates": min(counts),
            "maximum candidates": max(counts),
            "rankable images": sum(value >= 3 for value in counts),
            "insufficient-peer images": sum(value < 3 for value in counts),
        }
    )
display(pd.DataFrame(coverage_rows))
"""
    ),
    markdown(
        """
## Common crop preparation

Each route uses its own frozen mask, but every encoder sees the same geometric preparation:

1. keep only the masked object;
2. replace background with RGB `(238, 241, 245)`;
3. add 15 percent square padding;
4. resize once to 224 by 224; and
5. mark patches as valid only when foreground covers at least 50 percent of the patch.

The panel below is diagnostic evidence. It is private because it contains RGB objects and route masks.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt
from PIL import Image

from visual_peer_inspector.representation.crops import CropConfiguration, prepare_masked_crop

crop_configuration = CropConfiguration()
first_scene, first_view = sorted({(scene, view) for scene, view, _route in candidate_groups})[0]
figure, axes = plt.subplots(3, 3, figsize=(8, 8))
for row_index, route in enumerate(("supplied_mask", "reference_box", "automatic_box")):
    examples = candidate_groups[(first_scene, first_view, route)][:3]
    rgb = Image.open(examples[0].image_path).convert("RGB")
    for column_index, candidate in enumerate(examples):
        prepared = prepare_masked_crop(rgb, candidate.mask_rle, configuration=crop_configuration)
        axes[row_index, column_index].imshow(prepared.rgb)
        axes[row_index, column_index].set_title(f"{route}\\n{candidate.prompt_key}")
        axes[row_index, column_index].axis("off")
figure.suptitle("Common prepared crops across the three routes")
figure.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## Model registry and decision roles

The lightweight candidates form the core screen. Base models are confirmation checks only. ResNet50 provides a non-ViT global-feature control and does not enter P1.

The notebook resolves each Hugging Face model to an immutable revision before loading it. Register tokens are removed before spatial patch scoring because they do not correspond to image locations.
"""
    ),
    code(
        """
from visual_peer_inspector.models.visual_encoder import encoder_registry_rows

registry = pd.DataFrame(encoder_registry_rows())
display(
    registry[
        ["key", "family", "model_id", "screen_stage", "parameters_millions", "supports_patches", "gated"]
    ]
)
"""
    ),
    markdown(
        """
## B0, B1, and P1 definitions

- **B0:** cosine distance from one object's global feature to the normalized leave-one-out peer mean.
- **B1:** median cosine distance from one object's global feature to every other object in the view.
- **P1:** local patch mismatch. Each valid patch finds its nearest peer patch. The largest 10 percent of directed distances form a symmetric pair distance, and the object score is the median across peers.

P1 heatmaps are coarse feature-mismatch evidence. They are not segmentation masks and must not be described as exact defect boundaries.

Shape and layout scores are reported separately. Notebook 07 does not tune an appearance-geometry fusion weight.
"""
    ),
    markdown(
        """
## Stage 1: frozen feature extraction and label-free scores

The workflow checks one batch twice before the full run. It records feature dimensions, patch grids, repeatability, runtime, peak GPU memory, and private cache size. Results are saved scene by scene, so a stopped run can resume completed scene shards.

This cell does not open the evaluator package.
"""
    ),
    code(
        """
from visual_peer_inspector.workflows.representation_ds0 import (
    run_label_free_ds0,
    verify_artifact_manifest,
)

label_free_record_path = OUTPUT_ROOT / "safe" / "label_free_run_record.json"
if label_free_record_path.is_file():
    verify_artifact_manifest(OUTPUT_ROOT / "model_input")
    label_free_record = json.loads(label_free_record_path.read_text(encoding="utf-8"))
    print("Reused the verified label-free score package.")
else:
    label_free_record = run_label_free_ds0(
        rgb_root=RGB_ROOT,
        mask_route_root=MASK_ROUTE_ROOT,
        output_root=OUTPUT_ROOT,
        encoder_keys=ENCODER_KEYS,
        device=DEVICE,
        hf_token=HF_TOKEN,
        run_mode=RUN_MODE,
    )
display(pd.DataFrame([label_free_record]))
"""
    ),
    markdown(
        """
## Encoder availability and repeatability

Unavailable gated models remain visible in this table with a short reason. They are never silently replaced. Missing DINOv3 access does not invalidate the DINOv2 engineering run, but it leaves the DINOv3 comparison incomplete. A model must return finite, unit-normalized global features and pass the repeated-batch check before its scene shards are accepted.
"""
    ),
    code(
        """
availability = pd.read_csv(OUTPUT_ROOT / "safe" / "encoder_availability.csv")
runtime = pd.read_csv(OUTPUT_ROOT / "safe" / "runtime_summary.csv")
display(availability)
display(runtime)
"""
    ),
    markdown(
        """
## Feature cache and private diagnostics

The cache stores global features and coarse P1 heatmaps in private Kaggle output. Patch feature arrays are temporary and are not retained. The cache is excluded from GitHub and ClearML.

The next panel shows the highest mean P1 mismatch examples from each available ViT. It is for visual checking only and remains private.
"""
    ),
    code(
        """
cache_summary = json.loads((OUTPUT_ROOT / "safe" / "cache_summary.json").read_text(encoding="utf-8"))
display(pd.DataFrame(cache_summary.items(), columns=["measure", "value"]))

for panel in sorted((OUTPUT_ROOT / "private" / "figures").glob("p1_heatmap_examples_*.png")):
    print(panel.name)
    display(DisplayImage(filename=str(panel)))
"""
    ),
    markdown(
        """
## Stage 2: evaluator-only ranking metrics

The score package manifest and package record are verified before this stage starts. The evaluator then maps route-local candidates to randomized object identities and labels.

For the automatic route, unmatched proposals remain negative candidates and missed anomalous objects remain in the AP denominator. A top-ranked unmatched proposal is a Top-1 failure. Views with fewer than three proposals reduce reported coverage.
"""
    ),
    code(
        """
from visual_peer_inspector.workflows.representation_ds0 import run_private_ds0_evaluation

evaluation_record_path = OUTPUT_ROOT / "safe" / "run_record.json"
if evaluation_record_path.is_file():
    evaluation_record = json.loads(evaluation_record_path.read_text(encoding="utf-8"))
    print("Reused the verified evaluator outputs.")
else:
    evaluation_record = run_private_ds0_evaluation(
        representation_root=OUTPUT_ROOT,
        evaluator_root=EVALUATOR_ROOT,
    )
display(pd.DataFrame([evaluation_record]))
"""
    ),
    markdown(
        """
## Encoder comparison

The primary DS0 comparison uses supplied masks and B0. This isolates the frozen visual representation from proposal and segmentation errors. The results are provisional because final encoder selection is reserved for DS1.
"""
    ),
    code(
        """
encoder_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "encoder_summary.csv")
intervals = pd.read_csv(OUTPUT_ROOT / "safe" / "clustered_intervals.csv")
paired_differences = pd.read_csv(OUTPUT_ROOT / "safe" / "paired_differences.csv")
display(encoder_summary.sort_values("average_precision", ascending=False))
display(intervals.query("route == 'supplied_mask' and method == 'B0'"))
display(paired_differences.query("comparison_family == 'encoder_vs_anchor'"))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "encoder_b0_scene_ap.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "encoder_efficiency.png")))
"""
    ),
    markdown(
        """
## B0, B1, and P1 comparison

This comparison uses the same supplied-mask objects and frozen encoder. It shows whether robust global distances or patch evidence improve ranking over B0. Results are split by one-anomaly and two-anomaly scenes.
"""
    ),
    code(
        """
scorer_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "scorer_summary.csv")
scene_type_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "scene_type_summary.csv")
display(scorer_summary.sort_values(["encoder_key", "average_precision"], ascending=[True, False]))
display(scene_type_summary)
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "anchor_scorer_comparison.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "one_vs_two_anomaly_scene_ap.png")))
"""
    ),
    markdown(
        """
## Route impact and rankable coverage

The route table compares supplied masks, SAM masks prompted by reference boxes, and SAM masks prompted by automatic boxes. These are downstream ranking results, not segmentation IoU results.
"""
    ),
    code(
        """
route_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "route_summary.csv")
coverage_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "coverage_summary.csv")
display(route_summary)
display(coverage_summary)
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "anchor_route_comparison.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "rankable_coverage.png")))
"""
    ),
    markdown(
        """
## Geometry branch and encoder agreement

Shape and layout are kept separate from appearance. This avoids tuning a fusion weight on DS0. The agreement matrix shows how often encoders choose the same most unusual supplied-mask object in a view.
"""
    ),
    code(
        """
geometry_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "geometry_summary.csv")
agreement = pd.read_csv(OUTPUT_ROOT / "safe" / "model_agreement.csv")
display(geometry_summary)
display(agreement)
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "geometry_vs_appearance.png")))
display(DisplayImage(filename=str(OUTPUT_ROOT / "safe" / "figures" / "model_agreement.png")))
"""
    ),
    markdown(
        """
## View profile

Views are repeated observations, not independent scenes. Metrics are averaged across eligible views within each scene before dataset aggregation. This table remains useful for finding camera angles that are consistently difficult.
"""
    ),
    code(
        """
view_summary = pd.read_csv(OUTPUT_ROOT / "safe" / "view_summary.csv")
display(view_summary.sort_values("average_precision").head(30))
"""
    ),
    markdown(
        """
## Completion gates and decision

Passing this notebook means the representation and peer-scoring machinery is ready to scale. It does not mean that a final DINO model or final mask route has been selected.
"""
    ),
    code(
        """
completion_gates = pd.read_csv(OUTPUT_ROOT / "safe" / "completion_gates.csv")
decision = json.loads((OUTPUT_ROOT / "safe" / "promotion_decision.json").read_text(encoding="utf-8"))
display(completion_gates)
display(pd.DataFrame([decision]))
if not completion_gates["passed"].astype(bool).all():
    raise RuntimeError("Notebook 07 has a failed completion gate.")
"""
    ),
    markdown(
        """
## Safe ClearML finalisation

The finaliser runs only after all inference, evaluation, charts, and manifests are complete. The Kaggle run records a disabled receipt by default. After local verification of the downloaded output, the same finaliser can upload aggregate tables, charts, configuration hashes, runtime, memory, cache, coverage, and decision evidence.

It rejects RGB images, masks, crops, P1 heatmaps, feature arrays, per-object scores, labels, object IDs, scene IDs, crosswalks, credentials, and absolute paths. A ClearML timeout records a receipt but does not erase the completed model run.
"""
    ),
    code(
        """
TRACKING_MODE = os.getenv(
    "VPI_TRACKING_MODE",
    "disabled",
)
tracking_environment = os.environ.copy()
if TRACKING_MODE == "online":
    for name in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY"):
        value = read_kaggle_secret(name)
        if not value:
            raise RuntimeError(f"Kaggle Secret {name} is missing or disabled.")
        tracking_environment[name] = value
    tracking_environment.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    tracking_environment.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    tracking_environment.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")

receipt = OUTPUT_ROOT / "clearml_receipt.json"
command = [
    sys.executable,
    str(CODE_ROOT / "scripts" / "finalize_representation_clearml.py"),
    "--safe-root", str(OUTPUT_ROOT / "safe"),
    "--receipt", str(receipt),
    "--mode", TRACKING_MODE,
    "--task-name", "ds0-representation-peer-scores-kaggle-candidate",
    "--tag", "candidate",
    "--tag", "notebook-07",
    "--tag", "ds0",
    "--tag", "kaggle",
]
tracking_process = subprocess.run(
    command,
    env=tracking_environment,
    text=True,
    capture_output=True,
    timeout=240,
)
if tracking_process.returncode != 0:
    print(tracking_process.stderr[-1000:])
tracking_receipt = json.loads(receipt.read_text(encoding="utf-8"))
display(pd.DataFrame([tracking_receipt]))
"""
    ),
    markdown(
        """
## Final artifact verification

The same verifier is used locally after downloading Kaggle output. It checks all four artifact manifests, the completion record, tables, chart integrity, and the declared run mode.
"""
    ),
    code(
        """
verify_command = [
    sys.executable,
    str(CODE_ROOT / "scripts" / "verify_ds0_representation.py"),
    "--output-root", str(OUTPUT_ROOT),
]
if RUN_MODE == "model_evidence":
    verify_command.append("--require-model-evidence")
verified = subprocess.run(verify_command, check=True, text=True, capture_output=True)
verification_summary = json.loads(verified.stdout)
display(pd.DataFrame([verification_summary]))
"""
    ),
    markdown(
        """
## Next stage

Notebook 07 is complete when the model-evidence run passes and the downloaded output verifies locally.

The next work is:

1. prepare DS1 with 100 development scenes and supplied masks;
2. run the fixed B0 encoder screen;
3. retain DINOv2-small unless a challenger passes the declared AP, confidence, subgroup, runtime, and memory gates;
4. freeze the encoder;
5. compare B0, B1, P1, and the three mask routes on DS1; and
6. start Notebook 08 for identity association, view subsets, score fusion, and calibration.

Cross-view fusion is deliberately absent here because route prompt keys are local to each image and are not stable object identities.
"""
    ),
]


notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
