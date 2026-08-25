"""Pinned SAM 2.1 box-prompt inference with atomic DS0 result storage."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
from PIL import Image

from visual_peer_inspector.data.model_safe_package import (
    SMOKE20_COHORT_ID,
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.data.sam_prompt_package import verify_sam_prompt_package


SAM2_CODE_REPOSITORY = "https://github.com/facebookresearch/sam2"
SAM2_CODE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_MODEL_REPOSITORY = "facebook/sam2.1-hiera-large"
SAM2_MODEL_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
SAM2_CHECKPOINT_FILENAME = "sam2.1_hiera_large.pt"
SAM2_CHECKPOINT_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM2_CHECKPOINT_SIZE_BYTES = 898083611
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_RESULT_SCHEMA_VERSION = 1
MODEL_PRODUCER_KIND = "sam2.1_model"
CONTRACT_PRODUCER_KIND = "sam2.1_contract"
SCENE_PATTERN = re.compile(r"^S\d{6}$")
VIEW_PATTERN = re.compile(r"^V\d{2}$")


@dataclass(frozen=True)
class Sam2Configuration:
    code_repository: str = SAM2_CODE_REPOSITORY
    code_revision: str = SAM2_CODE_REVISION
    model_repository: str = SAM2_MODEL_REPOSITORY
    model_revision: str = SAM2_MODEL_REVISION
    checkpoint_filename: str = SAM2_CHECKPOINT_FILENAME
    checkpoint_sha256: str = SAM2_CHECKPOINT_SHA256
    checkpoint_size_bytes: int = SAM2_CHECKPOINT_SIZE_BYTES
    model_config: str = SAM2_MODEL_CONFIG
    device: str = "cuda"
    precision: str = "float16"
    multimask_output: bool = False
    mask_threshold: float = 0.0
    max_hole_area: float = 0.0
    max_sprinkle_area: float = 0.0
    seed: int = 20260824
    image_height: int = 256
    image_width: int = 256

    @property
    def sha256(self) -> str:
        return sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_repository": self.code_repository,
            "code_revision": self.code_revision,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "checkpoint_filename": self.checkpoint_filename,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "model_config": self.model_config,
            "device": self.device,
            "precision": self.precision,
            "multimask_output": self.multimask_output,
            "mask_threshold": self.mask_threshold,
            "max_hole_area": self.max_hole_area,
            "max_sprinkle_area": self.max_sprinkle_area,
            "seed": self.seed,
            "image_height": self.image_height,
            "image_width": self.image_width,
        }


@dataclass(frozen=True)
class SamPrompt:
    route: str
    prompt_key: str
    box_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class SamImageJob:
    opaque_scene_id: str
    view_id: str
    image_path: Path
    image_sha256: str
    prompts: tuple[SamPrompt, ...]
    prompt_package_sha256: str
    cohort_id: str = SMOKE20_COHORT_ID

    @property
    def key(self) -> tuple[str, str]:
        return self.opaque_scene_id, self.view_id


class SamWorker(Protocol):
    producer_kind: str
    configuration: Sam2Configuration

    def process(self, job: SamImageJob, attempt: int) -> dict[str, Any]: ...


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as deterministic uncompressed COCO RLE."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("A SAM mask must be two-dimensional")
    flat = array.reshape(-1, order="F").astype(np.uint8)
    counts: list[int] = []
    current = 0
    length = 0
    for value in flat:
        integer = int(value)
        if integer == current:
            length += 1
        else:
            counts.append(length)
            current = integer
            length = 1
    counts.append(length)
    return {"size": [int(array.shape[0]), int(array.shape[1])], "counts": counts}


def decode_binary_mask(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode and validate deterministic uncompressed COCO RLE."""

    size = rle.get("size")
    counts = rle.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not isinstance(counts, list)
        or not counts
    ):
        raise ValueError("Invalid mask RLE structure")
    height, width = [int(value) for value in size]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid mask RLE dimensions")
    total = height * width
    offset = 0
    normalised_counts: list[int] = []
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise ValueError("Mask RLE counts must be integers")
        integer = int(count)
        if integer < 0:
            raise ValueError("Mask RLE counts cannot be negative")
        offset += integer
        if offset > total:
            raise ValueError("Mask RLE length exceeds its dimensions")
        normalised_counts.append(integer)
    if offset != total:
        raise ValueError("Mask RLE length differs from its dimensions")
    flat = np.empty(total, dtype=bool)
    offset = 0
    current = False
    for count in normalised_counts:
        flat[offset : offset + count] = current
        offset += count
        current = not current
    return flat.reshape((height, width), order="F")


def binary_mask_sha256(mask: np.ndarray) -> str:
    array = np.asarray(mask, dtype=bool)
    header = f"{array.shape[0]}x{array.shape[1]}:".encode("ascii")
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return sha256(header + packed.tobytes()).hexdigest()


def verify_installed_sam2_revision(expected_revision: str) -> dict[str, Any]:
    """Verify that the installed SAM-2 package came from the pinned commit."""

    distribution = importlib.metadata.distribution("SAM-2")
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise ValueError("The installed SAM-2 package has no direct URL provenance")
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info", {})
    if vcs_info.get("vcs") != "git" or vcs_info.get("commit_id") != expected_revision:
        raise ValueError("The installed SAM-2 package is not the pinned code revision")
    installed_repository = str(direct_url.get("url", "")).rstrip("/").removesuffix(
        ".git"
    )
    if installed_repository != SAM2_CODE_REPOSITORY:
        raise ValueError("The installed SAM-2 package came from another repository")
    return {
        "distribution": distribution.metadata["Name"],
        "version": distribution.version,
        "repository": direct_url["url"],
        "commit_id": vcs_info["commit_id"],
    }


def _safe_media_path(root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("SAM image path is unsafe")
    output = root.joinpath(*relative.parts).resolve()
    if root not in output.parents:
        raise ValueError("SAM image path escapes the media root")
    return output


def build_sam_jobs(prompt_root: Path, media_root: Path) -> list[SamImageJob]:
    """Verify the prompt package and build all 400 opaque image jobs."""

    prompt_root = prompt_root.expanduser().resolve()
    media_root = media_root.expanduser().resolve()
    package_record = verify_sam_prompt_package(prompt_root)
    prompt_package_sha256 = file_sha256(prompt_root / "package_record.json")
    jobs: list[SamImageJob] = []
    for row in read_jsonl(prompt_root / "sam_prompt_manifest.jsonl"):
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        if not SCENE_PATTERN.fullmatch(scene_id) or not VIEW_PATTERN.fullmatch(view_id):
            raise ValueError("SAM prompt IDs do not follow the opaque format")
        prompts: list[SamPrompt] = []
        for route in ("reference_box", "automatic_box"):
            for prompt in row["routes"][route]:
                box = tuple(int(value) for value in prompt["box_xyxy"])
                prompts.append(
                    SamPrompt(
                        route=route,
                        prompt_key=str(prompt["prompt_key"]),
                        box_xyxy=box,
                    )
                )
        image_path = _safe_media_path(media_root, str(row["image_path"]))
        if file_sha256(image_path) != str(row["image_sha256"]):
            raise ValueError("SAM input image hash differs from the prompt manifest")
        jobs.append(
            SamImageJob(
                opaque_scene_id=scene_id,
                view_id=view_id,
                image_path=image_path,
                image_sha256=str(row["image_sha256"]),
                prompts=tuple(prompts),
                prompt_package_sha256=prompt_package_sha256,
                cohort_id=str(package_record["cohort_id"]),
            )
        )
    if len(jobs) != int(package_record["image_count"]) or len(jobs) != 400:
        raise ValueError("SAM job list must contain all 400 DS0 images")
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("SAM job list contains duplicate images")
    return sorted(jobs, key=lambda item: item.key)


def _prompt_output(
    prompt: SamPrompt,
    mask: np.ndarray,
    score: float,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    array = np.asarray(mask, dtype=bool)
    if array.shape != expected_shape:
        raise ValueError(f"SAM returned a mask with shape {array.shape}")
    foreground_pixels = int(array.sum())
    if not np.isfinite(float(score)):
        raise ValueError("SAM returned a non-finite predicted quality score")
    return {
        "prompt_key": prompt.prompt_key,
        "box_xyxy": list(prompt.box_xyxy),
        "predicted_quality_score": float(score),
        "foreground_pixels": foreground_pixels,
        "mask_status": "empty" if foreground_pixels == 0 else "non_empty",
        "mask_sha256": binary_mask_sha256(array),
        "mask_rle": encode_binary_mask(array),
    }


class ContractSamWorker:
    """Deterministic engineering worker that fills every prompt box."""

    producer_kind = CONTRACT_PRODUCER_KIND

    def __init__(self, configuration: Sam2Configuration):
        self.configuration = configuration

    def process(self, job: SamImageJob, attempt: int) -> dict[str, Any]:
        started = time.perf_counter()
        routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prompt in job.prompts:
            mask = np.zeros((256, 256), dtype=bool)
            x1, y1, x2, y2 = prompt.box_xyxy
            mask[y1:y2, x1:x2] = True
            routes[prompt.route].append(
                _prompt_output(
                    prompt,
                    mask,
                    0.99,
                    (self.configuration.image_height, self.configuration.image_width),
                )
            )
        return _build_result(
            job,
            routes,
            configuration=self.configuration,
            producer_kind=self.producer_kind,
            attempt=attempt,
            embedding_runtime_seconds=0.0,
            prompt_runtime_seconds=time.perf_counter() - started,
        )


class Sam2ImageWorker:
    """Official SAM 2.1 image predictor with one embedding per image."""

    producer_kind = MODEL_PRODUCER_KIND

    def __init__(
        self,
        configuration: Sam2Configuration,
        checkpoint_path: Path,
        *,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ):
        if configuration.device != "cuda":
            raise ValueError("The accepted SAM 2.1 worker requires CUDA")
        if configuration.precision != "float16":
            raise ValueError("The accepted SAM 2.1 worker requires float16 precision")
        if (configuration.image_height, configuration.image_width) != (256, 256):
            raise ValueError("The accepted DS0 SAM worker requires 256 by 256 images")
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if checkpoint_path.stat().st_size != configuration.checkpoint_size_bytes:
            raise ValueError("SAM checkpoint size differs from the pinned file")
        if file_sha256(checkpoint_path) != configuration.checkpoint_sha256:
            raise ValueError("SAM checkpoint hash differs from the pinned file")
        self.configuration = configuration
        self.progress_callback = progress_callback
        self._report("sam_import_started", {})
        self.installation_provenance = verify_installed_sam2_revision(
            configuration.code_revision
        )
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM 2.1 model inference")
        torch.manual_seed(configuration.seed)
        torch.cuda.manual_seed_all(configuration.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self._report("sam_model_build_started", {})
        model = build_sam2(
            configuration.model_config,
            str(checkpoint_path),
            device=configuration.device,
            mode="eval",
        )
        self.predictor = SAM2ImagePredictor(
            model,
            mask_threshold=configuration.mask_threshold,
            max_hole_area=configuration.max_hole_area,
            max_sprinkle_area=configuration.max_sprinkle_area,
        )
        torch.cuda.reset_peak_memory_stats()
        self._report("sam_worker_ready", {})

    def _report(self, stage: str, details: Mapping[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, dict(details))

    def process(self, job: SamImageJob, attempt: int) -> dict[str, Any]:
        with Image.open(job.image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        if image.shape != (256, 256, 3):
            raise ValueError(f"SAM input image has shape {image.shape}")
        boxes = np.asarray([prompt.box_xyxy for prompt in job.prompts], dtype=np.float32)
        if not len(boxes):
            raise ValueError("SAM image job has no prompts")

        torch = self.torch
        torch.cuda.synchronize()
        embedding_started = time.perf_counter()
        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                self.predictor.set_image(image)
            torch.cuda.synchronize()
            embedding_seconds = time.perf_counter() - embedding_started
            prompt_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                masks, scores, _ = self.predictor.predict(
                    box=boxes,
                    multimask_output=self.configuration.multimask_output,
                )
            torch.cuda.synchronize()
            prompt_seconds = time.perf_counter() - prompt_started
        finally:
            self.predictor.reset_predictor()

        mask_array = np.asarray(masks)
        if mask_array.ndim == 3 and len(job.prompts) == 1:
            mask_array = mask_array.reshape(1, 256, 256)
        elif mask_array.ndim == 4 and mask_array.shape[1] == 1:
            mask_array = mask_array[:, 0]
        if mask_array.shape != (len(job.prompts), 256, 256):
            raise ValueError(
                f"SAM mask batch shape {mask_array.shape} does not match "
                f"{len(job.prompts)} prompts"
            )
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_array.shape != (len(job.prompts),):
            raise ValueError("SAM score count does not match the prompts")

        routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prompt, mask, score in zip(job.prompts, mask_array, score_array):
            routes[prompt.route].append(
                _prompt_output(
                    prompt,
                    mask,
                    float(score),
                    (self.configuration.image_height, self.configuration.image_width),
                )
            )
        return _build_result(
            job,
            routes,
            configuration=self.configuration,
            producer_kind=self.producer_kind,
            attempt=attempt,
            embedding_runtime_seconds=embedding_seconds,
            prompt_runtime_seconds=prompt_seconds,
        )


def _build_result(
    job: SamImageJob,
    routes: Mapping[str, list[dict[str, Any]]],
    *,
    configuration: Sam2Configuration,
    producer_kind: str,
    attempt: int,
    embedding_runtime_seconds: float,
    prompt_runtime_seconds: float,
) -> dict[str, Any]:
    expected = Counter((prompt.route, prompt.prompt_key) for prompt in job.prompts)
    observed = Counter(
        (route, str(item["prompt_key"]))
        for route, items in routes.items()
        for item in items
    )
    if expected != observed:
        raise ValueError("SAM result prompts do not match the image job")
    return {
        "schema_version": SAM_RESULT_SCHEMA_VERSION,
        "producer_kind": producer_kind,
        "cohort_id": job.cohort_id,
        "opaque_scene_id": job.opaque_scene_id,
        "view_id": job.view_id,
        "image_sha256": job.image_sha256,
        "prompt_package_sha256": job.prompt_package_sha256,
        "configuration_sha256": configuration.sha256,
        "attempt": int(attempt),
        "status": "ok",
        "embedding_runtime_seconds": round(float(embedding_runtime_seconds), 6),
        "prompt_runtime_seconds": round(float(prompt_runtime_seconds), 6),
        "total_runtime_seconds": round(
            float(embedding_runtime_seconds) + float(prompt_runtime_seconds), 6
        ),
        "prompt_count": len(job.prompts),
        "routes": {route: list(routes[route]) for route in sorted(routes)},
    }


class SamResultStore:
    """Save only complete per-image results and resume by exact input hashes."""

    def __init__(
        self,
        root: Path,
        configuration_sha256: str,
        prompt_package_sha256: str,
        producer_kind: str,
        cohort_id: str = SMOKE20_COHORT_ID,
    ):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.configuration_sha256 = configuration_sha256
        self.prompt_package_sha256 = prompt_package_sha256
        self.producer_kind = producer_kind
        self.cohort_id = cohort_id
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        for path in sorted(self.root.glob("S*/V*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            self._validate_common(row)
            key = (str(row["opaque_scene_id"]), str(row["view_id"]))
            if key in self._rows:
                raise ValueError("Stored SAM results contain duplicate images")
            self._rows[key] = row

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [self._rows[key] for key in sorted(self._rows)]

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        return set(self._rows)

    def _validate_common(self, row: Mapping[str, Any]) -> None:
        if row.get("schema_version") != SAM_RESULT_SCHEMA_VERSION:
            raise ValueError("Stored SAM result schema is unsupported")
        if row.get("producer_kind") != self.producer_kind:
            raise ValueError("Stored SAM producer differs from this run")
        if row.get("configuration_sha256") != self.configuration_sha256:
            raise ValueError("Stored SAM configuration differs from this run")
        if row.get("prompt_package_sha256") != self.prompt_package_sha256:
            raise ValueError("Stored SAM prompt package differs from this run")
        if row.get("status") != "ok":
            raise ValueError("Only successful SAM image results may be stored")
        if row.get("cohort_id") != self.cohort_id:
            raise ValueError("Stored SAM result identifies another cohort")
        if not SCENE_PATTERN.fullmatch(str(row.get("opaque_scene_id", ""))):
            raise ValueError("Stored SAM result has an invalid opaque scene ID")
        if not VIEW_PATTERN.fullmatch(str(row.get("view_id", ""))):
            raise ValueError("Stored SAM result has an invalid view ID")
        if set(row.get("routes", {})) != {"automatic_box", "reference_box"}:
            raise ValueError("Stored SAM result has an unexpected route set")
        if int(row.get("attempt", 0)) < 1:
            raise ValueError("Stored SAM result has an invalid attempt number")
        for name in (
            "embedding_runtime_seconds",
            "prompt_runtime_seconds",
            "total_runtime_seconds",
        ):
            value = float(row.get(name, -1.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError("Stored SAM result has an invalid runtime")
        if not math.isclose(
            float(row["total_runtime_seconds"]),
            float(row["embedding_runtime_seconds"])
            + float(row["prompt_runtime_seconds"]),
            abs_tol=2e-6,
        ):
            raise ValueError("Stored SAM total runtime is inconsistent")
        observed_prompt_count = sum(len(items) for items in row["routes"].values())
        if int(row.get("prompt_count", -1)) != observed_prompt_count:
            raise ValueError("Stored SAM prompt count is inconsistent")
        for items in row["routes"].values():
            for item in items:
                mask = decode_binary_mask(item["mask_rle"])
                if binary_mask_sha256(mask) != item["mask_sha256"]:
                    raise ValueError("Stored SAM mask hash differs from its RLE")
                if int(item["foreground_pixels"]) != int(mask.sum()):
                    raise ValueError("Stored SAM foreground count differs from its mask")
                expected_mask_status = (
                    "empty" if int(item["foreground_pixels"]) == 0 else "non_empty"
                )
                if item.get("mask_status") != expected_mask_status:
                    raise ValueError("Stored SAM mask status is inconsistent")
                score = float(item["predicted_quality_score"])
                if not math.isfinite(score):
                    raise ValueError("Stored SAM score is not finite")
                box = [int(value) for value in item["box_xyxy"]]
                if len(box) != 4 or box != item["box_xyxy"]:
                    raise ValueError("Stored SAM prompt box is invalid")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
                    raise ValueError("Stored SAM prompt box is outside the image")

    def reconcile_jobs(self, jobs: Iterable[SamImageJob]) -> None:
        job_list = list(jobs)
        job_by_key = {job.key: job for job in job_list}
        if len(job_by_key) != len(job_list):
            raise ValueError("SAM job list contains duplicate images")
        unexpected = set(self._rows) - set(job_by_key)
        if unexpected:
            raise ValueError("Stored SAM results contain unexpected images")
        for key, row in self._rows.items():
            job = job_by_key[key]
            if job.cohort_id != self.cohort_id:
                raise ValueError("SAM job identifies another cohort")
            if row["image_sha256"] != job.image_sha256:
                raise ValueError("Stored SAM result identifies another image")
            expected = Counter(
                (prompt.route, prompt.prompt_key, prompt.box_xyxy)
                for prompt in job.prompts
            )
            observed = Counter(
                (route, str(item["prompt_key"]), tuple(item["box_xyxy"]))
                for route, items in row["routes"].items()
                for item in items
            )
            if expected != observed:
                raise ValueError("Stored SAM result has incomplete prompts")

    def write(self, job: SamImageJob, row: Mapping[str, Any]) -> Path:
        self._validate_common(row)
        if (str(row["opaque_scene_id"]), str(row["view_id"])) != job.key:
            raise ValueError("SAM result identifies another image job")
        if str(row["image_sha256"]) != job.image_sha256:
            raise ValueError("SAM result identifies another input image")
        if int(row["prompt_count"]) != len(job.prompts):
            raise ValueError("SAM result prompt count differs from the job")
        expected = Counter(
            (prompt.route, prompt.prompt_key, prompt.box_xyxy)
            for prompt in job.prompts
        )
        observed = Counter(
            (route, str(item["prompt_key"]), tuple(item["box_xyxy"]))
            for route, items in row["routes"].items()
            for item in items
        )
        if expected != observed:
            raise ValueError("SAM result prompt keys differ from the job")
        path = self.root / job.opaque_scene_id / f"{job.view_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("A completed SAM image result would change")
        else:
            temporary = path.with_suffix(".json.incomplete")
            temporary.write_bytes(data)
            os.replace(temporary, path)
        self._rows[job.key] = dict(row)
        return path

    def write_index(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(canonical_json_bytes(row) + b"\n" for row in self.rows)
        temporary = path.with_suffix(path.suffix + ".incomplete")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        return path


def run_sam_jobs(
    *,
    worker: SamWorker,
    jobs: Iterable[SamImageJob],
    result_store: SamResultStore,
    maximum_attempts_per_job: int = 2,
    maximum_consecutive_failed_jobs: int = 2,
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
    attempt_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run pending images with bounded retries and a consecutive-failure stop."""

    job_list = list(jobs)
    pending = [job for job in job_list if job.key not in result_store.completed_keys]
    attempts: list[dict[str, Any]] = []
    consecutive_failures = 0
    for completed, job in enumerate(pending, start=1):
        final_error: Exception | None = None
        for attempt in range(1, maximum_attempts_per_job + 1):
            started = time.perf_counter()
            try:
                row = worker.process(job, attempt)
                result_store.write(job, row)
                attempt_row = {
                    "status": "ok",
                    "attempt": attempt,
                    "opaque_scene_id": job.opaque_scene_id,
                    "view_id": job.view_id,
                    "runtime_seconds": round(time.perf_counter() - started, 6),
                    "error_type": None,
                }
                attempts.append(attempt_row)
                if attempt_callback is not None:
                    attempt_callback(attempt_row)
                final_error = None
                break
            except Exception as error:
                final_error = error
                attempt_row = {
                    "status": "error",
                    "attempt": attempt,
                    "opaque_scene_id": job.opaque_scene_id,
                    "view_id": job.view_id,
                    "runtime_seconds": round(time.perf_counter() - started, 6),
                    "error_type": type(error).__name__,
                }
                attempts.append(attempt_row)
                if attempt_callback is not None:
                    attempt_callback(attempt_row)
        if final_error is not None:
            consecutive_failures += 1
            if consecutive_failures >= maximum_consecutive_failed_jobs:
                raise RuntimeError("SAM stopped after consecutive failed images") from final_error
        else:
            consecutive_failures = 0
        if progress_callback is not None:
            progress_callback(completed, len(pending), attempts[-1])
    return attempts


def compare_repeat_results(
    first_rows: Iterable[Mapping[str, Any]],
    repeated_rows: Iterable[Mapping[str, Any]],
    *,
    score_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare prompt-level mask digests and predicted scores for a pilot repeat."""

    def index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], tuple[str, float]]:
        values: dict[tuple[str, str, str, str], tuple[str, float]] = {}
        for row in rows:
            for route, items in row["routes"].items():
                for item in items:
                    key = (
                        str(row["opaque_scene_id"]),
                        str(row["view_id"]),
                        str(route),
                        str(item["prompt_key"]),
                    )
                    if key in values:
                        raise ValueError("SAM repeat inputs contain duplicate prompts")
                    values[key] = (
                        str(item["mask_sha256"]),
                        float(item["predicted_quality_score"]),
                    )
        return values

    first = index(first_rows)
    repeated = index(repeated_rows)
    if set(first) != set(repeated):
        raise ValueError("SAM repeat coverage differs from the pilot")
    digest_matches = sum(first[key][0] == repeated[key][0] for key in first)
    score_matches = sum(
        abs(first[key][1] - repeated[key][1]) <= score_tolerance for key in first
    )
    count = len(first)
    return {
        "prompt_count": count,
        "matching_mask_digest_count": digest_matches,
        "mask_digest_agreement_rate": digest_matches / count if count else 0.0,
        "matching_score_count": score_matches,
        "score_agreement_rate": score_matches / count if count else 0.0,
        "score_tolerance": float(score_tolerance),
    }


def collect_sam_evidence(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return aggregate operational evidence without opaque image identifiers."""

    row_list = list(rows)
    route_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    view_values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    image_runtimes: list[float] = []
    prompt_counts: list[int] = []
    for row in row_list:
        image_runtimes.append(float(row["total_runtime_seconds"]))
        prompt_counts.append(int(row["prompt_count"]))
        for route, items in row["routes"].items():
            for item in items:
                area_fraction = int(item["foreground_pixels"]) / (256 * 256)
                score = float(item["predicted_quality_score"])
                route_values[route]["area_fraction"].append(area_fraction)
                route_values[route]["score"].append(score)
                view_values[(str(row["view_id"]), route)]["area_fraction"].append(
                    area_fraction
                )
                view_values[(str(row["view_id"]), route)]["score"].append(score)

    route_rows = []
    for route in sorted(route_values):
        areas = np.asarray(route_values[route]["area_fraction"], dtype=float)
        scores = np.asarray(route_values[route]["score"], dtype=float)
        route_rows.append(
            {
                "route": route,
                "mask_count": len(areas),
                "empty_mask_count": int(np.sum(areas == 0)),
                "empty_mask_rate": float(np.mean(areas == 0)),
                "mean_area_fraction": float(np.mean(areas)),
                "median_area_fraction": float(np.median(areas)),
                "p95_area_fraction": float(np.quantile(areas, 0.95)),
                "mean_predicted_quality_score": float(np.mean(scores)),
                "p10_predicted_quality_score": float(np.quantile(scores, 0.10)),
            }
        )
    view_rows = []
    for (view_id, route), values in sorted(view_values.items()):
        view_rows.append(
            {
                "view_id": view_id,
                "route": route,
                "mask_count": len(values["area_fraction"]),
                "empty_mask_count": sum(
                    value == 0 for value in values["area_fraction"]
                ),
                "mean_area_fraction": float(np.mean(values["area_fraction"])),
                "mean_predicted_quality_score": float(np.mean(values["score"])),
            }
        )
    return {
        "image_count": len(row_list),
        "scene_count": len({str(row["opaque_scene_id"]) for row in row_list}),
        "mask_count": sum(int(row["prompt_count"]) for row in row_list),
        "empty_mask_count": sum(
            int(item["foreground_pixels"]) == 0
            for row in row_list
            for items in row["routes"].values()
            for item in items
        ),
        "median_image_runtime_seconds": float(np.median(image_runtimes)),
        "p95_image_runtime_seconds": float(np.quantile(image_runtimes, 0.95)),
        "median_prompts_per_image": float(np.median(prompt_counts)),
        "route_rows": route_rows,
        "view_rows": view_rows,
        "image_runtimes": image_runtimes,
        "prompt_counts": prompt_counts,
        "area_values_by_route": {
            route: values["area_fraction"] for route, values in route_values.items()
        },
        "score_values_by_route": {
            route: values["score"] for route, values in route_values.items()
        },
    }


__all__ = [
    "CONTRACT_PRODUCER_KIND",
    "ContractSamWorker",
    "MODEL_PRODUCER_KIND",
    "SAM2_CHECKPOINT_FILENAME",
    "SAM2_CHECKPOINT_SHA256",
    "SAM2_CHECKPOINT_SIZE_BYTES",
    "SAM2_CODE_REVISION",
    "SAM2_MODEL_REVISION",
    "Sam2Configuration",
    "Sam2ImageWorker",
    "SamImageJob",
    "SamPrompt",
    "SamResultStore",
    "binary_mask_sha256",
    "build_sam_jobs",
    "collect_sam_evidence",
    "compare_repeat_results",
    "decode_binary_mask",
    "encode_binary_mask",
    "run_sam_jobs",
    "verify_installed_sam2_revision",
]
