"""Pinned LocateAnything proposal runner for opaque RGB inputs."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import signal
from statistics import mean, median
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)


DEFAULT_LOCATEANYTHING_MODEL = "nvidia/LocateAnything-3B"
DEFAULT_LOCATEANYTHING_REVISION = (
    "c32291ca5e996f5a7a485845b4f57a233936bba0"
)
DEFAULT_PROMPT = "object"
DEFAULT_UPSCALE_FACTOR = 1
DEFAULT_SEED = 32040
PINNED_MAX_NEW_TOKENS = 8192
PRIMARY_GENERATION_MODE = "hybrid"
FALLBACK_GENERATION_MODE = "slow"
PRIMARY_GENERATION_ATTEMPTS = 4
MAXIMUM_GENERATION_ATTEMPTS = 6
MODEL_PRODUCER_KIND = "locateanything_model"
CONTRACT_PRODUCER_KIND = "contract_smoke"
PRODUCER_KINDS = frozenset({MODEL_PRODUCER_KIND, CONTRACT_PRODUCER_KIND})

_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "cohort_id",
        "input_manifest_sha256",
        "opaque_scene_id",
        "view_id",
        "attempt",
        "image_sha256",
        "configuration_sha256",
        "execution_policy_sha256",
        "effective_generation_mode",
        "effective_max_new_tokens",
        "producer_kind",
        "model_id",
        "model_revision",
        "prompt",
        "proposal_count",
        "boxes_xyxy",
        "raw_response",
        "generation_stats",
        "response_token_count",
        "possibly_truncated",
        "truncation_check",
        "runtime_seconds",
        "status",
        "error_type",
        "error_message",
    }
)


def locateanything_execution_policy() -> dict[str, Any]:
    """Return the canonical retry policy without mutating model configuration."""

    return {
        "schema_version": 1,
        "policy_kind": "locateanything_truncation_recovery",
        "maximum_attempts_per_job": MAXIMUM_GENERATION_ATTEMPTS,
        "primary": {
            "attempts": list(range(1, PRIMARY_GENERATION_ATTEMPTS + 1)),
            "generation_mode": PRIMARY_GENERATION_MODE,
            "max_new_tokens": PINNED_MAX_NEW_TOKENS,
        },
        "truncation_fallback": {
            "attempts": list(
                range(PRIMARY_GENERATION_ATTEMPTS + 1, MAXIMUM_GENERATION_ATTEMPTS + 1)
            ),
            "generation_mode": FALLBACK_GENERATION_MODE,
            "max_new_tokens": PINNED_MAX_NEW_TOKENS,
            "trigger": "previous_attempt_possible_truncation",
        },
    }


def bounded_locateanything_execution_policy(
    configuration: LocateAnythingConfiguration,
    *,
    maximum_attempts_per_job: int,
    terminal_failure_disposition: str,
    attempt_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Describe a bounded run without changing the historical default policy."""

    if maximum_attempts_per_job < 1:
        raise ValueError("At least one proposal attempt is required")
    if maximum_attempts_per_job > PRIMARY_GENERATION_ATTEMPTS:
        raise ValueError("Bounded policies cannot enter the slow fallback phase")
    if terminal_failure_disposition not in {"abort", "carry_as_zero_proposal"}:
        raise ValueError("Unsupported terminal proposal failure disposition")
    if attempt_timeout_seconds is not None and attempt_timeout_seconds <= 0:
        raise ValueError("Proposal attempt timeout must be positive")
    return {
        "schema_version": 1,
        "policy_kind": "locateanything_bounded_generation",
        "maximum_attempts_per_job": maximum_attempts_per_job,
        "primary": {
            "attempts": list(range(1, maximum_attempts_per_job + 1)),
            "generation_mode": configuration.generation_mode,
            "max_new_tokens": configuration.max_new_tokens,
        },
        "truncation_fallback": {
            "attempts": [],
            "generation_mode": FALLBACK_GENERATION_MODE,
            "max_new_tokens": configuration.max_new_tokens,
            "trigger": "not_enabled",
        },
        "terminal_failure_disposition": terminal_failure_disposition,
        "attempt_timeout_seconds": attempt_timeout_seconds,
    }


def execution_policy_sha256(policy: Mapping[str, Any]) -> str:
    """Return the stable identity of one fully resolved execution policy."""

    return sha256(canonical_json_bytes(dict(policy))).hexdigest()


@contextmanager
def generation_wall_time_limit(seconds: float | None) -> Iterable[bool]:
    """Bound main-thread generation on platforms that provide interval timers."""

    supported = (
        seconds is not None
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
    )
    if not supported:
        yield False
        return
    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
    except (AttributeError, ValueError):
        yield False
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"LocateAnything generation exceeded {seconds:g} seconds")

    try:
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
    except (OSError, ValueError):
        try:
            signal.signal(signal.SIGALRM, previous_handler)
        except (OSError, ValueError):
            pass
        yield False
        return
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0 or previous_timer[1] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


EXECUTION_POLICY_SHA256 = sha256(
    canonical_json_bytes(locateanything_execution_policy())
).hexdigest()


@dataclass(frozen=True)
class LocateAnythingConfiguration:
    """Resolved settings for one proposal run."""

    model_id: str = DEFAULT_LOCATEANYTHING_MODEL
    model_revision: str = DEFAULT_LOCATEANYTHING_REVISION
    prompt: str = DEFAULT_PROMPT
    upscale_factor: int = DEFAULT_UPSCALE_FACTOR
    seed: int = DEFAULT_SEED
    max_new_tokens: int = PINNED_MAX_NEW_TOKENS
    generation_mode: str = PRIMARY_GENERATION_MODE
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1

    @property
    def sha256(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True)
class LocateAnythingGenerationPolicy:
    """Effective generation settings for one bounded retry attempt."""

    generation_mode: str
    max_new_tokens: int
    phase: str


def resolve_generation_policy(
    configuration: LocateAnythingConfiguration,
    attempt: int,
    execution_policy: Mapping[str, Any] | None = None,
) -> LocateAnythingGenerationPolicy:
    """Use the documented slow mode only after four pinned hybrid attempts."""

    if attempt < 1:
        raise ValueError("Proposal attempt must be positive")
    if execution_policy is not None:
        maximum_attempts = int(execution_policy["maximum_attempts_per_job"])
        if attempt > maximum_attempts:
            raise ValueError("Proposal attempt exceeds the active execution policy")
        primary = execution_policy["primary"]
        fallback = execution_policy["truncation_fallback"]
        if attempt in primary["attempts"]:
            return LocateAnythingGenerationPolicy(
                generation_mode=str(primary["generation_mode"]),
                max_new_tokens=int(primary["max_new_tokens"]),
                phase="primary",
            )
        if attempt in fallback["attempts"]:
            return LocateAnythingGenerationPolicy(
                generation_mode=str(fallback["generation_mode"]),
                max_new_tokens=int(fallback["max_new_tokens"]),
                phase="slow_fallback",
            )
        raise ValueError("Proposal attempt is not declared by the execution policy")
    if attempt <= PRIMARY_GENERATION_ATTEMPTS:
        return LocateAnythingGenerationPolicy(
            generation_mode=configuration.generation_mode,
            max_new_tokens=configuration.max_new_tokens,
            phase="primary",
        )
    return LocateAnythingGenerationPolicy(
        generation_mode=FALLBACK_GENERATION_MODE,
        max_new_tokens=configuration.max_new_tokens,
        phase="slow_fallback",
    )


@dataclass(frozen=True)
class ProposalJob:
    """One opaque RGB inference job."""

    cohort_id: str
    input_manifest_sha256: str
    opaque_scene_id: str
    view_id: str
    image_path: Path
    image_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.opaque_scene_id, self.view_id)


def _scaled_box(
    values: Sequence[int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = values
    x1 = max(0, min(width, round(x1 / 1000 * width)))
    y1 = max(0, min(height, round(y1 / 1000 * height)))
    x2 = max(0, min(width, round(x2 / 1000 * width)))
    y2 = max(0, min(height, round(y2 / 1000 * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def parse_locateanything_boxes(
    response: str, width: int, height: int
) -> list[list[int]]:
    """Parse normalized box tokens into deduplicated pixel coordinates."""

    patterns = (
        re.compile(r"<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>"),
        re.compile(
            r"<box>\s*(\d+)\s*[, ]+\s*(\d+)\s*[, ]+\s*(\d+)\s*[, ]+\s*(\d+)\s*</box>"
        ),
    )
    parsed: list[tuple[int, int, int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(str(response)):
            box = _scaled_box([int(value) for value in match.groups()], width, height)
            if box is not None:
                parsed.append(box)
    unique = sorted(
        set(parsed),
        key=lambda box: (-((box[2] - box[0]) * (box[3] - box[1])), box),
    )
    return [list(box) for box in unique]


def _safe_generation_stats(value: Any) -> Any:
    """Convert optional model timing statistics into bounded JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key)[:80]: _safe_generation_stats(item)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple)):
        return [_safe_generation_stats(item) for item in value[:100]]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _safe_generation_stats(item_method())
        except Exception:
            pass
    return f"<{type(value).__name__}>"


def _flatten_generation_stats(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_generation_stats(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_flatten_generation_stats(item, f"{prefix}[{index}]"))
    else:
        rows.append((prefix.lower(), value))
    return rows


def _detect_possible_truncation(
    generation_stats: Mapping[str, Any],
    response_token_count: int,
    max_new_tokens: int,
) -> tuple[bool, str]:
    """Prefer model statistics and otherwise use a labelled token heuristic."""

    flattened = _flatten_generation_stats(generation_stats)
    for key, value in flattened:
        if "trunc" in key and isinstance(value, bool):
            return value, "generation_stats"
    for key, value in flattened:
        if ("finish" in key or "stop" in key) and isinstance(value, str):
            reason = value.lower()
            if any(token in reason for token in ("length", "max_token", "token_limit")):
                return True, "generation_stats"
            if any(token in reason for token in ("eos", "stop", "end")):
                return False, "generation_stats"
    for key, value in flattened:
        if (
            "token" in key
            and any(token in key for token in ("generated", "output", "new"))
            and isinstance(value, (int, float))
        ):
            return value >= max_new_tokens - 4, "generation_stats"
    return response_token_count >= max_new_tokens - 4, "token_count_heuristic"


def build_proposal_jobs(model_input_root: Path, cohort_id: str) -> list[ProposalJob]:
    """Create a stable all-view job list from a verified opaque manifest."""

    model_input_root = model_input_root.expanduser().resolve()
    manifest_path = model_input_root / "neutral_corpus_manifest.jsonl"
    input_manifest_sha256 = file_sha256(manifest_path)
    rows = read_jsonl(manifest_path)
    jobs: list[ProposalJob] = []
    for row in sorted(rows, key=lambda value: str(value["opaque_scene_id"])):
        scene_id = str(row["opaque_scene_id"])
        for view_id in row["view_ids"]:
            image_path = model_input_root / str(row["media_root"]) / "rgb" / f"{view_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Opaque RGB input is missing: {scene_id}/{view_id}")
            jobs.append(
                ProposalJob(
                    cohort_id=cohort_id,
                    input_manifest_sha256=input_manifest_sha256,
                    opaque_scene_id=scene_id,
                    view_id=str(view_id),
                    image_path=image_path,
                    image_sha256=file_sha256(image_path),
                )
            )
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Proposal job list contains duplicate scene/view pairs")
    return jobs


def validate_proposal_result(
    row: Mapping[str, Any],
    execution_policy: Mapping[str, Any] | None = None,
) -> None:
    """Reject malformed or non-opaque proposal rows."""

    if set(row) != set(_RESULT_FIELDS):
        raise ValueError(f"Proposal result fields are not exact: {sorted(row)}")
    scene_id = str(row["opaque_scene_id"])
    view_id = str(row["view_id"])
    if not re.fullmatch(r"S\d{6}", scene_id):
        raise ValueError("Proposal result has an invalid opaque scene ID")
    if not re.fullmatch(r"V\d{2}", view_id):
        raise ValueError("Proposal result has an invalid view ID")
    if row["status"] not in {"ok", "error"}:
        raise ValueError("Proposal result has an invalid status")
    if row["producer_kind"] not in PRODUCER_KINDS:
        raise ValueError("Proposal result has an invalid producer kind")
    boxes = row["boxes_xyxy"]
    if not isinstance(boxes, list) or any(
        not isinstance(box, list)
        or len(box) != 4
        or any(not isinstance(value, int) for value in box)
        for box in boxes
    ):
        raise ValueError("Proposal boxes must be integer XYXY lists")
    if int(row["proposal_count"]) != len(boxes):
        raise ValueError("Proposal count does not match the stored boxes")
    if not isinstance(row["attempt"], int) or int(row["attempt"]) < 1:
        raise ValueError("Proposal attempt must be a positive integer")
    if not isinstance(row["response_token_count"], int) or int(row["response_token_count"]) < 0:
        raise ValueError("Response token count must be a non-negative integer")
    if not isinstance(row["possibly_truncated"], bool):
        raise ValueError("Truncation status must be Boolean")
    if row["truncation_check"] not in {
        "generation_stats",
        "token_count_heuristic",
        "not_available",
        "not_applicable",
    }:
        raise ValueError("Proposal result has an invalid truncation check")
    if not isinstance(row["generation_stats"], dict):
        raise ValueError("Generation statistics must be a JSON object")
    attempt = int(row["attempt"])
    active_policy = (
        locateanything_execution_policy()
        if execution_policy is None
        else dict(execution_policy)
    )
    maximum_attempts = int(active_policy["maximum_attempts_per_job"])
    if attempt > maximum_attempts:
        raise ValueError("Proposal attempt exceeds the active retry policy")
    expected_policy = resolve_generation_policy(
        LocateAnythingConfiguration(
            max_new_tokens=int(row["effective_max_new_tokens"]),
            generation_mode=str(active_policy["primary"]["generation_mode"]),
        ),
        attempt,
        active_policy,
    )
    if row["execution_policy_sha256"] != execution_policy_sha256(active_policy):
        raise ValueError("Proposal result uses another execution policy")
    if row["effective_generation_mode"] != expected_policy.generation_mode:
        raise ValueError("Proposal result has an invalid effective generation mode")
    if row["effective_max_new_tokens"] != expected_policy.max_new_tokens:
        raise ValueError("Proposal result has an invalid effective token limit")
    if (
        row["generation_stats"].get("effective_generation_mode")
        != row["effective_generation_mode"]
        or row["generation_stats"].get("effective_max_new_tokens")
        != row["effective_max_new_tokens"]
    ):
        raise ValueError("Generation statistics do not match the effective policy")
    for digest_name in ("image_sha256", "input_manifest_sha256", "configuration_sha256"):
        digest = str(row[digest_name])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Proposal result has an invalid {digest_name}")
    if row["status"] == "ok" and (
        row["error_type"] is not None or row["error_message"] is not None
    ):
        raise ValueError("Successful proposal rows cannot contain error details")
    if row["status"] == "error" and (
        not row["error_type"] or not isinstance(row["error_message"], str)
    ):
        raise ValueError("Failed proposal rows must contain sanitized error details")


class ProposalResultStore:
    """Append-only JSONL store with strict resume checks."""

    def __init__(
        self,
        path: Path,
        configuration_sha256: str,
        producer_kind: str,
        execution_policy: Mapping[str, Any] | None = None,
    ) -> None:
        if producer_kind not in PRODUCER_KINDS:
            raise ValueError("Proposal result store has an invalid producer kind")
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.configuration_sha256 = configuration_sha256
        self.producer_kind = producer_kind
        self.execution_policy = (
            locateanything_execution_policy()
            if execution_policy is None
            else dict(execution_policy)
        )
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._attempts: set[tuple[str, str, int]] = set()
        if self.path.exists():
            for row in read_jsonl(self.path):
                validate_proposal_result(row, self.execution_policy)
                if row["configuration_sha256"] != configuration_sha256:
                    raise ValueError("Existing proposal results use another configuration")
                if row["producer_kind"] != producer_kind:
                    raise ValueError("Existing proposal results use another producer kind")
                key = (str(row["opaque_scene_id"]), str(row["view_id"]))
                attempt_key = (*key, int(row["attempt"]))
                if attempt_key in self._attempts:
                    raise ValueError("Existing proposal results contain a duplicate attempt")
                previous = self._rows.get(key)
                if previous is None and int(row["attempt"]) != 1:
                    raise ValueError("Proposal history must start at attempt one")
                if previous is not None and int(row["attempt"]) <= int(previous["attempt"]):
                    raise ValueError("Proposal attempts must increase for each job")
                if previous is not None:
                    _validate_retry_transition(previous, row)
                self._attempts.add(attempt_key)
                self._history.append(row)
                self._rows[key] = row

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        return {
            key for key, row in self._rows.items() if row["status"] == "ok"
        }

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [self._rows[key] for key in sorted(self._rows)]

    @property
    def history_rows(self) -> list[dict[str, Any]]:
        return list(self._history)

    def next_attempt(self, job: ProposalJob) -> int:
        previous = self._rows.get(job.key)
        return int(previous["attempt"]) + 1 if previous is not None else 1

    def reconcile_jobs(self, jobs: Sequence[ProposalJob]) -> None:
        """Reject resume rows that do not match the current package."""

        expected = {job.key: job for job in jobs}
        extra = set(self._rows).difference(expected)
        if extra:
            raise ValueError("Existing proposal results contain jobs outside the input package")
        for key, row in self._rows.items():
            job = expected[key]
            if row["cohort_id"] != job.cohort_id:
                raise ValueError("Existing proposal result has another cohort")
            if row["input_manifest_sha256"] != job.input_manifest_sha256:
                raise ValueError("Existing proposal result has another input manifest")
            if row["image_sha256"] != job.image_sha256:
                raise ValueError("Existing proposal result has another image checksum")

    def append(self, row: Mapping[str, Any]) -> None:
        value = dict(row)
        validate_proposal_result(value, self.execution_policy)
        if value["configuration_sha256"] != self.configuration_sha256:
            raise ValueError("Proposal row uses another configuration")
        if value["producer_kind"] != self.producer_kind:
            raise ValueError("Proposal row uses another producer kind")
        key = (str(value["opaque_scene_id"]), str(value["view_id"]))
        attempt_key = (*key, int(value["attempt"]))
        if attempt_key in self._attempts:
            raise ValueError("Refusing to append a duplicate proposal attempt")
        previous = self._rows.get(key)
        if previous is None and int(value["attempt"]) != 1:
            raise ValueError("Proposal history must start at attempt one")
        if previous is not None and int(value["attempt"]) != int(previous["attempt"]) + 1:
            raise ValueError("Proposal retry attempt is not sequential")
        if previous is not None:
            _validate_retry_transition(previous, value)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json_bytes(value).decode("utf-8") + "\n")
            handle.flush()
        self._attempts.add(attempt_key)
        self._history.append(value)
        self._rows[key] = value


def _validate_retry_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    """Permit slow fallback only after a directly preceding truncation."""

    current_attempt = int(current["attempt"])
    if current_attempt <= PRIMARY_GENERATION_ATTEMPTS:
        return
    if previous.get("status") != "error" or previous.get("error_type") != "PossibleTruncation":
        raise ValueError("Slow fallback requires a directly preceding possible truncation")
    if int(previous["attempt"]) != current_attempt - 1:
        raise ValueError("Slow fallback must follow the preceding proposal attempt")


class LocateAnythingWorker:
    """Load the pinned model once and generate boxes for RGB images."""

    def __init__(
        self,
        configuration: LocateAnythingConfiguration,
        *,
        device: str = "cuda",
        dtype: Any | None = None,
        model_source: str | Path | None = None,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        execution_policy: Mapping[str, Any] | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.configuration = configuration
        self.execution_policy = (
            locateanything_execution_policy()
            if execution_policy is None
            else dict(execution_policy)
        )
        self.device = device
        self.producer_kind = MODEL_PRODUCER_KIND
        source = str(model_source or configuration.model_id)
        self.dtype = dtype or (
            torch.bfloat16
            if device.startswith("cuda")
            and torch.cuda.get_device_properties(0).major >= 8
            else torch.float16
        )
        common = {"trust_remote_code": True}
        if model_source is None:
            common["revision"] = configuration.model_revision
        source_kind = "local_snapshot" if model_source is not None else "hub"

        def report(stage: str, **details: Any) -> None:
            if progress_callback is not None:
                progress_callback(stage, details)

        report("dependencies_ready", device=device, dtype=str(self.dtype))
        report("tokenizer_load_started", source_kind=source_kind)
        self.tokenizer = AutoTokenizer.from_pretrained(
            source, **common
        )
        report("tokenizer_ready")
        report("processor_load_started", source_kind=source_kind)
        self.processor = AutoProcessor.from_pretrained(
            source, use_fast=False, **common
        )
        report("processor_ready")
        report("model_weights_load_started", source_kind=source_kind)
        model = AutoModel.from_pretrained(
            source,
            torch_dtype=self.dtype,
            **common,
        )
        report("model_weights_loaded")
        report("model_device_move_started", device=device)
        self.model = model.to(device).eval()
        report("model_ready", device=device)

    def use_configuration(self, configuration: LocateAnythingConfiguration) -> None:
        """Change only inference settings while keeping the loaded model fixed."""

        if (
            configuration.model_id != self.configuration.model_id
            or configuration.model_revision != self.configuration.model_revision
            or configuration.upscale_factor != self.configuration.upscale_factor
        ):
            raise ValueError(
                "A loaded LocateAnything worker cannot change model identity or image scale"
            )
        execution_policy = getattr(
            self, "execution_policy", locateanything_execution_policy()
        )
        primary = execution_policy["primary"]
        if (
            configuration.max_new_tokens != int(primary["max_new_tokens"])
            or configuration.generation_mode != str(primary["generation_mode"])
        ):
            raise ValueError(
                "A loaded LocateAnything worker cannot leave its execution policy"
            )
        self.configuration = configuration

    def _job_seed(self, scene_id: str, view_id: str, attempt: int = 1) -> int:
        if attempt < 1:
            raise ValueError("Proposal attempt must be positive")
        payload = (
            f"{self.configuration.seed}|{scene_id}|{view_id}|{attempt}".encode("utf-8")
        )
        return int.from_bytes(sha256(payload).digest()[:4], "big")

    def predict(
        self,
        image: Image.Image,
        scene_id: str,
        view_id: str,
        *,
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Return the model answer and optional generation statistics."""

        import torch

        execution_policy = getattr(
            self, "execution_policy", locateanything_execution_policy()
        )
        generation_policy = resolve_generation_policy(
            self.configuration, attempt, execution_policy
        )

        prompt = (
            "Locate all the instances that match the following description: "
            f"{self.configuration.prompt}."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)
        seed = self._job_seed(scene_id, view_id, attempt)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        attempt_timeout_seconds = execution_policy.get(
            "attempt_timeout_seconds"
        )
        with torch.inference_mode(), generation_wall_time_limit(
            attempt_timeout_seconds
        ) as timeout_guard_active:
            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=generation_policy.max_new_tokens,
                use_cache=True,
                generation_mode=generation_policy.generation_mode,
                do_sample=self.configuration.do_sample,
                temperature=self.configuration.temperature,
                top_p=self.configuration.top_p,
                top_k=(
                    None
                    if self.configuration.top_k <= 0
                    else self.configuration.top_k
                ),
                repetition_penalty=self.configuration.repetition_penalty,
                verbose=False,
            )
        result: dict[str, Any] = {
            "answer": response[0] if isinstance(response, tuple) else response,
            "attempt_timeout_seconds": attempt_timeout_seconds,
            "attempt_timeout_guard_active": timeout_guard_active,
        }
        if isinstance(response, tuple) and len(response) >= 3:
            result["stats"] = response[2]
        return result

    def run_job(self, job: ProposalJob, *, attempt: int = 1) -> dict[str, Any]:
        """Run one job and return a schema-checked result row."""

        started = time.perf_counter()
        execution_policy = getattr(
            self, "execution_policy", locateanything_execution_policy()
        )
        generation_policy = resolve_generation_policy(
            self.configuration, attempt, execution_policy
        )
        try:
            with Image.open(job.image_path) as opened:
                image = opened.convert("RGB")
            original_size = image.size
            if self.configuration.upscale_factor != 1:
                image = image.resize(
                    (
                        original_size[0] * self.configuration.upscale_factor,
                        original_size[1] * self.configuration.upscale_factor,
                    ),
                    Image.Resampling.BICUBIC,
                )
            prediction = self.predict(
                image,
                job.opaque_scene_id,
                job.view_id,
                attempt=attempt,
            )
            if isinstance(prediction, Mapping):
                raw_response = str(prediction.get("answer", ""))
                stats_value = _safe_generation_stats(prediction.get("stats", {}))
                generation_stats = (
                    stats_value
                    if isinstance(stats_value, dict)
                    else {"value": stats_value}
                )
                generation_stats["attempt_timeout_seconds"] = prediction.get(
                    "attempt_timeout_seconds"
                )
                generation_stats["attempt_timeout_guard_active"] = bool(
                    prediction.get("attempt_timeout_guard_active", False)
                )
            else:
                raw_response = str(prediction)
                generation_stats = {}
            response_token_count = len(
                self.tokenizer.encode(raw_response, add_special_tokens=False)
            )
            possibly_truncated, truncation_check = _detect_possible_truncation(
                generation_stats,
                response_token_count,
                generation_policy.max_new_tokens,
            )
            boxes = parse_locateanything_boxes(
                raw_response, original_size[0], original_size[1]
            )
            if possibly_truncated:
                status = "error"
                error_type = "PossibleTruncation"
                error_message = "The response reached the configured token-limit warning boundary"
            elif not boxes:
                status = "error"
                error_type = "NoValidBoxes"
                error_message = "No valid box tokens were parsed from the model response"
            else:
                status = "ok"
                error_type = None
                error_message = None
        except Exception as error:
            raw_response = ""
            generation_stats = {}
            response_token_count = 0
            possibly_truncated = False
            truncation_check = "not_available"
            boxes = []
            status = "error"
            error_type = type(error).__name__
            error_message = f"Proposal worker raised {type(error).__name__}"
        generation_stats.update(
            {
                "effective_generation_mode": generation_policy.generation_mode,
                "effective_max_new_tokens": generation_policy.max_new_tokens,
                "retry_policy_phase": generation_policy.phase,
                "attempt_timeout_seconds": execution_policy.get(
                    "attempt_timeout_seconds"
                ),
            }
        )
        row = {
            "schema_version": 1,
            "cohort_id": job.cohort_id,
            "input_manifest_sha256": job.input_manifest_sha256,
            "opaque_scene_id": job.opaque_scene_id,
            "view_id": job.view_id,
            "attempt": attempt,
            "image_sha256": job.image_sha256,
            "configuration_sha256": self.configuration.sha256,
            "execution_policy_sha256": execution_policy_sha256(execution_policy),
            "effective_generation_mode": generation_policy.generation_mode,
            "effective_max_new_tokens": generation_policy.max_new_tokens,
            "producer_kind": MODEL_PRODUCER_KIND,
            "model_id": self.configuration.model_id,
            "model_revision": self.configuration.model_revision,
            "prompt": self.configuration.prompt,
            "proposal_count": len(boxes),
            "boxes_xyxy": boxes,
            "raw_response": raw_response,
            "generation_stats": generation_stats,
            "response_token_count": response_token_count,
            "possibly_truncated": possibly_truncated,
            "truncation_check": truncation_check,
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
        }
        validate_proposal_result(row, execution_policy)
        return row


class ProposalContractWorker:
    """Deterministic non-model worker for local pipeline contract tests."""

    def __init__(
        self,
        configuration: LocateAnythingConfiguration,
        *,
        execution_policy: Mapping[str, Any] | None = None,
    ) -> None:
        self.configuration = configuration
        self.producer_kind = CONTRACT_PRODUCER_KIND
        self.execution_policy = (
            locateanything_execution_policy()
            if execution_policy is None
            else dict(execution_policy)
        )

    def run_job(self, job: ProposalJob, *, attempt: int = 1) -> dict[str, Any]:
        started = time.perf_counter()
        generation_policy = resolve_generation_policy(
            self.configuration, attempt, self.execution_policy
        )
        with Image.open(job.image_path) as opened:
            width, height = opened.size
        digest = sha256(
            f"{job.image_sha256}|{self.configuration.sha256}".encode("utf-8")
        ).digest()
        count = 1 + digest[0] % 5
        normalized_boxes: list[tuple[int, int, int, int]] = []
        for index in range(count):
            offset = index * 4
            x1 = 30 + digest[(offset + 1) % len(digest)] % 500
            y1 = 30 + digest[(offset + 2) % len(digest)] % 500
            box_width = 120 + digest[(offset + 3) % len(digest)] % 250
            box_height = 120 + digest[(offset + 4) % len(digest)] % 250
            normalized_boxes.append(
                (x1, y1, min(980, x1 + box_width), min(980, y1 + box_height))
            )
        raw_response = " ".join(
            f"<box><{x1}><{y1}><{x2}><{y2}></box>"
            for x1, y1, x2, y2 in normalized_boxes
        )
        boxes = parse_locateanything_boxes(raw_response, width, height)
        row = {
            "schema_version": 1,
            "cohort_id": job.cohort_id,
            "input_manifest_sha256": job.input_manifest_sha256,
            "opaque_scene_id": job.opaque_scene_id,
            "view_id": job.view_id,
            "attempt": attempt,
            "image_sha256": job.image_sha256,
            "configuration_sha256": self.configuration.sha256,
            "execution_policy_sha256": execution_policy_sha256(self.execution_policy),
            "effective_generation_mode": generation_policy.generation_mode,
            "effective_max_new_tokens": generation_policy.max_new_tokens,
            "producer_kind": CONTRACT_PRODUCER_KIND,
            "model_id": self.configuration.model_id,
            "model_revision": self.configuration.model_revision,
            "prompt": self.configuration.prompt,
            "proposal_count": len(boxes),
            "boxes_xyxy": boxes,
            "raw_response": raw_response,
            "generation_stats": {
                "effective_generation_mode": generation_policy.generation_mode,
                "effective_max_new_tokens": generation_policy.max_new_tokens,
                "retry_policy_phase": generation_policy.phase,
                "attempt_timeout_seconds": self.execution_policy.get(
                    "attempt_timeout_seconds"
                ),
            },
            "response_token_count": len(normalized_boxes) * 6,
            "possibly_truncated": False,
            "truncation_check": "not_applicable",
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "status": "ok",
            "error_type": None,
            "error_message": None,
        }
        validate_proposal_result(row, self.execution_policy)
        return row


def run_proposal_jobs(
    *,
    worker: Any,
    jobs: Sequence[ProposalJob],
    result_store: ProposalResultStore,
    maximum_attempts_per_job: int = 2,
    maximum_consecutive_failed_jobs: int | None = 3,
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run resumable jobs with bounded retries and checkpoint callbacks."""

    if maximum_attempts_per_job < 1:
        raise ValueError("At least one proposal attempt is required")
    if (
        maximum_consecutive_failed_jobs is not None
        and maximum_consecutive_failed_jobs < 1
    ):
        raise ValueError("Consecutive proposal failure limit must be positive")
    result_store.reconcile_jobs(jobs)
    pending = [job for job in jobs if job.key not in result_store.completed_keys]
    consecutive_failed_jobs = 0
    attempted_rows = 0
    session_started = time.perf_counter()
    for completed_index, job in enumerate(pending, 1):
        existing_rows = [
            row
            for row in result_store.history_rows
            if (row["opaque_scene_id"], row["view_id"]) == job.key
        ]
        final_row: dict[str, Any] | None = existing_rows[-1] if existing_rows else None
        while result_store.next_attempt(job) <= maximum_attempts_per_job:
            next_attempt = result_store.next_attempt(job)
            if (
                next_attempt > PRIMARY_GENERATION_ATTEMPTS
                and final_row is not None
                and final_row.get("error_type") != "PossibleTruncation"
            ):
                break
            final_row = worker.run_job(
                job,
                attempt=next_attempt,
            )
            result_store.append(final_row)
            attempted_rows += 1
            if final_row["status"] == "ok":
                break
        if final_row is None:
            final_row = next(
                row
                for row in result_store.rows
                if (row["opaque_scene_id"], row["view_id"]) == job.key
            )
        consecutive_failed_jobs = (
            consecutive_failed_jobs + 1 if final_row["status"] == "error" else 0
        )
        if progress_callback is not None:
            progress_callback(completed_index, len(pending), final_row)
        if (
            maximum_consecutive_failed_jobs is not None
            and consecutive_failed_jobs >= maximum_consecutive_failed_jobs
        ):
            raise RuntimeError(
                f"{maximum_consecutive_failed_jobs} consecutive proposal jobs failed"
            )
    return {
        "pending_job_count_at_start": len(pending),
        "attempted_rows_this_session": attempted_rows,
        "elapsed_seconds": round(time.perf_counter() - session_started, 4),
        "successful_job_count": len(result_store.completed_keys),
        "latest_row_count": len(result_store.rows),
        "history_row_count": len(result_store.history_rows),
    }


def repeat_agreement(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare parsed boxes for two runs over the same ordered jobs."""

    first = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): dict(row)
        for row in first_rows
    }
    second = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): dict(row)
        for row in second_rows
    }
    if set(first) != set(second):
        raise ValueError("Repeat runs do not contain the same jobs")
    matching = sum(
        first[key]["boxes_xyxy"] == second[key]["boxes_xyxy"]
        and first[key]["status"] == second[key]["status"]
        for key in first
    )
    return {
        "job_count": len(first),
        "matching_job_count": matching,
        "box_agreement_rate": matching / len(first) if first else 0.0,
    }


def summarize_proposal_rows(
    rows: Iterable[Mapping[str, Any]],
    execution_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return aggregate operational evidence without evaluator labels."""

    values = [dict(row) for row in rows]
    for row in values:
        validate_proposal_result(row, execution_policy)
    successful = [row for row in values if row["status"] == "ok"]
    runtimes = [float(row["runtime_seconds"]) for row in values]
    proposal_counts = [int(row["proposal_count"]) for row in successful]
    sorted_runtimes = sorted(runtimes)
    p95_index = max(0, round(0.95 * len(sorted_runtimes) + 0.5) - 1)
    return {
        "job_count": len(values),
        "successful_job_count": len(successful),
        "failed_job_count": len(values) - len(successful),
        "scene_count": len({str(row["opaque_scene_id"]) for row in values}),
        "mean_proposals_per_successful_image": (
            mean(proposal_counts) if proposal_counts else 0.0
        ),
        "median_proposals_per_successful_image": (
            median(proposal_counts) if proposal_counts else 0.0
        ),
        "minimum_proposals_per_successful_image": (
            min(proposal_counts) if proposal_counts else 0
        ),
        "maximum_proposals_per_successful_image": (
            max(proposal_counts) if proposal_counts else 0
        ),
        "zero_proposal_image_count": sum(value == 0 for value in proposal_counts),
        "possibly_truncated_count": sum(
            bool(row["possibly_truncated"]) for row in values
        ),
        "truncation_check_counts": dict(
            sorted(Counter(str(row["truncation_check"]) for row in values).items())
        ),
        "mean_runtime_seconds": mean(runtimes) if runtimes else 0.0,
        "median_runtime_seconds": median(runtimes) if runtimes else 0.0,
        "p95_runtime_seconds": (
            sorted_runtimes[p95_index] if sorted_runtimes else 0.0
        ),
        "total_runtime_seconds": sum(runtimes),
        "images_per_second": (
            len(values) / sum(runtimes) if runtimes and sum(runtimes) > 0 else 0.0
        ),
        "configuration_sha256": (
            str(values[0]["configuration_sha256"]) if values else ""
        ),
    }


def collect_proposal_evidence(
    rows: Iterable[Mapping[str, Any]],
    jobs: Sequence[ProposalJob],
    history_rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    expected_job_count: int = 400,
    expected_scene_count: int = 20,
    expected_view_count: int = 20,
) -> dict[str, Any]:
    """Collect aggregate tables, plotting values and private overlay inputs."""

    if min(expected_job_count, expected_scene_count, expected_view_count) < 1:
        raise ValueError("Expected proposal evidence counts must be positive")

    values = [dict(row) for row in rows]
    for row in values:
        validate_proposal_result(row)
    latest = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in values
    }
    job_by_key = {job.key: job for job in jobs}
    if set(latest) != set(job_by_key):
        raise ValueError("Proposal evidence requires one latest row for every job")
    view_groups: dict[str, list[dict[str, Any]]] = {}
    scene_groups: dict[str, list[dict[str, Any]]] = {}
    error_types: Counter[str] = Counter()
    attempt_counts: Counter[tuple[str, str]] = Counter()
    for row in history_rows or values:
        validate_proposal_result(row)
        attempt_counts[(str(row["opaque_scene_id"]), str(row["view_id"]))] += 1
    for row in latest.values():
        view_groups.setdefault(str(row["view_id"]), []).append(row)
        scene_groups.setdefault(str(row["opaque_scene_id"]), []).append(row)
        if row["error_type"]:
            error_types[str(row["error_type"])] += 1

    def aggregate(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        successful = [row for row in group if row["status"] == "ok"]
        return {
            "image_count": len(group),
            "successful_count": len(successful),
            "failed_count": len(group) - len(successful),
            "mean_proposal_count": (
                round(mean(int(row["proposal_count"]) for row in successful), 4)
                if successful
                else 0.0
            ),
            "mean_runtime_seconds": (
                round(mean(float(row["runtime_seconds"]) for row in group), 4)
                if group
                else 0.0
            ),
            "truncated_count": sum(bool(row["possibly_truncated"]) for row in group),
        }

    view_rows = [
        {"view_id": view_id, **aggregate(group)}
        for view_id, group in sorted(view_groups.items())
    ]
    scene_rows = [
        {"scene_index": index, **aggregate(scene_groups[scene_id])}
        for index, scene_id in enumerate(sorted(scene_groups), 1)
    ]
    summary = summarize_proposal_rows(latest.values())
    operational_rows: list[dict[str, Any]] = []
    box_area_fractions: list[float] = []
    box_aspect_ratios: list[float] = []
    invalid_box_count = 0
    duplicate_box_count = 0
    for job_index, key in enumerate(sorted(latest), 1):
        row = latest[key]
        job = job_by_key[key]
        with Image.open(job.image_path) as opened:
            width, height = opened.size
        boxes = [tuple(int(value) for value in box) for box in row["boxes_xyxy"]]
        duplicate_box_count += len(boxes) - len(set(boxes))
        row_areas: list[float] = []
        for x1, y1, x2, y2 in boxes:
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                invalid_box_count += 1
                continue
            box_width = x2 - x1
            box_height = y2 - y1
            area_fraction = box_width * box_height / (width * height)
            row_areas.append(area_fraction)
            box_area_fractions.append(area_fraction)
            box_aspect_ratios.append(box_width / box_height)
        operational_rows.append(
            {
                "job_index": job_index,
                "view_id": row["view_id"],
                "status": row["status"],
                "attempt_count": attempt_counts[key],
                "proposal_count": row["proposal_count"],
                "response_token_count": row["response_token_count"],
                "truncation_check": row["truncation_check"],
                "runtime_seconds": row["runtime_seconds"],
                "mean_box_area_fraction": (
                    round(mean(row_areas), 6) if row_areas else 0.0
                ),
            }
        )
    attempt_distribution = Counter(row["attempt_count"] for row in operational_rows)
    attempt_rows = [
        {"attempt_count": attempt_count, "image_count": image_count}
        for attempt_count, image_count in sorted(attempt_distribution.items())
    ]
    geometry_summary = {
        "box_count": len(box_area_fractions),
        "invalid_box_count": invalid_box_count,
        "duplicate_box_count": duplicate_box_count,
        "mean_box_area_fraction": (
            mean(box_area_fractions) if box_area_fractions else 0.0
        ),
        "median_box_area_fraction": (
            median(box_area_fractions) if box_area_fractions else 0.0
        ),
        "mean_box_aspect_ratio": (
            mean(box_aspect_ratios) if box_aspect_ratios else 0.0
        ),
        "median_box_aspect_ratio": (
            median(box_aspect_ratios) if box_aspect_ratios else 0.0
        ),
        "total_attempt_count": sum(attempt_counts.values()),
        "retried_image_count": sum(count > 1 for count in attempt_counts.values()),
    }
    gate_rows = [
        {
            "gate": "unique jobs",
            "status": (
                "pass" if summary["job_count"] == expected_job_count else "fail"
            ),
            "observed": summary["job_count"],
            "expected": expected_job_count,
        },
        {
            "gate": "successful jobs",
            "status": (
                "pass"
                if summary["successful_job_count"] == expected_job_count
                else "fail"
            ),
            "observed": summary["successful_job_count"],
            "expected": expected_job_count,
        },
        {
            "gate": "failed jobs",
            "status": "pass" if summary["failed_job_count"] == 0 else "fail",
            "observed": summary["failed_job_count"],
            "expected": 0,
        },
        {
            "gate": "truncated responses",
            "status": (
                "pass" if summary["possibly_truncated_count"] == 0 else "fail"
            ),
            "observed": summary["possibly_truncated_count"],
            "expected": 0,
        },
        {
            "gate": "scene coverage",
            "status": (
                "pass"
                if summary["scene_count"] == expected_scene_count
                else "fail"
            ),
            "observed": summary["scene_count"],
            "expected": expected_scene_count,
        },
        {
            "gate": "view coverage",
            "status": (
                "pass" if len(view_rows) == expected_view_count else "fail"
            ),
            "observed": len(view_rows),
            "expected": expected_view_count,
        },
        {
            "gate": "zero-proposal successes",
            "status": (
                "pass" if summary["zero_proposal_image_count"] == 0 else "fail"
            ),
            "observed": summary["zero_proposal_image_count"],
            "expected": 0,
        },
        {
            "gate": "invalid boxes",
            "status": "pass" if invalid_box_count == 0 else "fail",
            "observed": invalid_box_count,
            "expected": 0,
        },
        {
            "gate": "duplicate boxes",
            "status": "pass" if duplicate_box_count == 0 else "fail",
            "observed": duplicate_box_count,
            "expected": 0,
        },
    ]
    overlay_items: list[dict[str, Any]] = []
    coverage_scenes = sorted(scene_groups)[:20]
    coverage_views = sorted(view_groups)[:20]
    for scene_index, (scene_id, view_id) in enumerate(
        zip(coverage_scenes, coverage_views, strict=True), 1
    ):
        key = (scene_id, view_id)
        job = job_by_key[key]
        row = latest[key]
        overlay_items.append(
            {
                "label": f"Scene {scene_index:02d} / {view_id}",
                "path": job.image_path,
                "boxes": row["boxes_xyxy"],
            }
        )
    ranked_outliers: list[tuple[str, dict[str, Any]]] = []
    for reason, ordered in (
        (
            "highest count",
            sorted(
                operational_rows,
                key=lambda row: (
                    -int(row["proposal_count"]),
                    int(row["job_index"]),
                ),
            )[:5],
        ),
        (
            "slowest",
            sorted(
                operational_rows,
                key=lambda row: (
                    -float(row["runtime_seconds"]),
                    int(row["job_index"]),
                ),
            )[:5],
        ),
        (
            "largest response",
            sorted(
                operational_rows,
                key=lambda row: (
                    -int(row["response_token_count"]),
                    int(row["job_index"]),
                ),
            )[:5],
        ),
    ):
        ranked_outliers.extend((reason, row) for row in ordered)
    outlier_rows: list[dict[str, Any]] = []
    outlier_overlay_items: list[dict[str, Any]] = []
    seen_outliers: set[int] = set()
    sorted_keys = sorted(latest)
    for reason, row in ranked_outliers:
        job_index = int(row["job_index"])
        outlier_rows.append({"reason": reason, **row})
        if job_index in seen_outliers:
            continue
        seen_outliers.add(job_index)
        key = sorted_keys[job_index - 1]
        job = job_by_key[key]
        result = latest[key]
        outlier_overlay_items.append(
            {
                "label": f"{reason} / job {job_index:03d} / {result['view_id']}",
                "path": job.image_path,
                "boxes": result["boxes_xyxy"],
            }
        )
    return {
        "summary": summary,
        "gate_rows": gate_rows,
        "view_rows": view_rows,
        "scene_rows": scene_rows,
        "failure_rows": [
            {"error_type": name, "count": count}
            for name, count in sorted(error_types.items())
        ],
        "proposal_counts": [int(row["proposal_count"]) for row in latest.values()],
        "runtime_seconds": [float(row["runtime_seconds"]) for row in latest.values()],
        "response_token_counts": [int(row["response_token_count"]) for row in latest.values()],
        "box_area_fractions": box_area_fractions,
        "box_aspect_ratios": box_aspect_ratios,
        "operational_rows": operational_rows,
        "attempt_rows": attempt_rows,
        "geometry_summary": geometry_summary,
        "outlier_rows": outlier_rows,
        "overlay_items": overlay_items,
        "outlier_overlay_items": outlier_overlay_items,
    }


__all__ = [
    "CONTRACT_PRODUCER_KIND",
    "DEFAULT_LOCATEANYTHING_MODEL",
    "DEFAULT_LOCATEANYTHING_REVISION",
    "EXECUTION_POLICY_SHA256",
    "FALLBACK_GENERATION_MODE",
    "LocateAnythingConfiguration",
    "LocateAnythingGenerationPolicy",
    "LocateAnythingWorker",
    "MAXIMUM_GENERATION_ATTEMPTS",
    "MODEL_PRODUCER_KIND",
    "PINNED_MAX_NEW_TOKENS",
    "PRIMARY_GENERATION_ATTEMPTS",
    "PRIMARY_GENERATION_MODE",
    "ProposalContractWorker",
    "ProposalJob",
    "ProposalResultStore",
    "bounded_locateanything_execution_policy",
    "build_proposal_jobs",
    "collect_proposal_evidence",
    "execution_policy_sha256",
    "generation_wall_time_limit",
    "locateanything_execution_policy",
    "parse_locateanything_boxes",
    "repeat_agreement",
    "resolve_generation_policy",
    "run_proposal_jobs",
    "summarize_proposal_rows",
    "validate_proposal_result",
]
