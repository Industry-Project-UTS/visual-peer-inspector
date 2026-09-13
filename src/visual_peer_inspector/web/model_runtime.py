"""Lazy GPU adapters. Checkpoints are fetched only inside an inference worker."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata

import numpy as np

from .deployment import pipeline_config


class RuntimeUnavailable(RuntimeError):
    pass


class GPUModels:
    def __init__(self, *, allow_downloads=False):
        self.allow_downloads = allow_downloads
        self.config = pipeline_config()
        self.resource = None
        self.stage = None

    def _torch(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeUnavailable("cuda_unavailable")
        return torch

    def _snapshot(self, config):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return snapshot_download(
                repo_id=config["model_id"],
                revision=config["revision"],
                local_files_only=not self.allow_downloads,
                ignore_patterns=["*.md", "*.png", "*.jpg", "*.mp4", ".gitattributes"],
            )
        except LocalEntryNotFoundError as error:
            raise RuntimeUnavailable("models_not_cached") from error

    def release(self):
        self.resource = None
        self.stage = None
        gc.collect()
        # Releasing an unused adapter does not even import torch.
        import sys

        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load(self, stage):
        if self.stage == stage:
            return self.resource
        self.release()
        self._torch()
        if stage == "locate":
            from visual_peer_inspector.models.locateanything import (
                LocateAnythingConfiguration,
                LocateAnythingWorker,
                bounded_locateanything_execution_policy,
            )

            config = self.config["locateanything"]
            generation = LocateAnythingConfiguration(
                model_id=config["model_id"],
                model_revision=config["revision"],
                prompt=config["prompt"],
                max_new_tokens=config["max_new_tokens"],
                generation_mode=config["generation_mode"],
                seed=config["seed"],
                upscale_factor=config["upscale_factor"],
                do_sample=config["do_sample"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                repetition_penalty=config["repetition_penalty"],
            )
            policy = bounded_locateanything_execution_policy(
                generation,
                maximum_attempts_per_job=config["maximum_attempts"],
                terminal_failure_disposition="abort",
            )
            self.resource = LocateAnythingWorker(
                generation, model_source=self._snapshot(config), execution_policy=policy
            )
        elif stage == "sam":
            from pathlib import Path

            from visual_peer_inspector.models.sam2 import (
                Sam2Configuration,
                Sam2ImageWorker,
            )

            config = self.config["sam2"]
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                checkpoint = Path(
                    hf_hub_download(
                        config["model_id"],
                        config["filename"],
                        revision=config["revision"],
                        local_files_only=not self.allow_downloads,
                    )
                )
            except LocalEntryNotFoundError as error:
                raise RuntimeUnavailable("models_not_cached") from error
            if checkpoint.stat().st_size != config["size_bytes"]:
                raise RuntimeUnavailable("checkpoint_integrity")
            with checkpoint.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != config["sha256"]:
                raise RuntimeUnavailable("checkpoint_integrity")
            self.resource = Sam2ImageWorker(
                Sam2Configuration(
                    model_repository=config["model_id"],
                    model_revision=config["revision"],
                    code_revision=config["code_revision"],
                    checkpoint_filename=config["filename"],
                    checkpoint_sha256=config["sha256"],
                    checkpoint_size_bytes=config["size_bytes"],
                    model_config=config["model_config"],
                    precision=config["precision"],
                    seed=config["seed"],
                    multimask_output=config["multimask_output"],
                    mask_threshold=config["mask_threshold"],
                    max_hole_area=config["max_hole_area"],
                    max_sprinkle_area=config["max_sprinkle_area"],
                ),
                checkpoint,
            )
        elif stage == "encoder":
            from visual_peer_inspector.models.visual_encoder import (
                FrozenVisualEncoder,
                specification_by_key,
            )

            config = self.config["encoder"]
            specification = specification_by_key(config["key"])
            if specification.model_id != config["model_id"]:
                raise ValueError(
                    "Encoder manifest differs from the selected implementation."
                )
            # Populate the exact cache entry first; an offline worker never fetches weights.
            self._snapshot(config)
            self.resource = FrozenVisualEncoder(
                specification,
                revision=config["revision"],
                device="cuda",
            )
        else:
            raise ValueError("Unknown model stage")
        self.stage = stage
        return self.resource

    def propose(self, image, view_id):
        from visual_peer_inspector.models.locateanything import (
            _detect_possible_truncation,
            parse_locateanything_boxes,
        )
        from visual_peer_inspector.workflows.proposal_refinement_ds1 import (
            ProposalPolicy,
            apply_proposal_policy,
        )

        worker = self._load("locate")
        config = self.config["locateanything"]
        for attempt in range(1, config["maximum_attempts"] + 1):
            result = worker.predict(image, "uploaded_scene", view_id, attempt=attempt)
            answer = str(result["answer"])
            token_count = len(worker.tokenizer.encode(answer, add_special_tokens=False))
            truncated, _ = _detect_possible_truncation(
                result.get("stats", {}), token_count, config["max_new_tokens"]
            )
            if truncated:
                continue
            boxes = parse_locateanything_boxes(answer, image.width, image.height)
            kept, _, removed = apply_proposal_policy(
                boxes, ProposalPolicy(**self.config["proposal_policy"])
            )
            notes = []
            if removed.get("proposal_cap"):
                notes.append(
                    "More than eight objects were proposed; review the retained boxes."
                )
            return kept, notes
        raise RuntimeUnavailable("proposal_truncated")

    def segment(self, image, boxes):
        worker = self._load("sam")
        torch = worker.torch
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                worker.predictor.set_image(np.asarray(image))
                masks, scores, _ = worker.predictor.predict(
                    box=np.asarray(boxes, dtype=np.float32), multimask_output=False
                )
            torch.cuda.synchronize()
        finally:
            worker.predictor.reset_predictor()
        masks = (
            np.asarray(masks)
            .reshape(len(boxes), image.height, image.width)
            .astype(bool)
        )
        return masks, np.asarray(scores).reshape(len(boxes))

    def encode(self, crops):
        return self._load("encoder").encode(crops)


def runtime_versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "huggingface-hub",
            "numpy",
            "pillow",
            "SAM-2",
        )
    }
