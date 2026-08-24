"""Frozen visual encoder adapters used by the bounded Notebook 07 screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Sequence

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderSpecification:
    key: str
    model_id: str
    family: str
    backend: str
    screen_stage: str
    parameters_millions: int
    supports_patches: bool = True
    gated: bool = False


ENCODER_REGISTRY = (
    EncoderSpecification(
        "dinov2_small",
        "facebook/dinov2-small",
        "DINOv2",
        "transformers_dino",
        "core",
        21,
    ),
    EncoderSpecification(
        "dinov2_small_registers",
        "facebook/dinov2-with-registers-small",
        "DINOv2",
        "transformers_dino",
        "core",
        21,
    ),
    EncoderSpecification(
        "dinov3_small",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        "DINOv3",
        "transformers_dino",
        "core",
        21,
        gated=True,
    ),
    EncoderSpecification(
        "dinov3_small_plus",
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "DINOv3",
        "transformers_dino",
        "core",
        29,
        gated=True,
    ),
    EncoderSpecification(
        "dinov2_base",
        "facebook/dinov2-base",
        "DINOv2",
        "transformers_dino",
        "confirmation",
        86,
    ),
    EncoderSpecification(
        "dinov3_base",
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "DINOv3",
        "transformers_dino",
        "confirmation",
        86,
        gated=True,
    ),
    EncoderSpecification(
        "resnet50_control",
        "microsoft/resnet-50",
        "ResNet",
        "transformers_resnet",
        "control",
        26,
        supports_patches=False,
    ),
)


@dataclass(frozen=True)
class EncodedBatch:
    global_features: np.ndarray
    patch_features: tuple[np.ndarray, ...]
    patch_grid_size: int | None
    feature_dimension: int


def encoder_registry_rows() -> list[dict[str, Any]]:
    return [asdict(item) for item in ENCODER_REGISTRY]


def encoder_registry_sha256() -> str:
    payload = json.dumps(
        encoder_registry_rows(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_libraries() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "Frozen encoder inference requires torch and transformers"
        ) from error
    return torch, functional, AutoModel


class FrozenVisualEncoder:
    """Load one immutable model revision and expose global and spatial features."""

    def __init__(
        self,
        specification: EncoderSpecification,
        *,
        revision: str,
        device: str,
        token: str | None = None,
    ) -> None:
        if not revision or revision == "main":
            raise ValueError("An immutable model revision is required")
        self.specification = specification
        self.revision = revision
        self.device = device
        torch, _functional, AutoModel = _load_libraries()
        self._torch = torch
        self.model = AutoModel.from_pretrained(
            specification.model_id,
            revision=revision,
            token=token,
        ).eval().to(device)

    @staticmethod
    def resolve_revision(model_id: str, *, token: str | None = None) -> str:
        try:
            from huggingface_hub import model_info
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required to resolve revisions") from error
        revision = model_info(model_id, token=token).sha
        if not revision or len(revision) != 40:
            raise RuntimeError("Hugging Face did not return an immutable revision")
        return revision

    def _input_tensor(self, crops: Sequence[np.ndarray]) -> Any:
        array = np.stack([np.asarray(crop, dtype=np.float32) for crop in crops])
        if array.ndim != 4 or array.shape[1:] != (224, 224, 3):
            raise ValueError("Every prepared encoder crop must be 224 x 224 RGB")
        array = array / 255.0
        tensor = self._torch.from_numpy(array).permute(0, 3, 1, 2)
        mean = self._torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = self._torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return ((tensor - mean) / std).to(self.device)

    def encode(self, crops: Sequence[np.ndarray]) -> EncodedBatch:
        if not crops:
            raise ValueError("At least one crop is required")
        torch = self._torch
        _torch, functional, _AutoModel = _load_libraries()
        inputs = self._input_tensor(crops)
        use_half = self.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if use_half else "cpu",
            dtype=torch.float16,
            enabled=use_half,
        ):
            output = self.model(pixel_values=inputs)
        if self.specification.backend == "transformers_dino":
            tokens = output.last_hidden_state
            register_count = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_size = getattr(self.model.config, "patch_size", None)
            if isinstance(patch_size, (tuple, list)):
                patch_size = patch_size[0]
            patch_size = int(patch_size)
            grid_size = 224 // patch_size
            patch_start = 1 + register_count
            patch_tokens = tokens[:, patch_start:, :]
            if patch_tokens.shape[1] != grid_size * grid_size:
                raise RuntimeError("Encoder token layout does not match its patch configuration")
            global_features = functional.normalize(tokens[:, 0, :].float(), dim=1)
            patch_tokens = functional.normalize(patch_tokens.float(), dim=2)
            patches = tuple(
                value.detach().cpu().numpy().astype(np.float32)
                for value in patch_tokens
            )
        elif self.specification.backend == "transformers_resnet":
            feature_map = output.last_hidden_state
            if feature_map.ndim != 4:
                raise RuntimeError("ResNet feature map has unexpected dimensions")
            pooled = feature_map.mean(dim=(2, 3))
            global_features = functional.normalize(pooled.float(), dim=1)
            patches = ()
            grid_size = None
        else:
            raise ValueError("Unsupported encoder backend")
        global_array = global_features.detach().cpu().numpy().astype(np.float32)
        return EncodedBatch(
            global_features=global_array,
            patch_features=patches,
            patch_grid_size=grid_size,
            feature_dimension=int(global_array.shape[1]),
        )


class ContractVisualEncoder:
    """Deterministic non-model encoder used only for local contract tests."""

    specification = EncoderSpecification(
        "contract_encoder",
        "local/contract-encoder",
        "Contract",
        "contract",
        "contract",
        0,
    )
    revision = "0" * 40
    device = "cpu"

    def encode(self, crops: Sequence[np.ndarray]) -> EncodedBatch:
        globals_: list[np.ndarray] = []
        patches: list[np.ndarray] = []
        for crop in crops:
            array = np.asarray(crop, dtype=np.float32) / 255.0
            if array.shape != (224, 224, 3):
                raise ValueError("Contract crops must be 224 x 224 RGB")
            grid = array.reshape(16, 14, 16, 14, 3).mean(axis=(1, 3))
            patch = np.concatenate(
                [grid.reshape(-1, 3), np.square(grid).reshape(-1, 3)], axis=1
            )
            patch /= np.maximum(np.linalg.norm(patch, axis=1, keepdims=True), 1e-12)
            global_feature = np.concatenate([array.mean(axis=(0, 1)), array.std(axis=(0, 1))])
            global_feature /= max(float(np.linalg.norm(global_feature)), 1e-12)
            globals_.append(global_feature.astype(np.float32))
            patches.append(patch.astype(np.float32))
        return EncodedBatch(
            global_features=np.stack(globals_),
            patch_features=tuple(patches),
            patch_grid_size=16,
            feature_dimension=6,
        )


def specification_by_key(key: str) -> EncoderSpecification:
    for specification in ENCODER_REGISTRY:
        if specification.key == key:
            return specification
    raise KeyError(f"Unknown encoder key: {key}")


__all__ = [
    "ContractVisualEncoder",
    "ENCODER_REGISTRY",
    "EncodedBatch",
    "EncoderSpecification",
    "FrozenVisualEncoder",
    "encoder_registry_rows",
    "encoder_registry_sha256",
    "specification_by_key",
]
