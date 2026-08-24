"""Evaluator-only helpers for protected Visual Peer Inspector evidence.

Imports are lazy so a lightweight package builder does not load numerical
evaluation dependencies that it does not use.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DEFAULT_MATCH_IOU": (".proposals", "DEFAULT_MATCH_IOU"),
    "DS0_REFERENCE_SCHEMA_VERSION": (
        ".proposal_reference",
        "DS0_REFERENCE_SCHEMA_VERSION",
    ),
    "EVALUATOR_ONLY_USE": (".sam_mask_package", "EVALUATOR_ONLY_USE"),
    "MASK_EVALUATOR_PACKAGE_KIND": (
        ".sam_mask_package",
        "MASK_EVALUATOR_PACKAGE_KIND",
    ),
    "build_ds0_mask_evaluator_package": (
        ".sam_mask_package",
        "build_ds0_mask_evaluator_package",
    ),
    "build_ds0_reference_package": (
        ".proposal_reference",
        "build_ds0_reference_package",
    ),
    "build_sam_evaluator_crosswalk": (
        ".sam_crosswalk",
        "build_sam_evaluator_crosswalk",
    ),
    "clustered_bootstrap_intervals": (
        ".proposals",
        "clustered_bootstrap_intervals",
    ),
    "evaluate_proposal_dataset": (".proposals", "evaluate_proposal_dataset"),
    "reject_model_facing_mask_evaluator_package": (
        ".sam_mask_package",
        "reject_model_facing_mask_evaluator_package",
    ),
    "summarize_proposal_evaluation": (
        ".proposals",
        "summarize_proposal_evaluation",
    ),
    "verify_ds0_attributes_package": (
        ".proposal_reference",
        "verify_ds0_attributes_package",
    ),
    "verify_ds0_geometry_package": (
        ".proposal_reference",
        "verify_ds0_geometry_package",
    ),
    "verify_ds0_mask_evaluator_package": (
        ".sam_mask_package",
        "verify_ds0_mask_evaluator_package",
    ),
    "verify_sam_evaluator_crosswalk": (
        ".sam_crosswalk",
        "verify_sam_evaluator_crosswalk",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
