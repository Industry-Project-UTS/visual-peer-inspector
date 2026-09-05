"""Trusted geometry and protected attributes for the DS2 confirmation cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.evaluation.proposal_reference_ds1 import (
    build_ds1_reference_packages,
    verify_ds1_attributes_package,
    verify_ds1_geometry_package,
)


GEOMETRY_KIND = "ds2-oracle-geometry"
ATTRIBUTES_KIND = "ds2-evaluator-attributes"


def build_ds2_reference_packages(**paths: Path | int) -> dict[str, Any]:
    """Build the model-safe geometry and separate evaluator-only attributes."""

    return build_ds1_reference_packages(
        **paths,
        contract=DS2_CONFIRM_CONTRACT,
        geometry_kind=GEOMETRY_KIND,
        attributes_kind=ATTRIBUTES_KIND,
    )


def verify_ds2_geometry_package(root: Path) -> dict[str, Any]:
    """Verify every supplied mask and DS2 geometry package checksum."""

    return verify_ds1_geometry_package(
        root,
        contract=DS2_CONFIRM_CONTRACT,
        package_kind=GEOMETRY_KIND,
    )


def verify_ds2_attributes_package(root: Path) -> dict[str, Any]:
    """Verify the protected DS2 scene attributes without exposing them."""

    return verify_ds1_attributes_package(
        root,
        contract=DS2_CONFIRM_CONTRACT,
        package_kind=ATTRIBUTES_KIND,
    )


__all__ = [
    "build_ds2_reference_packages",
    "verify_ds2_attributes_package",
    "verify_ds2_geometry_package",
]
