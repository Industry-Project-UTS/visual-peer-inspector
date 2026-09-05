"""Frozen cohort contracts shared by preparation and model notebooks."""

from __future__ import annotations

from dataclasses import dataclass


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class CohortContract:
    """Expected identity and cardinality of one model-safe cohort."""

    cohort_id: str
    release_id: str
    role: str
    pilot_stage: str
    membership_sha256: str
    scene_count: int
    views_per_scene: int
    image_count: int
    asset_group_count: int

    def __post_init__(self) -> None:
        string_fields = (
            "cohort_id",
            "release_id",
            "role",
            "pilot_stage",
        )
        for field_name in string_fields:
            if not getattr(self, field_name).strip():
                raise ValueError(f"Cohort contract field is empty: {field_name}")
        if not _is_sha256(self.membership_sha256):
            raise ValueError("Cohort membership must be a lowercase SHA-256 digest")
        if self.scene_count <= 0 or self.views_per_scene <= 0:
            raise ValueError("Cohort scene and view counts must be positive")
        if self.image_count != self.scene_count * self.views_per_scene:
            raise ValueError("Cohort image count must equal scenes times views")
        if not 0 < self.asset_group_count <= self.scene_count:
            raise ValueError("Cohort asset-group count is outside the valid range")


DS1_SCREEN_CONTRACT = CohortContract(
    cohort_id="toysad8k-development-screen-100-v1",
    release_id="toysad8k-train-2000-v2",
    role="development",
    pilot_stage="screen_100",
    membership_sha256=(
        "308e2e104448612dc60b5392a5258d97"
        "f6d93f52ca553aa68fa1b447500f4f2f"
    ),
    scene_count=100,
    views_per_scene=20,
    image_count=2_000,
    asset_group_count=27,
)


DS2_CONFIRM_CONTRACT = CohortContract(
    cohort_id="toysad8k-development-confirm-300-v1",
    release_id="toysad8k-train-2000-v2",
    role="development",
    pilot_stage="confirm_300",
    membership_sha256=(
        "bc00e95781a6d52c25c9e40364f8dfa"
        "36ad7874cccd759488ac87ce55dec2eac"
    ),
    scene_count=300,
    views_per_scene=20,
    image_count=6_000,
    asset_group_count=82,
)


__all__ = ["CohortContract", "DS1_SCREEN_CONTRACT", "DS2_CONFIRM_CONTRACT"]
