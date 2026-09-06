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
    pilot_stage: str | None
    membership_sha256: str
    scene_count: int
    views_per_scene: int
    image_count: int
    asset_group_count: int
    model_membership_sha256: str | None = None

    def __post_init__(self) -> None:
        string_fields = ("cohort_id", "release_id", "role")
        for field_name in string_fields:
            if not getattr(self, field_name).strip():
                raise ValueError(f"Cohort contract field is empty: {field_name}")
        if self.role == "development":
            if self.pilot_stage is None or not self.pilot_stage.strip():
                raise ValueError("Development cohort requires a pilot stage")
        elif self.pilot_stage is not None:
            raise ValueError("Non-development cohort cannot have a pilot stage")
        if not _is_sha256(self.membership_sha256):
            raise ValueError("Cohort membership must be a lowercase SHA-256 digest")
        if self.model_membership_sha256 is not None and not _is_sha256(
            self.model_membership_sha256
        ):
            raise ValueError("Model membership must be a lowercase SHA-256 digest")
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


DEVELOPMENT_REMAINDER_CONTRACT = CohortContract(
    cohort_id="toysad8k-development-remainder-600-v1",
    release_id="toysad8k-train-2000-v2",
    role="development",
    pilot_stage="remainder_600",
    membership_sha256=(
        "e6b4e91dd99de72ddad6e462948c5a4"
        "c87b8594be230b3ed7950af841d0a2198"
    ),
    scene_count=600,
    views_per_scene=20,
    image_count=12_000,
    asset_group_count=164,
)


DEVELOPMENT_FULL_CONTRACT = CohortContract(
    cohort_id="toysad8k-development-full-1000-v1",
    release_id="toysad8k-train-2000-v2",
    role="development",
    pilot_stage="all_development",
    membership_sha256=(
        "00e400350a056631778bd9c4dace1667"
        "d95e52bf58c0a29ea7c48d190c41de0c"
    ),
    scene_count=1_000,
    views_per_scene=20,
    image_count=20_000,
    asset_group_count=273,
)


CALIBRATION_CONTRACT = CohortContract(
    cohort_id="toysad8k-calibration-600-v1",
    release_id="toysad8k-train-2000-v2",
    role="calibration",
    pilot_stage=None,
    membership_sha256=(
        "6700bb065210bfd45188babd2b3d4597"
        "3d6b4f57e5066d7c3dec4b9471baa137"
    ),
    scene_count=600,
    views_per_scene=20,
    image_count=12_000,
    asset_group_count=163,
)


INTERNAL_LOCK_CONTRACT = CohortContract(
    cohort_id="toysad8k-internal-lock-400-v1",
    release_id="toysad8k-train-2000-v2",
    role="internal_lock",
    pilot_stage=None,
    membership_sha256=(
        "819f21f49705f9398c809002905abe52"
        "b8305ad41e4cab349cbbf3a4bd54afe6"
    ),
    scene_count=400,
    views_per_scene=20,
    image_count=8_000,
    asset_group_count=109,
)


OFFICIAL_SEEN_TEST_CONTRACT = CohortContract(
    cohort_id="toysad8k-official-seen-test-715-v1",
    release_id="toysad8k-official-evaluation-v1",
    role="seen-test",
    pilot_stage=None,
    membership_sha256=(
        "2ab155c3b520d75c228d8a650323d13c"
        "7bc4a42589edbca19c878a3a44247a25"
    ),
    scene_count=715,
    views_per_scene=20,
    image_count=14_300,
    asset_group_count=113,
    model_membership_sha256=(
        "e8ce6470f9439f0548fadde36b720b93"
        "a1bfb5cbbec4fb565ed70c5edf83be01"
    ),
)


OFFICIAL_UNSEEN_TEST_CONTRACT = CohortContract(
    cohort_id="toysad8k-official-unseen-test-2437-v1",
    release_id="toysad8k-official-evaluation-v1",
    role="unseen-test",
    pilot_stage=None,
    membership_sha256=(
        "356ed4d864701dad16a805d89ef69b1"
        "d96dcb191f9f678c97c964746057ec776"
    ),
    scene_count=2_437,
    views_per_scene=20,
    image_count=48_740,
    asset_group_count=316,
    model_membership_sha256=(
        "1c3abcb450588f6a99c457b6b3ccb80"
        "afa008fd5dfbebbc97d725c67a52cfbcb"
    ),
)


__all__ = [
    "CALIBRATION_CONTRACT",
    "DEVELOPMENT_FULL_CONTRACT",
    "DEVELOPMENT_REMAINDER_CONTRACT",
    "DS1_SCREEN_CONTRACT",
    "DS2_CONFIRM_CONTRACT",
    "INTERNAL_LOCK_CONTRACT",
    "OFFICIAL_SEEN_TEST_CONTRACT",
    "OFFICIAL_UNSEEN_TEST_CONTRACT",
    "CohortContract",
]
