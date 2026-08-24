"""Dataset integrity, segmented access and archive-index helpers."""

from .archive_index import build_archive_index, read_central_directory_record
from .archive_members import (
    IndexedMember,
    extract_indexed_member,
    load_indexed_member,
    read_indexed_member,
)
from .integrity import VerificationReport, load_manifest, verify_chunk_directory
from .model_safe_package import (
    PackageAudit,
    SMOKE20_COHORT_ID,
    build_smoke20_rgb_package,
    choose_smoke20_rows,
    collect_model_input_evidence,
    verify_model_input_package,
    verify_trusted_package_inputs,
)
from .segmented_file import SegmentedFile
from .scene_catalog import (
    anomaly_variant,
    build_scene_catalog,
    load_split_mapping,
    scene2d_anomaly_flags,
)
from .sam_prompt_package import (
    SAM_PROMPT_SCHEMA_VERSION,
    build_sam_prompt_package,
    verify_sam_prompt_package,
)
from .selected_examples import ToysObjectView, ToysSceneView, load_toys_scene_view
from .toysad_selection import (
    RELEASE_ID,
    SourcePaths,
    anomaly_prefixes,
    assemble_eligible_scene_pool,
    audit_existing_release,
    audit_routine_release,
    audit_selected_design,
    deterministic_object_permutation,
    solve_asset_guarded_roles,
    solve_development_pilot,
    validate_model_facing_media_record,
    write_selection_release,
    verify_repeat_build_evidence,
)

__all__ = [
    "SegmentedFile",
    "IndexedMember",
    "VerificationReport",
    "PackageAudit",
    "build_archive_index",
    "build_scene_catalog",
    "extract_indexed_member",
    "load_indexed_member",
    "load_manifest",
    "read_indexed_member",
    "read_central_directory_record",
    "verify_chunk_directory",
    "SMOKE20_COHORT_ID",
    "SAM_PROMPT_SCHEMA_VERSION",
    "build_sam_prompt_package",
    "build_smoke20_rgb_package",
    "choose_smoke20_rows",
    "collect_model_input_evidence",
    "verify_model_input_package",
    "verify_trusted_package_inputs",
    "verify_sam_prompt_package",
    "anomaly_variant",
    "load_split_mapping",
    "scene2d_anomaly_flags",
    "ToysObjectView",
    "ToysSceneView",
    "load_toys_scene_view",
    "RELEASE_ID",
    "SourcePaths",
    "anomaly_prefixes",
    "assemble_eligible_scene_pool",
    "audit_existing_release",
    "audit_routine_release",
    "audit_selected_design",
    "deterministic_object_permutation",
    "solve_asset_guarded_roles",
    "solve_development_pilot",
    "validate_model_facing_media_record",
    "write_selection_release",
    "verify_repeat_build_evidence",
]
