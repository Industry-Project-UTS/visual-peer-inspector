"""Model adapters used by the Visual Peer Inspector experiments."""

from .locateanything import (
    DEFAULT_LOCATEANYTHING_REVISION,
    LocateAnythingConfiguration,
    LocateAnythingWorker,
    ProposalResultStore,
    build_proposal_jobs,
    parse_locateanything_boxes,
    summarize_proposal_rows,
)
from .sam2 import (
    ContractSamWorker,
    Sam2Configuration,
    Sam2ImageWorker,
    SamResultStore,
    build_sam_jobs,
    collect_sam_evidence,
    compare_repeat_results,
    run_sam_jobs,
)

__all__ = [
    "DEFAULT_LOCATEANYTHING_REVISION",
    "LocateAnythingConfiguration",
    "LocateAnythingWorker",
    "ProposalResultStore",
    "build_proposal_jobs",
    "build_sam_jobs",
    "collect_sam_evidence",
    "compare_repeat_results",
    "ContractSamWorker",
    "parse_locateanything_boxes",
    "summarize_proposal_rows",
    "Sam2Configuration",
    "Sam2ImageWorker",
    "SamResultStore",
    "run_sam_jobs",
]
