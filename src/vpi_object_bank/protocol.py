"""Predeclared comparison; view membership is independent of outcomes."""
from visual_peer_inspector.workflows.learned_peer_features import VIEW_SETS

SCHEMA = 'vpi-object-bank/v1'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
EXTRACTION = {
    'encoder': 'facebook/dinov2-small', 'revision': REVISION,
    'input_size': 224, 'padding_fraction': 0.15,
    'background_rgb': [238, 241, 245], 'minimum_foreground_pixels': 4,
    'coverage': 0.70, 'fallback': 'all', 'feature_dtype': 'float32',
    'grid_size': 16, 'feature_dimension': 384, 'batch_size': 4,
}
PROTOCOL = {
    'schema': SCHEMA,
    'scope': 'existing 100-scene development screen; no new roles or protected data',
    'view_sets': {key: list(value) for key, value in VIEW_SETS.items()},
    'extraction': EXTRACTION,
    'primary_view_set': 'three-primary',
    'routes': ['supplied_reference', 'automatic_reference', 'automatic_predicted'],
    'scorers': ['cached_forward', 'within_view', 'object_bank', 'view_mean_control'],
    'fusion': ['mean', 'top2'],
    'tail': {'fraction': 0.02, 'rounding': 'floor', 'peer_reduction': 'lower_median'},
    'normalization': 'average within-view percentile ranks; >=3 scored candidates',
    'unknown_identity': 'retain observation; bank score unavailable; exclude from peer banks',
    'automatic_association': 'unchanged Hungarian baseline; no new matcher until scoring gate',
    'bank_size_control': 'mean nearest-patch distance across peer views before peer lower median',
    'selection': {
        'primary': 'supplied_reference three-primary; paired asset bootstrap against regenerated within_view mean',
        'candidate_fusions': ['mean', 'top2'],
        'continue_mapping_if': 'positive top1 paired 95% lower bound, no AP/Recall2 regression, positive top1 mean across three-view schedules',
        'tie_break': 'mean fusion, then simpler within_view method',
        'otherwise': 'stop new scoring/matching; retain stronger existing method and report limitations',
        'interpretation': 'development selection only; no independent confirmation claim',
    },
    'metrics': ['top1_tie_expected', 'recall_at_2_tie_expected', 'ap_tie_expected', 'reciprocal_rank_tie_expected'],
    'bootstrap': {'unit': 'asset_group', 'samples': 2000, 'seed': 32040},
    'report_breakdowns': ['view_schedule', 'anomaly_count', 'strict_normal_majority', 'candidate_coverage', 'association', 'ranking_loss'],
    'clearml': 'verified aggregate artifacts after scientific seal; bounded separate process',
}
