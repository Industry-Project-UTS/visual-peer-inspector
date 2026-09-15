"""Frozen mapping comparison after the 08O assisted-scoring gate."""
from vpi_object_bank.protocol import PROTOCOL as BANK_PROTOCOL

PROTOCOL = {
    'schema': 'vpi-object-mapping/v1',
    'scope': 'existing 100 development scenes; 27 assets; no protected cohort',
    'view_sets': BANK_PROTOCOL['view_sets'],
    'primary_view_set': 'three-primary',
    'methods': ['existing_all_candidates', 'global_partial_assignment',
                'winner_cluster_consensus', 'correspondence_assisted_reference'],
    'scorer': '08O frozen object_bank_top2; byte-preserved implementation',
    'matching': {
        'features': 'existing per-observation DINOv2-small global embeddings',
        'maximum_cosine_distance': 0.35,
        'objective': 'maximize sum of (0.35 - cosine_distance) over all within-track cross-view pairs',
        'constraints': 'each candidate in exactly one track; at most one candidate per selected view; every linked pair passes threshold',
        'max_hypotheses': 3, 'time_limit_seconds': 5.0, 'max_tracks': 50000,
        'fallback': 'explicit singleton map if primary optimality is not established',
        'size_bias': 'more mutually compatible cross-view pairs receive greater total support',
        'uncertainty': 'retain alternative maps and objective gaps; not calibrated probabilities',
    },
    'primary_prediction': 'best proven map with unchanged scorer; alternatives are diagnostic only',
    'singleton_scoring': 'retained as a hypothesized distinct object as in existing automatic route; self-contamination from fragmentation remains measurable',
    'selection': {
        'global_promotion': 'primary three-view top1 paired lower 95% bound > 0 against winner consensus; AP and Recall2 nonregression; identity link F1 improvement and no extra fragmentation vs existing all-candidate association; positive top1 gain across three-view schedules',
        'otherwise': 'stop matching/scoring search; retain winner consensus automatic baseline and supplied-correspondence development fallback',
        'interpretation': 'development decision; automatic app reliability and independent confirmation remain unproved',
    },
    'bootstrap': BANK_PROTOCOL['bootstrap'],
    'metrics': BANK_PROTOCOL['metrics'],
    'labels': 'matcher never receives reference identities or anomaly labels; seal predictions before evaluation',
}
