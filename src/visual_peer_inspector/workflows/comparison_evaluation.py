"""Evaluate locked development scores without changing their predictions."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from visual_peer_inspector.representation.cross_view_identity import CandidateObservation, ViewObservation, match_cross_view_identities
from visual_peer_inspector.evaluation.context_ranker import ContextGroup, assign_asset_folds, context_features, fit_context_ranker, score_context_group
from visual_peer_inspector.workflows.comparison_io import atomic_json, canonical, digest_file, read_json, seal, verify_seal
from visual_peer_inspector.workflows.scoring_comparison import METHODS, PROTOCOL

CONTEXT_METHOD = "context_ranker_oof"
EVALUATION_METHODS = METHODS + (CONTEXT_METHOD,)


def metrics(items, labels):
    """Retain all positive identities in the denominator, including missed ones."""
    positives = {key for key, value in labels.items() if value == 1}
    if any(not np.isfinite(row["score"]) for row in items):
        raise ValueError("Scored candidates must have finite values")
    if len({row["id"] for row in items}) != len(items):
        raise ValueError("Ranking identifiers must be unique")
    if not positives:
        raise ValueError("Anomaly ranking requires a positive object; all-normal detection is separate")
    ranked = sorted(items, key=lambda r: (-r["score"], r["id"]))
    seen, hits, ap = set(), 0, 0.0
    positive_flags = []
    for position, row in enumerate(ranked, 1):
        ref = row.get("reference")
        hit = ref in positives and ref not in seen
        positive_flags.append(hit)
        if hit:
            hits += 1
            ap += hits / position
            seen.add(ref)
    top = bool(positive_flags and positive_flags[0])
    tied = [r for r in ranked if ranked and np.isclose(r["score"], ranked[0]["score"], atol=1e-12, rtol=0)]
    tied_flags = [r.get("reference") in positives for r in tied]
    negatives = [row for row in ranked if row.get("reference") not in positives]
    auc_pairs = []
    for positive in positives:
        positive_scores = [row["score"] for row in ranked if row.get("reference") == positive]
        if positive_scores:
            value = max(positive_scores)
            auc_pairs.extend(1.0 if value > row["score"] else 0.5 if np.isclose(value, row["score"], atol=1e-12, rtol=0) else 0.0 for row in negatives)
        else:
            auc_pairs.extend(0.0 for _row in negatives)
    return {
        "ap": ap / len(positives), "top1": float(top),
        "recall_at_2": sum(positive_flags[:2]) / len(positives),
        "auroc": float(np.mean(auc_pairs)) if auc_pairs else 0.0,
        "reciprocal_rank": next((1/(i+1) for i, flag in enumerate(positive_flags) if flag), 0.0),
        "top_tie_count": len(tied), "top1_tie_expected": float(np.mean(tied_flags)) if tied_flags else 0.0,
        "top1_tie_best": float(any(tied_flags)), "top1_tie_worst": float(bool(tied_flags) and all(tied_flags)),
        "ranked_count": len(ranked), "positive_count": len(positives),
        "winner": ranked[0]["id"] if ranked else None,
        "winner_support": ranked[0].get("support", 1) if ranked else 0,
        "winner_purity": ranked[0].get("purity", 1.0) if ranked else 0.0,
    }


def normalized_scores(objects, method):
    available = [obj for obj in objects if obj["scores"][method] is not None]
    if len(available) < 3:
        return {}
    values = np.asarray([o["scores"][method] for o in available])
    # Ties share their mean rank; object identifiers do not change the numeric score.
    scores = (rankdata(values, method="average") - 1) / (len(values) - 1)
    return {o["candidate_id"]: float(s) for o, s in zip(available, scores, strict=True)}


def identity_groups(rows, route, views):
    observations = []
    for view in views:
        row = rows[route, view]
        candidates = tuple(CandidateObservation(candidate_id=o["candidate_id"], embedding=tuple(o["embedding"])) for o in row["objects"])
        observations.append(ViewObservation(view_id=view, candidates=candidates))
    result = match_cross_view_identities(observations)
    return [(t.identity_id, [(m.view_id, m.candidate_id) for m in t.members]) for t in result.tracks]


def track_references(groups, correspondence, route, references):
    """Assign identities for evaluation only, using one-to-one majority overlap.

    A mixed or tied track does not become a true positive just because one of
    its observations is anomalous. Unmatched observations count against purity.
    """
    names = sorted(references)
    votes = np.zeros((len(groups), len(names)), dtype=float)
    for i, (_, members) in enumerate(groups):
        for view, candidate in members:
            ref = correspondence.get(f"{route}/{view}/{candidate}")
            if ref in names:
                votes[i, names.index(ref)] += 1
    result = {group: (None, 0.0) for group, _ in groups}
    if not groups or not names:
        return result
    left, right = linear_sum_assignment(-votes)
    for i, j in zip(left, right, strict=True):
        purity = votes[i, j] / len(groups[i][1])
        # A strict majority avoids granting credit to conflicting identities.
        result[groups[i][0]] = (names[j] if purity > .5 else None, float(purity))
    return result


def fused_ranking(rows, truth, route, views, method, *, identity, precomputed=None):
    correspondence = truth["correspondence"]
    scores = {v: normalized_scores(rows[route, v]["objects"], method) for v in views}
    if identity == "predicted":
        groups = precomputed if precomputed is not None else identity_groups(rows, route, views)
        refs = track_references(groups, correspondence, route, truth["labels"])
    else:
        mapping = defaultdict(list)
        for view in views:
            for candidate in scores[view]:
                reference = correspondence.get(f"{route}/{view}/{candidate}")
                key = reference if reference is not None else f"unmatched-{view}-{candidate}"
                mapping[key].append((view, candidate))
        groups = sorted(mapping.items())
        refs = {key: (key if key in truth["labels"] else None, 1.0) for key in mapping}
    items = []
    for group, members in groups:
        values = [scores[v][c] for v, c in members if c in scores[v]]
        if values:
            reference, purity = refs[group]
            items.append({"id": group, "score": float(np.mean(values)), "reference": reference, "support": len(values), "purity": purity})
    return items


def paired_intervals(frame, methods, *, samples=2000, seed=32040):
    """Resample asset groups, retaining all their paired scenes on each draw."""
    outputs = []
    group_columns = ["route", "identity", "view_count"]
    for key, group in frame.groupby(group_columns):
        for measure in ("ap", "top1", "auroc"):
            pivot = group.pivot(index=["scene", "asset_group"], columns="method", values=measure)
            for candidate in methods:
                if candidate == "P1_current":
                    continue
                paired = (pivot[candidate] - pivot["P1_current"]).reset_index(name="difference")
                groups = [x["difference"].to_numpy() for _, x in paired.groupby("asset_group")]
                rng = np.random.default_rng(seed)
                draws = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(samples)]
                low, high = np.quantile(draws, [.025, .975])
                outputs.append(dict(zip(group_columns, key)) | {"measure": measure, "candidate": candidate, "baseline": "P1_current", "difference": paired["difference"].mean(), "ci_low": low, "ci_high": high, "asset_groups": len(groups), "scenes": len(paired)})
    return pd.DataFrame(outputs)


def _context_groups(records_by_scene, truth_by_scene):
    groups = []
    feature_names = None
    for scene in sorted(records_by_scene):
        truth = truth_by_scene[scene]
        rows = {(row["route"], row["view"]): row for row in records_by_scene[scene]}
        for route in ("supplied", "automatic"):
            for view in sorted({view for current_route, view in rows if current_route == route}):
                objects = rows[route, view]["objects"]
                if len(objects) < 3 or any(item["scores"].get(method) is None for item in objects for method in METHODS):
                    continue
                features, names = context_features(objects, METHODS)
                if feature_names is None:
                    feature_names = names
                elif feature_names != names:
                    raise ValueError("Context feature schema changed within one run")
                labels = np.asarray(
                    [
                        truth["labels"].get(
                            truth["correspondence"].get(f"{route}/{view}/{item['candidate_id']}"),
                            0,
                        )
                        for item in objects
                    ],
                    dtype=np.int8,
                )
                groups.append(
                    ContextGroup(
                        scene=scene,
                        view=view,
                        route=route,
                        asset_group=truth["asset_group"],
                        candidate_ids=tuple(item["candidate_id"] for item in objects),
                        features=features,
                        labels=labels,
                    )
                )
    if feature_names is None:
        raise ValueError("No scoreable groups are available for the context ranker")
    return groups, feature_names


def _context_scores(records_by_scene, truth_by_scene, output, binding):
    groups, feature_names = _context_groups(records_by_scene, truth_by_scene)
    fold_count = min(int(PROTOCOL["context_ranker"]["folds"]), len({group.asset_group for group in groups}))
    folds = assign_asset_folds({group.asset_group for group in groups}, fold_count)
    binding_record = {
        "prediction_lock_sha256": binding["prediction_lock_sha256"],
        "evaluator_lock_sha256": binding["evaluator_lock_sha256"],
        "context_protocol": PROTOCOL["context_ranker"],
        "feature_names": list(feature_names),
        "folds": folds,
    }
    binding_sha256 = sha256(canonical(binding_record)).hexdigest()
    checkpoint_root = output / "context-folds"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    scores = {}
    fold_rows = []
    for route in ("supplied", "automatic"):
        route_groups = [group for group in groups if group.route == route]
        for fold in range(fold_count):
            path = checkpoint_root / f"{route}-fold-{fold:02d}.json"
            train = [group for group in route_groups if folds[group.asset_group] != fold]
            held_out = [group for group in route_groups if folds[group.asset_group] == fold]
            if not held_out:
                raise ValueError("Every context fold must contain held-out groups")
            if path.is_file():
                record = read_json(path)
                if record.get("binding_sha256") != binding_sha256:
                    raise ValueError("Context checkpoint belongs to a different run")
                model = record["model"]
                resumed = True
            else:
                model = fit_context_ranker(
                    train,
                    regularization=float(PROTOCOL["context_ranker"]["regularization"]),
                )
                atomic_json(
                    path,
                    {
                        "binding_sha256": binding_sha256,
                        "route": route,
                        "fold": fold,
                        "held_out_assets": sorted(asset for asset, assigned in folds.items() if assigned == fold),
                        "model": model,
                    },
                )
                resumed = False
            for group in held_out:
                values = score_context_group(group, model)
                for candidate, value in zip(group.candidate_ids, values, strict=True):
                    scores[(group.scene, group.view, group.route, candidate)] = float(value)
            fold_rows.append(
                {
                    "route": route,
                    "fold": fold,
                    "train_groups": len(train),
                    "held_out_groups": len(held_out),
                    "held_out_assets": len({group.asset_group for group in held_out}),
                    "training_groups_with_positive": model["training_groups_with_positive"],
                    "loss": model["loss"],
                    "iterations": model["iterations"],
                    "checkpoint_resumed": resumed,
                }
            )
    final_models = {
        route: fit_context_ranker(
            [group for group in groups if group.route == route],
            regularization=float(PROTOCOL["context_ranker"]["regularization"]),
        )
        for route in ("supplied", "automatic")
    }
    atomic_json(
        output / "private/context_model_candidate.json",
        {
            "status": "development candidate, not a final or protected result",
            "binding": binding_record,
            "binding_sha256": binding_sha256,
            "feature_names": list(feature_names),
            "models_fit_on_all_development_groups": final_models,
        },
    )
    atomic_json(
        output / "private/context_oof_scores.json",
        {
            "binding_sha256": binding_sha256,
            "scores": [
                {"scene": key[0], "view": key[1], "route": key[2], "candidate_id": key[3], "score": value}
                for key, value in sorted(scores.items())
            ],
        },
    )
    return scores, pd.DataFrame(fold_rows)


def evaluate(model_root, prediction_root, evaluator_root, output, *, allow_contract=False):
    model_root, prediction_root, evaluator_root, output = map(Path, (model_root, prediction_root, evaluator_root, output))
    if (output / "artifact_lock.json").is_file():
        verify_seal(output)
        return pd.read_csv(output / "shareable/ranking_summary.csv")
    lock = verify_seal(prediction_root, name="prediction_lock.json")
    verify_seal(prediction_root, name="feature_lock.json")
    verify_seal(evaluator_root)
    verify_seal(model_root)
    binding = read_json(prediction_root / "run_binding.json")
    evaluator_index = read_json(evaluator_root / "index.json")
    if binding["input_lock_sha256"] != digest_file(model_root / "artifact_lock.json") or evaluator_index["model_lock_sha256"] != binding["input_lock_sha256"]:
        raise ValueError("Prediction, model and evaluator packages do not match")
    if binding["protocol"] != json_normalized_protocol():
        raise ValueError("Prediction protocol does not match the evaluator")
    if binding["producer"] != "dinov2-small" and not allow_contract:
        raise ValueError("Contract-test features cannot support scientific results")
    scenes = lock["metadata"]["scenes"]
    if not set(scenes) <= set(evaluator_index["scenes"]):
        raise ValueError("Evaluator is missing requested scenes")
    safe, private = output / "shareable", output / "private"
    safe.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    evaluation_binding = {
        "prediction_lock_sha256": digest_file(prediction_root / "prediction_lock.json"),
        "evaluator_lock_sha256": digest_file(evaluator_root / "artifact_lock.json"),
        "model_lock_sha256": digest_file(model_root / "artifact_lock.json"),
        "protocol": PROTOCOL,
        "evaluation_source_sha256": digest_file(Path(__file__)),
        "context_source_sha256": digest_file(Path(context_features.__code__.co_filename)),
    }
    binding_path = output / "evaluation_binding.json"
    if binding_path.is_file() and read_json(binding_path) != json.loads(canonical(evaluation_binding)):
        raise ValueError("Existing evaluation output belongs to different inputs or code")
    atomic_json(binding_path, evaluation_binding)
    atomic_json(output / "run_status.json", {"status": "running", "scene_count": len(scenes)})
    try:
        records_by_scene = {scene: read_json(prediction_root / "scenes" / f"{scene}.json") for scene in scenes}
        truth_by_scene = {scene: read_json(evaluator_root / "scenes" / f"{scene}.json") for scene in scenes}
        context, fold_frame = _context_scores(records_by_scene, truth_by_scene, output, evaluation_binding)
        fold_frame.to_csv(safe / "context_fold_summary.csv", index=False)
        scene_metrics, per_view, runtime = [], [], []
        subsets = {"2": PROTOCOL["secondary_views"]["2"], "3": PROTOCOL["primary_views"], "5": PROTOCOL["secondary_views"]["5"], "20": [f"V{i:02d}" for i in range(20)]}
        for scene_number, scene in enumerate(scenes, 1):
            truth = truth_by_scene[scene]
            records = records_by_scene[scene]
            rows = {(r["route"], r["view"]): r for r in records}
            if len(rows) != 40 or len(records) != 40:
                raise ValueError("Expected both routes for all 20 views")
            inputs = {(r["route"], r["view"]): r for r in read_json(model_root / "scenes" / f"{scene}.json")}
            common = {"scene": scene, "asset_group": truth["asset_group"], "category": truth["category"], "families": "+".join(truth["families"])}
            for (route, view), row in rows.items():
                if {o["candidate_id"] for o in row["objects"]} != {o["candidate_id"] for o in inputs[route, view]["objects"]}:
                    raise ValueError("Prediction candidates do not match input candidates")
                for item in row["objects"]:
                    item["scores"][CONTEXT_METHOD] = context.get((scene, view, route, item["candidate_id"]))
                runtime.append({"route": route, "seconds": row["seconds"], "cache_hit": row["feature_cache_hit"], "insufficient": row["status"] != "scored", "count": len(row["objects"]), "fallback_current": row["patch_fallbacks"].get("P1_current", 0), "fallback_forward": row["patch_fallbacks"].get("forward_02_cov70", 0)})
                for method in EVALUATION_METHODS:
                    items = [{"id": o["candidate_id"], "score": o["scores"][method], "reference": truth["correspondence"].get(f"{route}/{view}/{o['candidate_id']}")} for o in row["objects"] if o["scores"].get(method) is not None]
                    per_view.append(common | {"route": route, "view": view, "method": method} | metrics(items, truth["labels"]))
            for route in ("supplied", "automatic"):
                for count, views in subsets.items():
                    for identity in ("reference", "predicted"):
                        if identity == "predicted" and (route != "automatic" or count == "20"):
                            continue
                        groups = identity_groups(rows, route, views) if identity == "predicted" else None
                        for method in EVALUATION_METHODS:
                            items = fused_ranking(rows, truth, route, views, method, identity=identity, precomputed=groups)
                            scene_metrics.append(common | {"route": route, "identity": identity, "view_count": int(count), "method": method} | metrics(items, truth["labels"]))
            atomic_json(output / "progress.json", {"evaluated_scenes": scene_number, "expected_scenes": len(scenes)})
            print(f"Evaluated development scene {scene_number}/{len(scenes)}", flush=True)
        frame, views_frame = pd.DataFrame(scene_metrics), pd.DataFrame(per_view)
        frame.to_csv(private / "scene_metrics.csv", index=False)
        views_frame.to_csv(private / "per_view_metrics.csv", index=False)
        keys = ["route", "identity", "view_count", "method"]
        summary = frame.groupby(keys).agg(scenes=("scene", "size"), asset_groups=("asset_group", "nunique"), ap=("ap", "mean"), top1=("top1", "mean"), recall_at_2=("recall_at_2", "mean"), auroc=("auroc", "mean"), tie_expected=("top1_tie_expected", "mean"), tie_best=("top1_tie_best", "mean"), tie_worst=("top1_tie_worst", "mean"), winner_support=("winner_support", "mean"), winner_purity=("winner_purity", "mean")).reset_index()
        summary.to_csv(safe / "ranking_summary.csv", index=False)
        paired_intervals(frame, EVALUATION_METHODS).to_csv(safe / "paired_asset_intervals.csv", index=False)
        for subgroup in ("category", "families"):
            frame.groupby(keys + [subgroup]).agg(scenes=("scene", "size"), ap=("ap", "mean"), top1=("top1", "mean"), auroc=("auroc", "mean")).reset_index().to_csv(safe / f"{subgroup}_summary.csv", index=False)
        views_frame.groupby(["route", "view", "method"])[["ap", "top1", "auroc"]].mean().reset_index().to_csv(safe / "view_summary.csv", index=False)
        pd.DataFrame(runtime).groupby("route").agg(route_views=("count", "size"), objects=("count", "sum"), insufficient_views=("insufficient", "sum"), cached_views=("cache_hit", "sum"), seconds=("seconds", "sum"), p1_fallbacks=("fallback_current", "sum"), forward_fallbacks=("fallback_forward", "sum")).reset_index().to_csv(safe / "runtime_summary.csv", index=False)
        atomic_json(safe / "evaluation_receipt.json", {"prediction_lock_sha256": evaluation_binding["prediction_lock_sha256"], "evaluator_lock_sha256": evaluation_binding["evaluator_lock_sha256"], "scene_count": len(scenes), "producer": binding["producer"], "scope": "development only with asset-grouped out-of-fold context scores", "methods": list(EVALUATION_METHODS), "primary_comparison": "automatic masks, predicted identity, 3 views", "excluded_failed_scenes": 0, "promotion": "review required; no automatic promotion", "warning": "No result is untouched confirmation. Reference identity and supplied masks are diagnostic routes."})
        from visual_peer_inspector.workflows.comparison_figures import render_evidence
        render_evidence(model_root, prediction_root, evaluator_root, output, frame, summary)
        atomic_json(output / "run_status.json", {"status": "complete", "scene_count": len(scenes)})
        seal(output, [p for p in output.rglob("*") if p.is_file() and p.name != "artifact_lock.json"])
        return summary
    except Exception as error:
        atomic_json(output / "run_status.json", {"status": "failed", "scene_count": len(scenes), "error_type": type(error).__name__})
        raise


def json_normalized_protocol():
    import json
    return json.loads(json.dumps(PROTOCOL))
