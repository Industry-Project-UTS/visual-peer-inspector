"""Aggregate charts and private, traceable object-level evidence."""

from hashlib import sha256
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import prepare_masked_crop
from visual_peer_inspector.workflows.comparison_io import atomic_json, read_json, safe_path
from visual_peer_inspector.workflows.scoring_comparison import CROP, PROTOCOL


def mark_box(axis, box, *, color, label=""):
    x1, y1, x2, y2 = box
    axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=4 if color == "#14853b" else 2))
    if label:
        axis.text(x1, max(0, y1-3), label, color=color, fontsize=7, bbox={"facecolor": "white", "alpha": .8, "pad": 1})


def render_evidence(model_root, prediction_root, evaluator_root, output, frame, summary):
    safe, private = output / "shareable", output / "private"
    figures = private / "figures"
    figures.mkdir(exist_ok=True)
    contract_note = "CONTRACT TEST: NOT DINO RESULTS\n" if read_json(prediction_root / "run_binding.json")["producer"] != "dinov2-small" else ""
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    routes = [("supplied", "reference"), ("automatic", "reference"), ("automatic", "predicted")]
    methods = tuple(summary.method.drop_duplicates())
    for axis, (route, identity) in zip(axes, routes, strict=True):
        table = summary[(summary.route == route) & (summary.identity == identity)]
        for method in methods:
            points = table[table.method == method].sort_values("view_count")
            axis.plot(points.view_count, points.ap, marker="o", label=method)
        axis.set(title=f"{route} masks / {identity} identity", xlabel="Views", ylim=(0, 1))
    axes[0].set_ylabel("Mean average precision")
    axes[-1].legend(fontsize=7)
    fig.suptitle(contract_note + "Development comparison: ranking, masks and identity")
    fig.tight_layout()
    fig.savefig(safe / "ranking_by_route_and_views.png", dpi=170)
    plt.close(fig)
    primary = frame[(frame.route == "automatic") & (frame.identity == "predicted") & (frame.view_count == 3)]
    pivot = primary.pivot(index="scene", columns="method", values="ap")
    primary_summary = summary[(summary.route == "automatic") & (summary.identity == "predicted") & (summary.view_count == 3)]
    candidate_method = primary_summary.sort_values(["ap", "top1", "method"], ascending=[False, False, True]).iloc[0].method
    delta = (pivot[candidate_method] - pivot.P1_current).sort_values()
    selection = {}
    for scene in sorted(pivot.index, key=lambda s: sha256(("gallery-v1/"+s).encode()).hexdigest())[:12]:
        selection[scene] = "fixed hash sample"
    for scene in delta[delta < 0].head(6).index:
        selection.setdefault(scene, "largest development AP decrease")
    for scene in delta[delta > 0].tail(6).index:
        selection.setdefault(scene, "largest development AP increase")
    for scene in sorted(pivot.index):
        if len(selection) >= min(24, len(pivot)):
            break
        selection.setdefault(scene, "deterministic coverage fill")
    manifest = []
    from visual_peer_inspector.workflows.comparison_evaluation import identity_groups
    context_record = read_json(output / "private/context_oof_scores.json")
    context_scores = {
        (row["scene"], row["view"], row["route"], row["candidate_id"]): row["score"]
        for row in context_record["scores"]
    }
    for ordinal, (scene, reason) in enumerate(selection.items(), 1):
        inputs = {(r["route"], r["view"]): r for r in read_json(model_root / "scenes" / f"{scene}.json")}
        predictions = {(r["route"], r["view"]): r for r in read_json(prediction_root / "scenes" / f"{scene}.json")}
        for (route, view), prediction in predictions.items():
            for item in prediction["objects"]:
                item["scores"]["context_ranker_oof"] = context_scores.get((scene, view, route, item["candidate_id"]))
        truth = read_json(evaluator_root / "scenes" / f"{scene}.json")
        views = PROTOCOL["primary_views"]
        tracks = dict(identity_groups(predictions, "automatic", views))
        shown_methods = ("P1_current", "forward_02_cov70", "P2_consensus", "context_ranker_oof")
        method_titles = {
            "P1_current": "P1 current",
            "forward_02_cov70": "Directed patch",
            "P2_consensus": "P2 consensus",
            "context_ranker_oof": "Context OOF",
        }
        fig, axes = plt.subplots(3, 6, figsize=(18, 10), layout="constrained")
        for row_number, view in enumerate(views):
            row = inputs["automatic", view]
            rgb = np.asarray(Image.open(safe_path(model_root, row["rgb"])).convert("RGB"))
            union = np.zeros(rgb.shape[:2], bool)
            for obj in row["objects"]:
                union |= decode_binary_mask(obj["mask"])
            axes[row_number, 0].imshow(rgb)
            for obj in row["objects"]:
                mark_box(axes[row_number, 0], obj["box"], color="#2780ba")
            axes[row_number, 0].set_title(f"{view}: automatic object boxes", fontsize=8.5, pad=4)
            if row_number == 0:
                axes[row_number, 1].imshow(union, cmap="gray", vmin=0, vmax=1)
                axes[row_number, 1].set_title("SAM foreground masks", fontsize=8.5, pad=4)
            elif row_number == 1:
                axes[row_number, 1].imshow(np.where(union[:, :, None], rgb, 0))
                axes[row_number, 1].set_title("SAM objects on black", fontsize=8.5, pad=4)
            else:
                scored = predictions["automatic", view]["objects"]
                available = [o for o in scored if o["scores"]["forward_02_cov70"] is not None]
                if available:
                    winner = sorted(available, key=lambda o: (-o["scores"]["forward_02_cov70"], o["candidate_id"]))[0]
                    obj = next(o for o in row["objects"] if o["candidate_id"] == winner["candidate_id"])
                    crop = prepare_masked_crop(rgb, decode_binary_mask(obj["mask"]), configuration=CROP)
                    axes[row_number, 1].imshow(crop.rgb, extent=(0,16,16,0))
                    heatmap = np.asarray(winner["heatmaps"]["forward_02_cov70"], dtype=float)
                    axes[row_number, 1].imshow(heatmap, cmap="magma", alpha=.6, extent=(0,16,16,0))
                axes[row_number, 1].set_title("Directed patch evidence", fontsize=8.5, pad=4)
            for column, method in enumerate(shown_methods, 2):
                metric = primary[(primary.scene == scene) & (primary.method == method)].iloc[0]
                chosen = {candidate for v, candidate in tracks.get(metric.winner, []) if v == view}
                axes[row_number, column].imshow(rgb)
                for reference_obj in inputs["supplied", view]["objects"]:
                    reference = truth["correspondence"].get(f"supplied/{view}/{reference_obj['candidate_id']}")
                    if truth["labels"].get(reference) == 1:
                        mark_box(axes[row_number, column], reference_obj["box"], color="#14853b")
                for obj in row["objects"]:
                    if obj["candidate_id"] in chosen:
                        mark_box(axes[row_number, column], obj["box"], color="#d32f2f")
                outcome = "correct" if metric.top1 else "incorrect or unmatched"
                axes[row_number, column].set_title(
                    f"{method_titles[method]}: {outcome}\nsupport {int(metric.winner_support)}/3",
                    fontsize=8.5,
                    pad=4,
                )
        for axis in axes.flat:
            axis.axis("off")
        fig.suptitle(
            contract_note
            + f"Development example {ordinal}: {reason}\n"
            "Blue: automatic boxes. Red: fused selected track. Green: reference unusual object. "
            f"Best fixed screen method: {method_titles.get(candidate_method, candidate_method)}.",
            fontsize=11,
        )
        filename = f"example_{ordinal:02d}.png"
        fig.savefig(figures / filename, dpi=160, bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
        manifest.append({"file": filename, "scene": scene, "selection_reason": reason, "views": views, "shown_methods": list(shown_methods), "best_screen_method": candidate_method, "note": "Green boxes come from supplied reference masks, so missed automatic detections remain visible. Ground truth is evaluator-only; it is never passed to the fixed scorers. The context ranker uses only out-of-fold development predictions."})
    atomic_json(private / "gallery_manifest.json", manifest)
