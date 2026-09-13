"""Label-free mask and score previews for the prediction notebook."""

from hashlib import sha256
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.workflows.comparison_io import atomic_json, read_json, safe_path


def prediction_previews(model_root, output, scenes):
    from visual_peer_inspector.workflows.comparison_figures import mark_box
    private, shareable = output / "private-previews", output / "shareable"
    private.mkdir(exist_ok=True)
    shareable.mkdir(exist_ok=True)
    contract_note = "CONTRACT TEST: NOT DINO RESULTS\n" if read_json(output / "run_binding.json")["producer"] != "dinov2-small" else ""
    runtime = []
    for scene in scenes:
        for row in read_json(output / "scenes" / f"{scene}.json"):
            runtime.append({"route": row["route"], "seconds": row["seconds"], "objects": len(row["objects"]), "cache_hit": row["feature_cache_hit"], "insufficient": row["status"] != "scored"})
    frame = pd.DataFrame(runtime)
    frame.groupby("route").agg(route_views=("objects", "size"), objects=("objects", "sum"), seconds=("seconds", "sum"), cached_views=("cache_hit", "sum"), insufficient_views=("insufficient", "sum")).reset_index().to_csv(shareable / "prediction_runtime.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    for route, table in frame.groupby("route"):
        ax.hist(table.seconds, bins=20, alpha=.55, label=route)
    ax.set(xlabel="Seconds per route/view (feature extraction plus scoring)", ylabel="Views", title=contract_note + "Development prediction runtime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(shareable / "prediction_runtime.png", dpi=160)
    plt.close(fig)
    gallery = []
    for ordinal, scene in enumerate(sorted(scenes, key=lambda s: sha256(("prediction-preview/"+s).encode()).hexdigest())[:12], 1):
        inputs = {(r["route"], r["view"]):r for r in read_json(model_root / "scenes" / f"{scene}.json")}
        predictions = {(r["route"], r["view"]):r for r in read_json(output / "scenes" / f"{scene}.json")}
        fig, axes = plt.subplots(2, 4, figsize=(12,6))
        for row_number, route in enumerate(("supplied", "automatic")):
            row, prediction = inputs[route, "V00"], predictions[route, "V00"]
            rgb = np.asarray(Image.open(safe_path(model_root, row["rgb"])).convert("RGB"))
            union = np.zeros(rgb.shape[:2], bool)
            for obj in row["objects"]:
                union |= decode_binary_mask(obj["mask"])
            axes[row_number, 0].imshow(rgb)
            for obj in row["objects"]:
                mark_box(axes[row_number, 0], obj["box"], color="#2780ba")
            axes[row_number, 0].set_title(route + ": all objects")
            axes[row_number, 1].imshow(union, cmap="gray", vmin=0, vmax=1)
            axes[row_number, 1].set_title("Foreground masks")
            axes[row_number, 2].imshow(np.where(union[:, :, None], rgb, 0))
            axes[row_number, 2].set_title("Objects on black")
            axes[row_number, 3].imshow(rgb)
            for method, color in [
                ("P1_current", "#b52b2b"),
                ("forward_02_cov70", "#7042b8"),
                ("P2_consensus", "#d47a00"),
            ]:
                scored = [o for o in prediction["objects"] if o["scores"][method] is not None]
                if scored:
                    winner = sorted(scored, key=lambda o:(-o["scores"][method],o["candidate_id"]))[0]
                    obj = next(o for o in row["objects"] if o["candidate_id"] == winner["candidate_id"])
                    mark_box(axes[row_number, 3], obj["box"], color=color, label=method)
            axes[row_number, 3].set_title("Per-view picks\nCorrectness not evaluated", fontsize=10)
        for axis in axes.flat:
            axis.axis("off")
        fig.suptitle(contract_note + f"Label-free development preview {ordinal}: V00")
        fig.tight_layout()
        filename = f"preview_{ordinal:02d}.png"
        fig.savefig(private / filename, dpi=160)
        plt.close(fig)
        gallery.append({"file": filename, "scene": scene, "view": "V00", "selection": "fixed hash order", "accuracy_evaluated": False})
    atomic_json(private / "gallery_manifest.json", gallery)
    return [p for directory in (private, shareable) for p in directory.iterdir() if p.is_file()]
