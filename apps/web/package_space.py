"""Build an allowlisted local Spaces folder. Never uploads or downloads anything."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def package(output: Path):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Choose an empty output directory; existing files are never replaced."
        )
    required = (
        "src/vpi_object_bank/scoring.py",
        "src/vpi_object_bank/prediction.py",
        "src/visual_peer_inspector/web/decision.py",
        "src/visual_peer_inspector/web/session_cache.py",
        "src/visual_peer_inspector/web/frozen_policy.py",
        "src/visual_peer_inspector/web/frozen_review_policy.json",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise ValueError(f"Selected multi-view method dependencies are missing: {missing}")
    sources = [ROOT / "pyproject.toml"]
    for directory in (
        ROOT / "src/visual_peer_inspector",
        ROOT / "src/vpi_object_bank",
        ROOT / "apps/web",
    ):
        sources.extend(
            p
            for p in directory.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix in {".py", ".json", ".css", ".svg", ".txt", ".md"}
        )
    if any(p.is_symlink() for p in sources):
        raise ValueError("Deployment input may not contain symbolic links.")
    output.mkdir(parents=True, exist_ok=True)
    for source in sorted(sources):
        target = output / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    from visual_peer_inspector.web.frozen_policy import POLICY_SHA256
    sealed = output / "src/visual_peer_inspector/web/frozen_review_policy.json"
    if hashlib.sha256(sealed.read_bytes()).hexdigest() != POLICY_SHA256:
        raise ValueError("Packaged review policy differs from sealed calibration policy.")
    for name in ("Dockerfile", "README.md", ".dockerignore"):
        shutil.copyfile(ROOT / "apps/web/space" / name, output / name)
    manifest = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        manifest.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (output / "package-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "directory": str(output),
        "files": len(manifest),
        "bytes": sum(r["bytes"] for r in manifest),
        "uploaded": False,
        "model_weights_included": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2))
