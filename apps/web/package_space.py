"""Build an allowlisted local Spaces folder. Never uploads or downloads anything."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def package(output: Path):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Choose an empty output directory; existing files are never replaced."
        )
    sources = [ROOT / "pyproject.toml"]
    for directory in (ROOT / "src", ROOT / "apps/web"):
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
