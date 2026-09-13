"""Small, strict file contracts for the scoring study."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f"Invalid package path: {relative}")
    return path


def token(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid opaque token")
    return value


def seal(root, paths, *, name="artifact_lock.json", metadata=None):
    root = Path(root)
    files = {str(Path(p).relative_to(root)): digest_file(p) for p in sorted(paths)}
    value = {"schema": 1, "files": files, "metadata": metadata or {}}
    atomic_json(root / name, value)
    return value


def verify_seal(root, *, name="artifact_lock.json"):
    value = read_json(Path(root) / name)
    if not value["files"]:
        raise ValueError("Empty artifact inventory")
    for relative, expected in value["files"].items():
        if digest_file(safe_path(root, relative)) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    return value


def implementation_digest(source_root):
    root = Path(source_root)
    paths = sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "scripts").glob("*scoring_comparison*.py"))
    paths += sorted((root / "requirements").glob("*comparison*.txt"))
    return sha256(canonical({str(p.relative_to(root)): digest_file(p) for p in paths})).hexdigest()
