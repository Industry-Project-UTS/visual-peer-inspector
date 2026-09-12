"""Keep notebook input discovery consistent across generated stages."""

from __future__ import annotations

import ast
import re
from hashlib import sha256


def canonical_kernel_id(title: str) -> str:
    """Use the title-derived URL returned by Kaggle when a notebook is created."""
    return "esreekarreddy/" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

ROOT_DISCOVERY = '''def {name}() -> list[Path]:
    """Return dataset and notebook mounts without their container directories."""
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    candidates = [path for path in root.iterdir()
                  if path.is_dir() and path.name not in {{"datasets", "notebooks", "models"}}]
    for pattern in ("datasets/*/*", "notebooks/*/*", "models/*/*/*/*"):
        for path in root.glob(pattern):
            if not path.is_dir():
                continue
            versions = path / "versions"
            if versions.is_dir():
                candidates.extend(child for child in versions.iterdir() if child.is_dir())
            else:
                candidates.append(path)
    return sorted({{path.resolve() for path in candidates}}, key=lambda path: path.as_posix())
'''

RECOVERY_DISCOVERY = '''def recovery_archives(root: Path, filename: str) -> list[Path]:
    """Find a ZIP or byte-preserved Kaggle upload; the caller verifies its hashes."""
    return sorted({path.resolve() for name in (filename, filename + ".bin")
                   for path in root.rglob(name) if path.is_file()}, key=lambda path: path.as_posix())
'''


def harden_source(source: str) -> str:
    """Update transport helpers without changing inference or evaluation cells."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in tree.body:
        # Helpers with explicit input roots retain their existing signature.
        if isinstance(node, ast.FunctionDef) and node.name in {"kaggle_roots", "mounted_roots"} and not node.args.args:
            replacement = ROOT_DISCOVERY.format(name=node.name)
            if "return [Path.cwd().resolve()]" in ast.get_source_segment(source, node):
                replacement = replacement.replace("return []", "return [Path.cwd().resolve()]")
            replacements.append((node.lineno - 1, node.end_lineno, replacement))
    for start, end, replacement in reversed(replacements):
        lines[start:end] = [replacement]
    source = "".join(lines)
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == "recovery_archives":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "rglob" and node.args:
                argument = ast.get_source_segment(source, node.args[0])
                in_recovery_helper = isinstance(function, ast.FunctionDef) and "recovery" in function.name
                if ".zip" not in argument or not (in_recovery_helper or "recovery" in argument):
                    continue
                receiver = ast.get_source_segment(source, node.func.value)
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                replacements.append((start, end, f"recovery_archives({receiver}, {argument})"))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    if replacements and "def recovery_archives(" not in source:
        # Keep imports first, then define the helper before inline discovery.
        import_ends = [node.end_lineno for node in ast.parse(source).body
                       if isinstance(node, (ast.Import, ast.ImportFrom))]
        lines = source.splitlines(keepends=True)
        insertion = max(import_ends, default=0)
        lines[insertion:insertion] = ["\n" + RECOVERY_DISCOVERY + "\n"]
        source = "".join(lines)
    ast.parse(source)
    return source


def harden_notebook(notebook, *, preserve_ids: bool = False):
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            cell.source = harden_source(cell.source)
        if not preserve_ids:
            cell.id = sha256(f"{index}:{cell.cell_type}:{cell.source}".encode()).hexdigest()[:8]
    return notebook
