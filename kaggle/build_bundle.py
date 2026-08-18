"""Build the single-file Kaggle kernel from src/arcttt + an entry script.

Kaggle script kernels ship one file with no package installs, so the modules
are concatenated in dependency order with intra-package imports stripped
(every name resolves at the flat top level). The output is compile-checked
and grepped for leftover package imports so drift fails loudly at build time.

Usage: python kaggle/build_bundle.py <entry.py> <output.py>
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE_ORDER = (
    "tasks",
    "augment",
    "serialize",
    "vote",
    "lora",
    "decode",
    "model",
    "solve",
)

_INTRA_IMPORT = re.compile(r"^\s*from arcttt(\.\w+)? import ")


def strip_intra_imports(source: str) -> str:
    """Drop `from arcttt... import ...` lines, including parenthesized spans."""

    lines = source.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        if skipping:
            if ")" in line:
                skipping = False
            continue
        if _INTRA_IMPORT.match(line):
            if "(" in line and ")" not in line:
                skipping = True
            continue
        kept.append(line)
    return "\n".join(kept)


def build(entry: Path, output: Path, extra_modules: tuple[str, ...] = ()) -> None:
    parts = [
        "from __future__ import annotations",
        "# arc-ttt bundled single-file pipeline (built by kaggle/build_bundle.py)",
    ]
    for name in MODULE_ORDER + extra_modules:
        source = (ROOT / "src" / "arcttt" / f"{name}.py").read_text()
        source = strip_intra_imports(source)
        source = source.replace("from __future__ import annotations", "")
        parts.append(f"\n\n# === arcttt/{name}.py ===\n{source}")
    entry_source = strip_intra_imports(entry.read_text())
    entry_source = entry_source.replace("from __future__ import annotations", "")
    parts.append(f"\n\n# === entry: {entry.name} ===\n{entry_source}")
    bundle = "\n".join(parts) + "\n"

    ast.parse(bundle)  # syntax gate
    if re.search(r"^\s*from arcttt", bundle, re.MULTILINE):
        raise SystemExit("leftover intra-package import in bundle")
    compiled = compile(bundle, str(output), "exec")  # bytecode gate
    del compiled
    output.write_text(bundle)
    print(f"wrote {output} ({len(bundle.splitlines())} lines)")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]), tuple(sys.argv[3:]))
