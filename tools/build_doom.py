#!/usr/bin/env python3
"""Build the vendored Doom engine only when its source inputs change."""
import hashlib
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1] / "third_party/metal-doom"
inputs = sorted(p for folder in ("engine", "tools", "vendor")
                for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
h = hashlib.sha256()
for p in inputs:
    h.update(str(p.relative_to(root)).encode())
    h.update(p.read_bytes())
stamp = root / "build/megakernel-inputs.sha256"
outputs = [root / ("build/" + name) for name in ("doom-engine.metal", "initial-memory.bin")]
if not all(p.exists() for p in outputs) or not stamp.exists() or stamp.read_text() != h.hexdigest():
    subprocess.run([str(root / "engine/build_wasm.sh")], cwd=root, check=True)
    subprocess.run(["python3", str(root / "tools/to_metal.py")], cwd=root, check=True)
    stamp.write_text(h.hexdigest())
