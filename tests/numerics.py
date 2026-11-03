#!/usr/bin/env python3
"""Independent CPU reference used only in tests; production math stays on GPU."""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def constants():
    values = {}
    for name, expr in re.findall(r"^#define\s+(\w+)\s+([^\n]+)$", (ROOT / "src/Config.h").read_text(), re.M):
        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value
            if isinstance(node, ast.Name):
                return values[node.id]
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
                a, b = visit(node.left), visit(node.right)
                return a + b if isinstance(node.op, ast.Add) else a * b
            raise ValueError(f"Unsupported dimension expression: {expr}")
        values[name] = visit(ast.parse(expr, mode="eval"))
    return values


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def words(path, code):
    data = path.read_bytes()
    assert len(data) % 4 == 0
    return list(struct.unpack(f"<{len(data) // 4}{code}", data))


def reference(initial, corpus, steps, seed, c):
    """Scalar float64 forward/backward; round stored parameters to float32."""
    w = initial[:]
    v, e, ctx, hidden, inputs = (c[k] for k in ("VOCAB", "EMBED", "CONTEXT", "HIDDEN", "INPUTS"))
    wh, bh, wo, bo = (c[k] for k in ("W_HIDDEN", "W_HBIAS", "W_OUTPUT", "W_OBIAS"))
    rng = seed or 1
    first = ema = None
    for step in range(steps):
        rng ^= (rng << 13) & 0xffffffff
        rng ^= rng >> 17
        rng ^= (rng << 5) & 0xffffffff
        offset = rng % (len(corpus) - ctx)
        target = corpus[offset + ctx]
        x = [w[corpus[offset + i // e] * e + i % e] for i in range(inputs)]
        h = [math.tanh(w[bh + j] + sum(w[wh + j * inputs + i] * x[i] for i in range(inputs))) for j in range(hidden)]
        logits = [w[bo + k] + sum(w[wo + k * hidden + j] * h[j] for j in range(hidden)) for k in range(v)]
        maximum = max(logits)
        p = [math.exp(z - maximum) for z in logits]
        total = sum(p)
        p = [value / total for value in p]
        loss = -math.log(p[target])
        if first is None:
            first = ema = loss
        else:
            ema = .98 * ema + .02 * loss
        dz = [prob - (k == target) for k, prob in enumerate(p)]
        # Differentiate before mutating any output weights.
        dh = [(1 - h[j] * h[j]) * sum(w[wo + k * hidden + j] * dz[k] for k in range(v)) for j in range(hidden)]
        for k in range(v):
            for j in range(hidden):
                idx = wo + k * hidden + j
                w[idx] = f32(w[idx] - .025 * dz[k] * h[j])
            w[bo + k] = f32(w[bo + k] - .025 * dz[k])
        for j in range(hidden):
            for i in range(inputs):
                idx = wh + j * inputs + i
                w[idx] = f32(w[idx] - .025 * dh[j] * x[i])
            w[bh + j] = f32(w[bh + j] - .025 * dh[j])
    return w, first, ema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/metal-megakernel")
    parser.add_argument("--output", type=Path, default=ROOT / "results/numerics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    c = constants()
    reports = []
    for steps, seed in ((1, 42), (4, 0), (32, 12345)):
        prefix = args.output / f"steps{steps}-seed{seed}"
        run = args.output / f"run-{steps}-{seed}"
        command = [str(args.binary), "--headless", "--steps", str(steps), "--seed", str(seed), "--seconds", "5", "--dump-prefix", str(prefix), "--output", str(run)]
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
        (prefix.with_suffix(".log")).write_text(result.stdout)
        assert result.returncode == 0, result.stdout
        actual = words(Path(str(prefix) + "-final.bin"), "f")
        initial = words(Path(str(prefix) + "-initial.bin"), "f")
        corpus = words(Path(str(prefix) + "-corpus.bin"), "I")
        summary = json.loads((run / "summary.json").read_text())
        assert len(initial) == len(actual) == c["PARAMS"]
        assert summary["status"] == 4 and not summary["error"]
        assert summary["steps"] == steps, summary
        expected, first, final = reference(initial, corpus, steps, seed, c)
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        assert all(math.isfinite(x) for x in actual)
        assert initial[:c["W_HIDDEN"]] == actual[:c["W_HIDDEN"]], "Frozen embeddings changed"
        assert max(errors) < 2e-5, f"Parameter error {max(errors)} at {errors.index(max(errors))}"
        assert abs(summary["firstLoss"] - first) < 2e-5
        assert abs(summary["finalLoss"] - final) < 2e-5
        assert any(a != b for a, b in zip(initial, actual)), "No parameter update"
        report = {"steps": steps, "seed": seed, "maxParameterError": max(errors), "firstLossError": abs(summary["firstLoss"] - first), "finalLossError": abs(summary["finalLoss"] - final), "pass": True}
        reports.append(report)
        print(json.dumps(report))
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("PASS: GPU forward, loss, gradients, SGD updates and frozen embeddings match independent scalar CPU reference.")


if __name__ == "__main__":
    main()
