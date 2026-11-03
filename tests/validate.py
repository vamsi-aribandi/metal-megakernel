#!/usr/bin/env python3
"""Audit a real persistent run's trace, artifacts, and dispatch source boundary."""
import argparse
import json
import math
from pathlib import Path
import re

from numerics import ROOT, constants


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(directory, minimum_seconds, scripted):
    summary = json.loads((directory / "summary.json").read_text())
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines() if line]
    require(summary["status"] == 4 and not summary["error"], "Metal command did not complete successfully")
    require(summary["seconds"] >= minimum_seconds, "Run shorter than requested validation scope")
    require(summary["dispatches"] == summary["threadgroups"] == 1, "Must use one dispatch and one threadgroup")
    require(summary["threads"] == 96 and summary["simdGroups"] == 3, "Expected three independently progressing SIMD groups")
    require(summary["doneFlags"] == [1, 1, 1], "One or more GPU roles did not exit normally")
    require(all(value > 0 for value in summary["peerProgress"]), "Missing in-kernel observation of peer work")
    require(summary["stopMilliseconds"] < 2000, "GPU stop took over two seconds")
    require(summary["observedVersions"] > 5 and summary["inferenceVersion"] > 128, "Inference did not consume successive trained snapshots")
    require(summary.get("snapshotErrors") == 0, "Snapshot integrity error, or missing verification metric")
    require(summary.get("snapshotVerifiedCopies", summary.get("verifiedSnapshotCopies", 0)) > 5, "Insufficient verified snapshot copies")
    require(summary["coherentFrameCopies"] > 5, "Missing coherent live framebuffer copies")
    require(summary["skippedTokens"] == 0, "Token ring overrun lost generated text")
    require(all(math.isfinite(summary[k]) for k in ("firstLoss", "finalLoss")), "Nonfinite training loss")
    require(summary["finalLoss"] < summary["firstLoss"] * .9, "Training did not substantially reduce loss")
    live = [s for s in samples if s["status"] == 3]
    require(len(live) >= max(5, int(minimum_seconds * 4)), "Insufficient observations before command completion")
    require(all(math.isfinite(s[k]) for s in live for k in ("loss", "x", "y", "angle")), "Nonfinite live telemetry")
    overlap = 0
    for before, after in zip(live, live[1:]):
        require(after["t"] > before["t"], "Nonmonotonic telemetry timestamps")
        for key in ("steps", "tokens", "frames", "version"):
            require(after[key] >= before[key], f"Nonmonotonic {key}")
        overlap += all(after[key] > before[key] for key in ("steps", "tokens", "frames"))
    require(overlap >= max(3, int(len(live) * .25)), "Too few intervals with all three workloads progressing before completion")
    require(all(n >= max(3, int(minimum_seconds * 3)) for n in summary["liveProgressSamples"]), "A GPU workload stalled")
    if scripted:
        require(minimum_seconds >= 6, "Scripted input validation needs at least six seconds")
        matching = {s["requestedKeys"] for s in live if s["requestedKeys"] != 0 and s["keys"] == s["requestedKeys"]}
        require({1, 2, 4, 8}.issubset(matching), f"GPU did not observe each scripted input: {matching}")
        for key in (1, 2):
            phase = [s for s in live if s["keys"] == s["requestedKeys"] == key]
            displacement = max(math.hypot(s["x"] - phase[0]["x"], s["y"] - phase[0]["y"]) for s in phase)
            require(displacement > .1, f"Translation key {key} did not move the GPU player")
        for key in (4, 8):
            phase = [s for s in live if s["keys"] == s["requestedKeys"] == key]
            require(max(s["angle"] for s in phase) - min(s["angle"] for s in phase) > .25, f"Turn key {key} did not rotate the GPU player")
    text = (directory / "generated.txt").read_text()
    require(text == summary["text"] and len(text) == summary["tokens"], "Text artifact/token count mismatch")
    require(len(text) > 20 and len(set(text)) > 5, "Insufficient generated character diversity")
    require(all(ch == "\n" or 32 <= ord(ch) <= 126 for ch in text), "Generated token outside vocabulary")
    tokens = [json.loads(line) for line in (directory / "tokens.jsonl").read_text().splitlines() if line]
    require(len(tokens) == summary["tokens"], "Missing per-token evidence")
    require([item["index"] for item in tokens] == list(range(len(tokens))), "Token ring order or publication failure")
    require(all(0 <= item["token"] < 96 for item in tokens), "Invalid token id")
    decoded = "".join("\n" if item["token"] == 95 else chr(item["token"] + 32) for item in tokens)
    require(decoded == text, "Published token data differs from generated text")
    require(all(a["version"] <= b["version"] for a, b in zip(tokens, tokens[1:])), "Generated tokens regressed to older weights")
    require(all(item["version"] <= summary["steps"] for item in tokens), "Token references a nonexistent trained model")
    token_versions = len({item["version"] for item in tokens})
    require(token_versions > 5 and token_versions == summary["tokenWeightVersions"], "Generated text does not span successive models")
    require(summary["tokenVersionRegressions"] == 0, "Host observed a token model-version regression")
    c = constants()
    frame = (directory / "frame.rgba").read_bytes()
    require(len(frame) == c["WIDTH"] * c["HEIGHT"] * 4, "Framebuffer size mismatch")
    require(all(frame[i] == 255 for i in range(3, len(frame), 4)), "Framebuffer has unpublished/invalid pixels")
    colors = {frame[i:i+4] for i in range(0, len(frame), 4)}
    # A valid close-up of a wall can contain fewer than ten colors. Check
    # recognizable rendered content rather than requiring a distant viewpoint.
    require(len(colors) >= 6, "Framebuffer lacks rendered scene content")
    def pixel(x, y):
        offset = (y * c["WIDTH"] + x) * 4
        return tuple(frame[offset:offset+4])
    require(pixel(c["WIDTH"] // 2, c["HEIGHT"] // 2) == (215, 228, 219, 255), "GPU crosshair missing")
    require(pixel(8, 8) == (105, 122, 135, 255), "GPU minimap boundary missing")
    require(pixel(12, 12) == (12, 18, 24, 255), "GPU minimap room missing")
    require(bytes((255, 116, 77, 255)) in colors, "GPU player marker missing")
    host = (ROOT / "src/main.mm").read_text()
    shader = (ROOT / "src/Persistent.metal").read_text()
    require(len(re.findall(r"\[enc\s+dispatchThreadgroups:", host)) == 1, "Host source does not contain exactly one dispatch site")
    require(len(re.findall(r"\[command\s+commit\]", host)) == 1, "Host source has additional command submissions")
    require(re.findall(r"kernel\s+void\s+(\w+)", shader) == ["persistent"], "Unexpected workload kernels")
    require("threadgroup_barrier(" not in shader, "Full threadgroup barrier can block independent roles")
    result = {"pass": True, "seconds": summary["seconds"], "liveSamples": len(live), "simultaneousProgressIntervals": overlap, "steps": summary["steps"], "tokens": summary["tokens"], "frames": summary["frames"], "firstLoss": summary["firstLoss"], "finalLoss": summary["finalLoss"], "observedVersions": summary["observedVersions"], "tokenWeightVersions": token_versions, "verifiedSnapshotCopies": summary["snapshotVerifiedCopies"], "frameColors": len(colors), "scriptedInputChecked": scripted}
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--min-seconds", type=float, default=10)
    parser.add_argument("--scripted", action="store_true")
    args = parser.parse_args()
    validate(args.directory, args.min_seconds, args.scripted)


if __name__ == "__main__":
    main()
