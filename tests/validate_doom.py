#!/usr/bin/env python3
"""Replay a combined persistent run in original native Doom C and audit overlap.

This is CPU-only validation of already-captured GPU evidence. The production app
never imports this reference or computes Doom/ML on the CPU. Run one invocation
per recording: the original C engine has process-global state.

Every GPU publication must be recorded before its acknowledgement, including the
initial frame. For a catch-up batch, apply its key transitions once, tick by the
recorded delta, and render ONCE. Rendering each intermediate tick is incorrect.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DOOM = ROOT / "third_party/metal-doom"
WIDTH, HEIGHT = 320, 200
FIELDS = "gametic gamestate health armor x y z angle kills items secrets rndindex prndindex leveltime weapon exited".split()
KEY_CODES = (0xad, 0xaf, 0xac, 0xae, 0xa0, 0xa1, 0xa3, 0xa2, 13, 0xb6,
             49, 50, 51, 52, 53, 54, 55, 9, 27)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    require(bool(rows), f"Empty evidence: {path}")
    return rows


def native_reference(wad_path, library_path, frames, output):
    """Compare every framebuffer byte via SHA256 plus every diagnostic integer."""
    engine = ctypes.CDLL(str(library_path.resolve()))
    engine.engine_alloc.argtypes = [ctypes.c_uint32]
    engine.engine_alloc.restype = ctypes.c_void_p
    engine.engine_init.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
    engine.engine_init.restype = ctypes.c_int
    engine.engine_key.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    engine.engine_key.restype = None
    engine.engine_tick.argtypes = []
    engine.engine_tick.restype = None
    engine.engine_render.argtypes = []
    engine.engine_render.restype = None
    engine.engine_diagnostics.argtypes = []
    engine.engine_diagnostics.restype = ctypes.POINTER(ctypes.c_int32)
    engine.engine_framebuffer.argtypes = []
    engine.engine_framebuffer.restype = ctypes.c_void_p
    engine.engine_log.argtypes = []
    engine.engine_log.restype = ctypes.c_char_p
    wad = wad_path.read_bytes()
    pointer = engine.engine_alloc(len(wad))
    require(bool(pointer), "Original C reference could not allocate IWAD")
    ctypes.memmove(pointer, wad, len(wad))
    result = engine.engine_init(pointer, len(wad))
    require(result == 0, f"Original C initialization failed: {result}, {engine.engine_log()!r}")
    previous_tick = previous_keys = maximum_gap = observed_keys = 0
    cpu_rows = []
    start = time.monotonic()
    for index, frame in enumerate(frames):
        tick, keys = frame["tick"], frame["keys"]
        require(type(tick) is int and type(keys) is int, f"Noninteger Doom input at publication {index+1}")
        require(0 <= keys < 1 << len(KEY_CODES), f"Unknown key bits: {keys}")
        require(frame["sequence"] == index + 1, "Missing/duplicated Doom publication; exact replay is impossible")
        require(frame["renders"] == index + 1, "Unrecorded or extra Doom render; exact replay is impossible")
        if index == 0:
            require(tick == keys == 0, "First publication must be initialized Doom at tick0,keys0")
        else:
            gap = tick - previous_tick
            require(1 <= gap <= 8, f"Invalid tick batch: {previous_tick} -> {tick}")
            maximum_gap = max(maximum_gap, gap)
            for bit, code in enumerate(KEY_CODES):
                if (keys ^ previous_keys) & (1 << bit):
                    engine.engine_key((keys >> bit) & 1, code)
            for _ in range(gap):
                engine.engine_tick()
            # Original render changes some engine state: the exact schedule matters.
            engine.engine_render()
        expected = list(engine.engine_diagnostics()[:16])
        actual = frame["diagnostics"]
        require(len(actual) == 16 and all(type(v) is int and -(1 << 31) <= v < 1 << 31 for v in actual), "Doom diagnostics must be sixteen signed32 integers")
        require(expected[0] == tick + 1 and expected[15] == 0, f"Reference tick/exit invariant failed at tick {tick}")
        differences = {name: {"cpu": a, "gpu": b} for name, a, b in zip(FIELDS, expected, actual) if a != b}
        require(not differences, f"First Doom state divergence at tick {tick}: {differences}")
        pixels = ctypes.string_at(engine.engine_framebuffer(), WIDTH * HEIGHT * 4)
        digest = sha256(pixels)
        require(re.fullmatch(r"[0-9a-f]{64}", frame["digest"]) is not None, "Malformed captured framebuffer digest")
        require(digest == frame["digest"], f"Doom frame mismatch at tick {tick}: CPU {digest}, GPU {frame['digest']}")
        cpu_rows.append({"sequence": index + 1, "tick": tick, "keys": keys, "renders": index + 1,
                         "digest": digest, "diagnostics": expected})
        observed_keys |= keys
        previous_tick, previous_keys = tick, keys
    (output / "doom-cpu-reference.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cpu_rows))
    (output / "doom-cpu-frame.bgra").write_bytes(pixels)
    return {"framesCompared": len(cpu_rows), "diagnosticValuesCompared": len(cpu_rows) * 16,
            "frameBytesComparedBySHA256": len(cpu_rows) * WIDTH * HEIGHT * 4,
            "maxTicksPerFrame": maximum_gap, "observedKeyMask": observed_keys,
            "finalDigest": cpu_rows[-1]["digest"], "cpuReplaySeconds": time.monotonic() - start,
            "reference": "Native Doom C with documented cache padding, independent of WASM and Metal lowering",
            "referenceLibrarySHA256": sha256(library_path.read_bytes())}


def audit_run(directory, minimum_seconds, scripted, expected_keys):
    summary = json.loads((directory / "summary.json").read_text())
    samples = jsonl(directory / "samples.jsonl")
    frames = jsonl(directory / "doom-frames.jsonl")
    require(summary.get("game") == "doom", "Recording is not the full Doom integration")
    require(summary["status"] == 4 and not summary["error"], "Metal command did not complete successfully")
    require(summary["seconds"] >= minimum_seconds, "Run is shorter than requested validation scope")
    require(summary["dispatches"] == summary["submissions"] == summary["threadgroups"] == 1, "Need exactly one GPU submission, dispatch and threadgroup")
    require(summary["threads"] == 96 and summary["simdGroups"] == 3, "Expected three SIMD roles in one96-thread group")
    require(summary["doneFlags"] == [1, 1, 1], "A combined workload did not shut down cleanly")
    require(summary["doomResult"] == summary["doomFault"] == 0 and summary["doomDone"] == 1, "Doom initialization, trap, or shutdown error")
    require(summary["stopMilliseconds"] < 2000, "Combined GPU stop took over two seconds")
    require(all(value > 0 for value in summary["peerProgress"]), "Missing in-kernel observation of peer workloads")
    require(summary["snapshotErrors"] == 0 and summary["snapshotVerifiedCopies"] > 5, "Evolving model snapshots were not verified")
    require(summary["observedVersions"] > 5 and summary["inferenceVersion"] > 128, "Inference did not consume successive updated models")
    require(all(math.isfinite(summary[key]) for key in ("firstLoss", "finalLoss")), "Nonfinite training loss")
    require(summary["finalLoss"] < summary["firstLoss"] * .9, "Training loss did not substantially decrease")
    live = [sample for sample in samples if sample["status"] == 3]
    require(len(live) >= max(5, int(minimum_seconds * 4)), "Insufficient live observations before command completion")
    overlap = 0
    for before, after in zip(live, live[1:]):
        require(after["t"] > before["t"], "Nonmonotonic sample timestamp")
        require(all(after[key] >= before[key] for key in ("steps", "tokens", "frames", "version")), "Work counters regressed")
        overlap += all(after[key] > before[key] for key in ("steps", "tokens", "frames"))
    require(overlap >= max(3, len(live) // 10), "Insufficient intervals with simultaneous training,inference,Doom progress")
    require(all(count >= max(3, int(minimum_seconds)) for count in summary["liveProgressSamples"]), "A persistent workload stalled")
    require(all(frame["status"] == 3 for frame in frames[:-1]), "A Doom frame before the final one was not captured from the running command")
    require(frames[-1]["status"] in (3, 4), "Invalid final frame command status")
    require(len(frames) == summary["doomFrameCount"] == summary["doomRenders"], "Published frame/render evidence is incomplete")
    require(frames[-1]["tick"] == summary["doomTick"], "Final tick differs from last captured publication")
    require(all(a["t"] <= b["t"] for a, b in zip(frames, frames[1:])), "Nonmonotonic frame timestamp")
    require(len(frames) >= max(5, int(minimum_seconds * 2)), "Too few actual Doom frames for a live demo")
    require(summary["doomTick"] > 0, "Doom never advanced a game tick")
    if scripted or expected_keys:
        observed = 0
        for frame in frames:
            observed |= frame["keys"]
        require(observed & expected_keys == expected_keys, f"GPU did not observe required native controls: {expected_keys:#x}, observed {observed:#x}")
        require(len({f["keys"] for f in frames if f["keys"]}) >= 2, "Native input recording lacks distinct controls")
        positions = {(f["diagnostics"][4], f["diagnostics"][5]) for f in frames}
        angles = {f["diagnostics"][7] for f in frames}
        require(len(positions) > 1 or len(angles) > 1, "Doom player did not respond to movement/turn controls")
        matched = {s["requestedKeys"] for s in live if s["requestedKeys"] and s["keys"] == s["requestedKeys"]}
        require(len(matched) >= 2, "Host controls were not observed by the GPU during the running command")
    tokens = jsonl(directory / "tokens.jsonl")
    text = (directory / "generated.txt").read_text()
    require(summary["skippedTokens"] == 0 and len(tokens) == summary["tokens"] == len(text), "Token publication was lost")
    require([row["index"] for row in tokens] == list(range(len(tokens))), "Token evidence is missing or out of order")
    require(all(0 <= row["token"] < 96 for row in tokens), "Invalid language-model token")
    decoded = "".join("\n" if row["token"] == 95 else chr(row["token"] + 32) for row in tokens)
    require(decoded == text == summary["text"], "Text does not match published model tokens")
    require(len(text) > 20 and len(set(text)) > 5, "Insufficient text generated while Doom ran")
    versions = {row["version"] for row in tokens}
    require(len(versions) > 5 and len(versions) == summary["tokenWeightVersions"], "Text did not use successive trained weights")
    require(summary["tokenVersionRegressions"] == 0 and all(a["version"] <= b["version"] for a, b in zip(tokens, tokens[1:])), "Inference regressed to older weights")
    require(all(row["version"] <= summary["steps"] for row in tokens), "A token references a nonexistent model")
    raw = (directory / "doom-frame.bgra").read_bytes()
    require(len(raw) == WIDTH * HEIGHT * 4, "Wrong Doom framebuffer size")
    require(sha256(raw) == frames[-1]["digest"], "Saved raw Doom frame differs from the final captured digest")
    require(all(raw[i] == 0 for i in range(3, len(raw), 4)), "Raw Doom framebuffer must retain original zero alpha")
    require(len({raw[i:i+4] for i in range(0, len(raw), 4)}) > 20, "Doom framebuffer lacks game image content")
    host = (ROOT / "src/main.mm").read_text()
    shader = (ROOT / "src/Persistent.metal").read_text()
    require(summary["hostSourceSHA256"] == sha256(host.encode()), "Host source differs from the recorded run")
    combined = ((DOOM / "build/doom-engine.metal").read_text() + "\n" +
                (DOOM / "app/persistent_protocol.h").read_text() + "\n#define DOOM_EMBEDDED 1\n" +
                (DOOM / "app/persistent.metal").read_text() + "\n" +
                (ROOT / "src/Config.h").read_text() + "\n" + shader)
    require(summary["shaderSHA256"] == sha256(combined.encode()), "Compiled shader source differs from the recorded run")
    require(len(re.findall(r"\[enc\s+dispatchThreadgroups:", host)) == 1, "Host source has multiple/absent workload dispatch sites")
    require(len(re.findall(r"\[command\s+commit\]", host)) == 1, "Host source has multiple/absent workload submissions")
    require(re.findall(r"kernel\s+void\s+(\w+)", shader) == ["persistent"], "Unexpected combined shader entry points")
    require("threadgroup_barrier(" not in shader, "Full-group barriers can block independent roles")
    return summary, frames, {"seconds": summary["seconds"], "liveSamples": len(live), "overlapIntervals": overlap,
                             "steps": summary["steps"], "tokens": len(tokens), "tokenWeightVersions": len(versions),
                             "doomTicks": summary["doomTick"], "doomFrames": len(frames),
                             "firstLoss": summary["firstLoss"], "finalLoss": summary["finalLoss"],
                             "stopMilliseconds": summary["stopMilliseconds"], "scriptedControlsChecked": scripted,
                             "hostSourceSHA256": sha256(host.encode()), "outerShaderSHA256": sha256(shader.encode()),
                             "compiledShaderSHA256": summary["shaderSHA256"], "binarySHA256": summary["binarySHA256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--wad", required=True, type=Path)
    parser.add_argument("--library", type=Path, default=DOOM / "build/libdoom_reference.dylib")
    parser.add_argument("--min-seconds", type=float, default=10)
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--expect-keys", type=lambda value: int(value, 0), default=0,
                        help="Required observed Doom key bits, e.g.0x4c for left,right,fire")
    args = parser.parse_args()
    report_path = args.directory / "doom-validation.json"
    report_path.unlink(missing_ok=True)
    require(args.min_seconds > 0 and math.isfinite(args.min_seconds), "Minimum duration must be finite and positive")
    require(0 <= args.expect_keys < 1 << len(KEY_CODES), "Expected key mask is outside native Doom controls")
    summary, frames, report = audit_run(args.directory, args.min_seconds, args.scripted, args.expect_keys)
    require(sha256(args.wad.read_bytes()) == summary["wadSHA256"], "Reference IWAD differs from the GPU run")
    report["doomReference"] = native_reference(args.wad, args.library, frames, args.directory)
    report["wadSHA256"] = summary["wadSHA256"]
    report["passed"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
