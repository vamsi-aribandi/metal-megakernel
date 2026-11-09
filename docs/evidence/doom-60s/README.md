# Recorded Doom integration evidence

One native AppKit run on Apple M4, macOS 26.5.1, with 60 seconds of play after
startup. The command buffer stayed running while all three workers progressed.
`doom-validation.json` records independent native-C replay of every Doom
publication and all diagnostics. `numerics.json` records independent model math.
The source and binary hashes are in the summary and validation report.

The IWAD is not included. Its SHA256 identifies the tested shareware asset.
Frame records contain hashes, inputs, and diagnostics; no game image bytes are
included here. Full local recordings under `results/doom-final-60s` additionally
contain final raw/display frames and CPU replay output.

`doom-cache-regression.json` records both serial and cooperative standalone
Metal replay of the earlier 2,056-frame schedule that exposed cache-boundary
reads. Both builds use documented zero-filled cache padding.

The native window ran during the test, but the Mac was locked when computer-use
inspection was attempted, so no visual inspection or screenshot is claimed.
