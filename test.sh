#!/bin/zsh
set -eu
cd "${0:A:h}"
./build.sh
python3 tests/numerics.py
./build/metal-megakernel --headless --seconds 10 --test-input --output results/check
python3 tests/validate.py results/check --min-seconds 10 --scripted
