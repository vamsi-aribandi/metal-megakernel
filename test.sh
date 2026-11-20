#!/bin/zsh
set -eu
cd "${0:A:h}"
wad=${1:-${DOOM_WAD:-}}
if [[ -z "$wad" ]]; then
  print -u2 'Usage: ./test.sh /path/to/doom1.wad (or set DOOM_WAD)'
  exit 2
fi
./build.sh
python3 tests/numerics.py
./third_party/metal-doom/engine/build_reference.sh
./build/metal-megakernel --headless --wad "$wad" --seconds 12 --test-input --output results/check
python3 tests/validate_doom.py results/check --wad "$wad" --min-seconds 12 --scripted --expect-keys 207
