#!/bin/zsh
set -eu
cd "${0:A:h}"
./build.sh
exec ./build/metal-megakernel "$@"
