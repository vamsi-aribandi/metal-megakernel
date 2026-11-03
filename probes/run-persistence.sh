#!/bin/zsh
set -eu
cd "${0:A:h}"
mkdir -p build
xcrun clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal persistence.mm -o build/persistence
exec ./build/persistence "$@"
