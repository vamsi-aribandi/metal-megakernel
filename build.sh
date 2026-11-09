#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ ! -f third_party/metal-doom/tools/to_metal.py ]]; then
  git submodule update --init --recursive
fi
# Rebuild when an engine input changes, including an updated submodule checkout.
python3 tools/build_doom.py
mkdir -p build
xcrun clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra \
  -Wl,-sectcreate,__TEXT,__info_plist,src/Info.plist \
  -framework Cocoa -framework Metal src/main.mm -o build/metal-megakernel
mkdir -p 'build/Metal Megakernel.app/Contents/MacOS'
cp src/Info.plist 'build/Metal Megakernel.app/Contents/Info.plist'
ln -sf ../../../metal-megakernel 'build/Metal Megakernel.app/Contents/MacOS/metal-megakernel'
print 'Built build/metal-megakernel'
