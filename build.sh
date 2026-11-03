#!/bin/zsh
set -eu
cd "${0:A:h}"
mkdir -p build
xcrun clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra \
  -framework Cocoa -framework Metal src/main.mm -o build/metal-megakernel
mkdir -p 'build/Metal Megakernel.app/Contents/MacOS'
cp src/Info.plist 'build/Metal Megakernel.app/Contents/Info.plist'
ln -sf ../../../metal-megakernel 'build/Metal Megakernel.app/Contents/MacOS/metal-megakernel'
print 'Built build/metal-megakernel'
