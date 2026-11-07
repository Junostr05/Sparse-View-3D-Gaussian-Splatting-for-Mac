#!/bin/bash
# Renders the artwork and writes App/Assets.xcassets/AppIcon.appiconset (all macOS sizes).
set -euo pipefail
cd "$(dirname "$0")/../.."
WORK="$(mktemp -d)"
swift Tools/branding/make_icon.swift "$WORK"
SET="App/Assets.xcassets/AppIcon.appiconset"
mkdir -p "$SET" docs/assets
rm -f "$SET"/*.png
write() { sips -z "$1" "$1" "$WORK/icon_1024.png" --out "$SET/$2" >/dev/null; }
write 16 icon_16x16.png;     write 32 icon_16x16@2x.png
write 32 icon_32x32.png;     write 64 icon_32x32@2x.png
write 128 icon_128x128.png;  write 256 icon_128x128@2x.png
write 256 icon_256x256.png;  write 512 icon_256x256@2x.png
write 512 icon_512x512.png;  cp "$WORK/icon_1024.png" "$SET/icon_512x512@2x.png"
cat > "$SET/Contents.json" <<'JSON'
{
  "images" : [
    { "filename" : "icon_16x16.png",      "idiom" : "mac", "scale" : "1x", "size" : "16x16" },
    { "filename" : "icon_16x16@2x.png",   "idiom" : "mac", "scale" : "2x", "size" : "16x16" },
    { "filename" : "icon_32x32.png",      "idiom" : "mac", "scale" : "1x", "size" : "32x32" },
    { "filename" : "icon_32x32@2x.png",   "idiom" : "mac", "scale" : "2x", "size" : "32x32" },
    { "filename" : "icon_128x128.png",    "idiom" : "mac", "scale" : "1x", "size" : "128x128" },
    { "filename" : "icon_128x128@2x.png", "idiom" : "mac", "scale" : "2x", "size" : "128x128" },
    { "filename" : "icon_256x256.png",    "idiom" : "mac", "scale" : "1x", "size" : "256x256" },
    { "filename" : "icon_256x256@2x.png", "idiom" : "mac", "scale" : "2x", "size" : "256x256" },
    { "filename" : "icon_512x512.png",    "idiom" : "mac", "scale" : "1x", "size" : "512x512" },
    { "filename" : "icon_512x512@2x.png", "idiom" : "mac", "scale" : "2x", "size" : "512x512" }
  ],
  "info" : { "author" : "xcode", "version" : 1 }
}
JSON
cat > App/Assets.xcassets/Contents.json <<'JSON'
{ "info" : { "author" : "xcode", "version" : 1 } }
JSON
cp "$WORK/logo.png" docs/assets/logo.png
cp "$WORK/mark_1024.png" docs/assets/mark.png
rm -rf "$WORK"
echo "wrote $SET, docs/assets/logo.png and docs/assets/mark.png"
