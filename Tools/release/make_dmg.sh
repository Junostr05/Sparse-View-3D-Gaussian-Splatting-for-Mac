#!/bin/bash
# Builds the Release app and packs it into a DMG in build/.
#
#   Tools/release/make_dmg.sh                        # ad-hoc signed, for local use only
#   Tools/release/make_dmg.sh --sign "Developer ID Application: Name (TEAMID)" \
#                             --notarize <notarytool-keychain-profile>
#   Tools/release/make_dmg.sh --bundle-model <DepthAnything3BaseQ8.mlpackage>
#
# --bundle-model copies the converted Depth Anything 3 package into the app (Contents/Resources/Models) and the
# Depth Anything 3 licence and notice into Contents/Resources/Licenses, then signs the app again. Without it the
# DMG has no model, as before.
#
# Signing and notarization need an Apple Developer ID certificate in the keychain and a notarytool
# profile (`xcrun notarytool store-credentials`). Without them the DMG still works on this Mac, but
# Gatekeeper blocks it on other Macs. Nothing here uploads anything.
set -euo pipefail

IDENTITY=""
PROFILE=""
BUNDLE_MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sign) IDENTITY="$2"; shift 2 ;;
    --notarize) PROFILE="$2"; shift 2 ;;
    --bundle-model) BUNDLE_MODEL="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -n "$PROFILE" ] && [ -z "$IDENTITY" ]; then
  echo "--notarize needs --sign" >&2; exit 2
fi
if [ -n "$BUNDLE_MODEL" ]; then
  # Made absolute here: the script changes directory below, and a relative path must keep its meaning.
  BUNDLE_PARENT=$(cd "$(dirname "$BUNDLE_MODEL")" 2>/dev/null && pwd) || { echo "--bundle-model: no such folder: $BUNDLE_MODEL" >&2; exit 2; }
  BUNDLE_MODEL="$BUNDLE_PARENT/$(basename "$BUNDLE_MODEL")"
fi

cd "$(dirname "$0")/../.."
ROOT="$PWD"
BUILD="$ROOT/build"
mkdir -p "$BUILD"

# Checked before the build, so a bad path fails at once.
MODEL_FILES=(Manifest.json Data/com.apple.CoreML/model.mlmodel Data/com.apple.CoreML/weights/weight.bin)
LICENSE_DIR="$ROOT/Tools/release/licenses"
LICENSE_FILES=(DepthAnything3-LICENSE.txt DepthAnything3-NOTICE.txt)
if [ -n "$BUNDLE_MODEL" ]; then
  [ -d "$BUNDLE_MODEL" ] || { echo "--bundle-model: not a package directory: $BUNDLE_MODEL" >&2; exit 2; }
  for f in "${MODEL_FILES[@]}"; do
    [ -f "$BUNDLE_MODEL/$f" ] || { echo "--bundle-model: $BUNDLE_MODEL is not a valid package (missing $f)" >&2; exit 2; }
  done
  for f in "${LICENSE_FILES[@]}"; do
    [ -f "$LICENSE_DIR/$f" ] || { echo "missing licence file: $LICENSE_DIR/$f" >&2; exit 2; }
  done
fi

xcodegen generate >/dev/null

# Apple Silicon only (ARCHITECTURE.md): the default Release build would also compile x86_64.
ARCH_ARGS=(ARCHS=arm64 ONLY_ACTIVE_ARCH=NO)
SIGNING_ARGS=(CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO)
if [ -n "$IDENTITY" ]; then
  SIGNING_ARGS=(CODE_SIGN_IDENTITY="$IDENTITY" CODE_SIGN_STYLE=Manual OTHER_CODE_SIGN_FLAGS="--timestamp")
fi

xcodebuild -project SparseSplat.xcodeproj -scheme SparseSplat -configuration Release \
  -destination 'platform=macOS' -derivedDataPath "$BUILD/DerivedData" \
  "${ARCH_ARGS[@]}" "${SIGNING_ARGS[@]}" build | tail -3

APP="$BUILD/DerivedData/Build/Products/Release/SparseSplat.app"
[ -d "$APP" ] || { echo "app not found at $APP" >&2; exit 1; }
# A clean Release build has been seen to skip the compiled icon (Assets.car); a second run picks it up.
if [ ! -f "$APP/Contents/Resources/Assets.car" ]; then
  echo "icon missing after the first build, building again"
  xcodebuild -project SparseSplat.xcodeproj -scheme SparseSplat -configuration Release \
    -destination 'platform=macOS' -derivedDataPath "$BUILD/DerivedData" \
    "${ARCH_ARGS[@]}" "${SIGNING_ARGS[@]}" build | tail -3
  [ -f "$APP/Contents/Resources/Assets.car" ] || { echo "the app still has no compiled icon" >&2; exit 1; }
fi

# Models and licences from an earlier --bundle-model run may still sit in the derived app. Remove them, so a build
# without the flag never ships a model.
rm -rf "$APP/Contents/Resources/Models" "$APP/Contents/Resources/Licenses"
if [ -n "$BUNDLE_MODEL" ]; then
  # Folder and file names must match ModelDescriptor.depthAnything3Base (id and packageName) in SplatIO.
  MODEL_DEST="$APP/Contents/Resources/Models/depth-anything-3-base/DepthAnything3BaseQ8.mlpackage"
  mkdir -p "$(dirname "$MODEL_DEST")"
  cp -R "$BUNDLE_MODEL" "$MODEL_DEST"
  for f in "${MODEL_FILES[@]}"; do
    [ -f "$MODEL_DEST/$f" ] || { echo "the copied package is incomplete (missing $f)" >&2; exit 1; }
  done
  LICENSE_DEST="$APP/Contents/Resources/Licenses"
  mkdir -p "$LICENSE_DEST"
  for f in "${LICENSE_FILES[@]}"; do
    cp "$LICENSE_DIR/$f" "$LICENSE_DEST/$f"
  done
  # Adding resources invalidates the signature, so the bundle is signed again.
  if [ -n "$IDENTITY" ]; then
    codesign --force --deep --timestamp --options runtime --sign "$IDENTITY" "$APP"
  else
    codesign --force --deep --sign - "$APP"
  fi
  echo "bundled depth model added: $(du -sh "$(dirname "$MODEL_DEST")" | cut -f1 | tr -d ' ') (package) and the licences; app signed again"
fi

VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$APP/Contents/Info.plist")
DMG="$BUILD/SparseSplat-$VERSION.dmg"
STAGING="$BUILD/dmg-staging"
rm -rf "$STAGING" "$DMG"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

hdiutil create -volname "SparseSplat" -srcfolder "$STAGING" -ov -format UDZO "$DMG" >/dev/null

if [ -n "$IDENTITY" ]; then
  codesign --force --timestamp --sign "$IDENTITY" "$DMG"
fi
if [ -n "$PROFILE" ]; then
  xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$DMG"
fi

echo "$DMG"
codesign -dv "$APP" 2>&1 | grep -E "Identifier|Signature|TeamIdentifier" || true
