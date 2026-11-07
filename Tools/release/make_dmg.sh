#!/bin/bash
# Builds the Release app and packs it into a DMG in build/.
#
#   Tools/release/make_dmg.sh                        # ad-hoc signed, for local use only
#   Tools/release/make_dmg.sh --sign "Developer ID Application: Name (TEAMID)" \
#                             --notarize <notarytool-keychain-profile>
#
# Signing and notarization need an Apple Developer ID certificate in the keychain and a notarytool
# profile (`xcrun notarytool store-credentials`). Without them the DMG still works on this Mac, but
# Gatekeeper blocks it on other Macs. Nothing here uploads anything.
set -euo pipefail

IDENTITY=""
PROFILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sign) IDENTITY="$2"; shift 2 ;;
    --notarize) PROFILE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -n "$PROFILE" ] && [ -z "$IDENTITY" ]; then
  echo "--notarize needs --sign" >&2; exit 2
fi

cd "$(dirname "$0")/../.."
ROOT="$PWD"
BUILD="$ROOT/build"
mkdir -p "$BUILD"

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
