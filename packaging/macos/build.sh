#!/usr/bin/env bash
# Builds Prism for macOS 11+ (Apple silicon and Intel in one universal app) and writes
#   dist/PrismVideoEnhancer-<version>-macos-universal.dmg
#
# Run on a Mac with the Xcode command line tools (`xcode-select --install`) and rustup:
#   bash packaging/macos/build.sh
#
#   SIGN_IDENTITY   e.g. "Developer ID Application: Name (TEAMID)". Without it the app is
#                   ad-hoc signed: it runs, but Gatekeeper asks for confirmation on first open
#                   (right-click > Open) when the .dmg was downloaded.
#   NOTARY_PROFILE  keychain profile for `xcrun notarytool` (needs SIGN_IDENTITY); notarizes and
#                   staples the .dmg.
#   ORT_VERSION     ONNX Runtime to bundle (default 1.20.1, universal2 build with Core ML; MIT).
#
# FFmpeg is not bundled: install it with `brew install ffmpeg` (Prism looks in /opt/homebrew/bin
# and /usr/local/bin, so apps opened from Finder find it too).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
ORT_VERSION="${ORT_VERSION:-1.20.1}"
APP_NAME="Prism Video Enhancer"
BUNDLE_ID="io.github.prism-video-enhancer"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
DIST="$ROOT/dist"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== building $APP_NAME $VERSION (arm64 + x86_64)"
rustup target add aarch64-apple-darwin x86_64-apple-darwin >/dev/null
export MACOSX_DEPLOYMENT_TARGET=11.0
for t in aarch64-apple-darwin x86_64-apple-darwin; do
  cargo build --profile dist --target "$t" -p prism-app -p prism-cli
done
for b in prism-app prism; do
  lipo -create "$TARGET_DIR/aarch64-apple-darwin/dist/$b" "$TARGET_DIR/x86_64-apple-darwin/dist/$b" -output "$WORK/$b"
done

echo "== ONNX Runtime $ORT_VERSION"
ORT_PKG="onnxruntime-osx-universal2-$ORT_VERSION"
curl -sSL --fail -o "$WORK/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_PKG.tgz"
tar -xzf "$WORK/ort.tgz" -C "$WORK"

echo "== app bundle"
APP="$WORK/stage/$APP_NAME.app"
mkdir -p "$APP/Contents/MacOS/runtime" "$APP/Contents/Resources"
install -m 755 "$WORK/prism-app" "$WORK/prism" "$APP/Contents/MacOS/"
# Prism loads runtime/libonnxruntime.dylib next to its executable.
cp -L "$WORK/$ORT_PKG/lib/libonnxruntime.$ORT_VERSION.dylib" "$APP/Contents/MacOS/runtime/libonnxruntime.dylib"
cp "$WORK/$ORT_PKG/LICENSE" "$APP/Contents/Resources/ONNXRUNTIME-LICENSE.txt"
[ -f "$WORK/$ORT_PKG/ThirdPartyNotices.txt" ] && cp "$WORK/$ORT_PKG/ThirdPartyNotices.txt" "$APP/Contents/Resources/ONNXRUNTIME-ThirdPartyNotices.txt"
cp README.md "$APP/Contents/Resources/"

"$WORK/prism-app" --write-icons "$WORK/icons"
ICONSET="$WORK/prism.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  cp "$WORK/icons/prism-$s.png" "$ICONSET/icon_${s}x${s}.png"
  cp "$WORK/icons/prism-$((s * 2)).png" "$ICONSET/icon_${s}x${s}@2x.png"
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/prism.icns"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>prism-app</string>
  <key>CFBundleIconFile</key><string>prism</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

echo "== signing"
if [ -n "${SIGN_IDENTITY:-}" ]; then
  # Inside out: the runtime library, the helper binary, then the app.
  for f in "$APP/Contents/MacOS/runtime/libonnxruntime.dylib" "$APP/Contents/MacOS/prism"; do
    codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$f"
  done
  codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP"

echo "== dmg"
ln -s /Applications "$WORK/stage/Applications"
mkdir -p "$DIST"
DMG="$DIST/PrismVideoEnhancer-$VERSION-macos-universal.dmg"
rm -f "$DMG"
# hdiutil occasionally fails with "Resource busy" on build machines; retry a few times.
for attempt in 1 2 3; do
  hdiutil create -volname "$APP_NAME" -srcfolder "$WORK/stage" -ov -format UDZO "$DMG" >/dev/null && break
  [ "$attempt" = 3 ] && exit 1
  sleep 5
done
if [ -n "${SIGN_IDENTITY:-}" ]; then
  codesign --force --sign "$SIGN_IDENTITY" "$DMG"
  if [ -n "${NOTARY_PROFILE:-}" ]; then
    xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
    xcrun stapler staple "$DMG"
  fi
fi
echo "== done: $DMG"
echo "Command-line tool: sudo ln -sf \"/Applications/$APP_NAME.app/Contents/MacOS/prism\" /usr/local/bin/prism"
