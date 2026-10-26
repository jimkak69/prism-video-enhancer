#!/usr/bin/env bash
# Builds Prism for Linux x86_64 and writes to dist/:
#   prism-video-enhancer-<version>-linux-x86_64.tar.gz   (per-user install via install.sh)
#   prism-video-enhancer_<version>_amd64.deb             (system install under /opt)
#
# Run on Linux (or `packaging/linux/docker-build.ps1` from Windows, which uses Debian 11 so the
# binaries run on glibc 2.31 and newer: Ubuntu 20.04+, Debian 11+, Fedora 32+).
# Needs: cargo, curl, tar, dpkg-deb (for the .deb).
#
#   ORT_VERSION   ONNX Runtime to bundle (default 1.20.1, CPU build; MIT licence)
#   ORT_FLAVOR    "cpu" (default) or "gpu" for the CUDA 12 build (much larger; needs CUDA/cuDNN)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
ORT_VERSION="${ORT_VERSION:-1.20.1}"
ORT_FLAVOR="${ORT_FLAVOR:-cpu}"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
DIST="$ROOT/dist"
NAME="prism-video-enhancer"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== building $NAME $VERSION"
cargo build --profile dist -p prism-app -p prism-cli
BIN="$TARGET_DIR/dist"

echo "== ONNX Runtime $ORT_VERSION ($ORT_FLAVOR)"
ORT_PKG="onnxruntime-linux-x64-$ORT_VERSION"
[ "$ORT_FLAVOR" = gpu ] && ORT_PKG="onnxruntime-linux-x64-gpu-$ORT_VERSION"
curl -sSL --fail -o "$WORK/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_PKG.tgz"
tar -xzf "$WORK/ort.tgz" -C "$WORK"
ORT_DIR="$WORK/$ORT_PKG"

echo "== icons"
"$BIN/prism-app" --write-icons "$WORK/icons"

# Application folder shared by both packages.
APP="$WORK/app"
mkdir -p "$APP/runtime"
install -m 755 "$BIN/prism-app" "$BIN/prism" "$APP/"
# Prism loads runtime/libonnxruntime.so; providers (CUDA) load from the same folder.
cp -L "$ORT_DIR/lib/libonnxruntime.so.$ORT_VERSION" "$APP/runtime/libonnxruntime.so"
for f in "$ORT_DIR"/lib/libonnxruntime_providers_*.so; do [ -e "$f" ] && cp -L "$f" "$APP/runtime/"; done
cp "$ORT_DIR/LICENSE" "$APP/runtime/ONNXRUNTIME-LICENSE.txt"
[ -f "$ORT_DIR/ThirdPartyNotices.txt" ] && cp "$ORT_DIR/ThirdPartyNotices.txt" "$APP/runtime/ONNXRUNTIME-ThirdPartyNotices.txt"
cp "$WORK/icons/prism-256.png" "$APP/prism.png"
cp README.md "$APP/README.md"

desktop() { # $1 = Exec path, $2 = Icon name or path
  cat <<EOF
[Desktop Entry]
Type=Application
Name=Prism Video Enhancer
GenericName=Video Enhancer
Comment=Upscale, restore, interpolate and stabilize video
Exec=$1 %F
Icon=$2
Terminal=false
Categories=AudioVideo;Video;AudioVideoEditing;
MimeType=video/mp4;video/quicktime;video/x-matroska;video/webm;video/x-msvideo;video/mpeg;video/mp2t;
StartupWMClass=$NAME
EOF
}

mkdir -p "$DIST"

echo "== tar.gz"
TGZ_ROOT="$WORK/$NAME-$VERSION-linux-x86_64"
cp -r "$APP" "$TGZ_ROOT"
mkdir -p "$TGZ_ROOT/icons"
for s in 16 32 48 64 128 256 512; do cp "$WORK/icons/prism-$s.png" "$TGZ_ROOT/icons/"; done
cat > "$TGZ_ROOT/install.sh" <<'EOF'
#!/usr/bin/env bash
# Installs Prism for the current user: ~/.local/share/prism-video-enhancer, launchers in
# ~/.local/bin, a menu entry and icons. Run ./install.sh --uninstall to remove it.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
DEST="$DATA/prism-video-enhancer"
BINDIR="$HOME/.local/bin"
if [ "${1:-}" = "--uninstall" ]; then
  rm -rf "$DEST" "$BINDIR/prism-app" "$BINDIR/prism" "$DATA/applications/prism-video-enhancer.desktop"
  for s in 16 32 48 64 128 256 512; do rm -f "$DATA/icons/hicolor/${s}x${s}/apps/prism-video-enhancer.png"; done
  echo "Prism Video Enhancer removed (settings and models in ~/.local/share/Prism Video Enhancer were kept)."
  exit 0
fi
rm -rf "$DEST"
mkdir -p "$DEST" "$BINDIR" "$DATA/applications"
cp -r "$SRC/prism-app" "$SRC/prism" "$SRC/runtime" "$SRC/prism.png" "$SRC/README.md" "$DEST/"
ln -sf "$DEST/prism-app" "$BINDIR/prism-app"
ln -sf "$DEST/prism" "$BINDIR/prism"
for s in 16 32 48 64 128 256 512; do
  mkdir -p "$DATA/icons/hicolor/${s}x${s}/apps"
  cp "$SRC/icons/prism-$s.png" "$DATA/icons/hicolor/${s}x${s}/apps/prism-video-enhancer.png"
done
sed -e "s|@EXEC@|$DEST/prism-app|" "$SRC/prism-video-enhancer.desktop.in" > "$DATA/applications/prism-video-enhancer.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q "$DATA/icons/hicolor" >/dev/null 2>&1 || true
echo "Installed to $DEST"
command -v ffmpeg >/dev/null || echo "Note: FFmpeg was not found. Install it with your package manager, e.g. sudo apt install ffmpeg"
case ":$PATH:" in *":$BINDIR:"*) ;; *) echo "Note: add $BINDIR to PATH to run 'prism' from a terminal." ;; esac
EOF
chmod 755 "$TGZ_ROOT/install.sh"
desktop "@EXEC@" "prism-video-enhancer" > "$TGZ_ROOT/prism-video-enhancer.desktop.in"
tar -czf "$DIST/$NAME-$VERSION-linux-x86_64.tar.gz" -C "$WORK" "$(basename "$TGZ_ROOT")"

if command -v dpkg-deb >/dev/null; then
  echo "== deb"
  DEB="$WORK/deb"
  mkdir -p "$DEB/DEBIAN" "$DEB/opt" "$DEB/usr/bin" "$DEB/usr/share/applications"
  cp -r "$APP" "$DEB/opt/$NAME"
  ln -s "/opt/$NAME/prism-app" "$DEB/usr/bin/prism-app"
  ln -s "/opt/$NAME/prism" "$DEB/usr/bin/prism"
  desktop "/opt/$NAME/prism-app" "$NAME" > "$DEB/usr/share/applications/$NAME.desktop"
  for s in 16 32 48 64 128 256 512; do
    mkdir -p "$DEB/usr/share/icons/hicolor/${s}x${s}/apps"
    cp "$WORK/icons/prism-$s.png" "$DEB/usr/share/icons/hicolor/${s}x${s}/apps/$NAME.png"
  done
  SIZE="$(du -sk "$DEB" | cut -f1)"
  cat > "$DEB/DEBIAN/control" <<EOF
Package: $NAME
Version: $VERSION
Section: video
Priority: optional
Architecture: amd64
Maintainer: Prism Video Enhancer <noreply@localhost>
Installed-Size: $SIZE
Depends: ffmpeg, curl, libc6 (>= 2.31)
Recommends: libarchive-tools | 7zip | p7zip-full, xdg-desktop-portal
Description: Video upscaling, restoration and frame interpolation
 Prism Video Enhancer upscales, denoises, interpolates, stabilizes and converts video to HDR
 using FFmpeg, native kernels and ONNX Runtime networks (bundled CPU runtime).
EOF
  dpkg-deb --root-owner-group --build "$DEB" "$DIST/${NAME}_${VERSION}_amd64.deb" >/dev/null
fi

echo "== done"
ls -la "$DIST"
