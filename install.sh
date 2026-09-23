#!/usr/bin/env bash
# glyphc installer.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/MrModelOS/Glyph/main/install.sh | bash
#   bash install.sh [--cargo] [--prefix DIR] [--version v2.1.0]
set -euo pipefail

REPO="MrModelOS/Glyph"
BINARY="glyphc"
CARGO_GIT_URL="https://github.com/${REPO}"
PREFIX="/usr/local/bin"
USE_CARGO=0
VERSION=""

usage() {
    cat <<'EOF'
glyphc installer

Usage:
  install.sh [OPTIONS]

Options:
  -h, --help        Show this help
  --cargo           Skip the release archive and install via cargo
  --prefix <dir>    Install directory (default: /usr/local/bin)
  --version <tag>   Install a specific release tag (for example v2.1.0)

The installer selects the release asset matching the current operating system
and CPU architecture. If the archive is unavailable, it falls back to:

  cargo install --git https://github.com/MrModelOS/Glyph --locked
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --cargo)
            USE_CARGO=1
            shift
            ;;
        --prefix)
            [[ $# -ge 2 ]] || { echo "--prefix requires a directory" >&2; exit 2; }
            PREFIX="$2"
            shift 2
            ;;
        --version)
            [[ $# -ge 2 ]] || { echo "--version requires a tag" >&2; exit 2; }
            VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

have() { command -v "$1" >/dev/null 2>&1; }

move_binary() {
    local source="$1"
    local destination="${PREFIX}/${BINARY}"
    mkdir -p "$PREFIX"
    if [[ -w "$PREFIX" ]]; then
        install -m 0755 "$source" "$destination"
    elif have sudo; then
        sudo install -m 0755 "$source" "$destination"
    else
        echo "Cannot write to ${PREFIX}; use --prefix or sudo." >&2
        return 1
    fi
    echo "Installed ${BINARY} to ${destination}"
    "$destination" --version || true
}

install_from_tarball() {
    have curl || { echo "curl not found; skipping release archive." >&2; return 1; }
    have tar || { echo "tar not found; skipping release archive." >&2; return 1; }

    local platform arch tag url tmpdir archive binary
    case "$(uname -s)" in
        Linux*) platform="linux" ;;
        Darwin*) platform="macos" ;;
        MINGW*|MSYS*|CYGWIN*) platform="windows" ;;
        *) echo "Unsupported operating system: $(uname -s)" >&2; return 1 ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) echo "Unsupported CPU architecture: $(uname -m)" >&2; return 1 ;;
    esac

    if [[ -z "$VERSION" ]]; then
        local api_json
        api_json=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest") || return 1
        tag=$(printf '%s\n' "$api_json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v\{0,1\}\([^"]*\)".*/v\1/p' | head -n 1)
        [[ -n "$tag" ]] || { echo "Could not determine the latest release tag." >&2; return 1; }
        VERSION="$tag"
    fi
    [[ "$VERSION" == v* ]] || VERSION="v${VERSION}"
    url="https://github.com/${REPO}/releases/download/${VERSION}/glyphc-${VERSION}-${platform}-${arch}.tar.gz"

    tmpdir=$(mktemp -d)
    archive="${tmpdir}/glyphc.tar.gz"
    echo "Downloading ${url} ..."
    if ! curl -fL --retry 2 --connect-timeout 15 "$url" -o "$archive"; then
        rm -rf "$tmpdir"
        return 1
    fi
    if ! tar -xzf "$archive" -C "$tmpdir"; then
        rm -rf "$tmpdir"
        echo "Could not extract ${url}." >&2
        return 1
    fi
    binary=$(find "$tmpdir" -type f -name "$BINARY" -print -quit)
    if [[ -z "$binary" ]]; then
        rm -rf "$tmpdir"
        echo "Archive did not contain ${BINARY}." >&2
        return 1
    fi
    move_binary "$binary"
    rm -rf "$tmpdir"
}

install_from_cargo() {
    have cargo || { echo "cargo not found. Install Rust from https://rustup.rs" >&2; return 1; }
    echo "Installing via cargo: cargo install --git ${CARGO_GIT_URL} --locked"
    cargo install --git "$CARGO_GIT_URL" --locked --force
    local cargo_bin="${CARGO_HOME:-${HOME}/.cargo}/bin/${BINARY}"
    if [[ "$PREFIX" != "/usr/local/bin" && -x "$cargo_bin" ]]; then
        move_binary "$cargo_bin"
    elif [[ -x "$cargo_bin" ]]; then
        echo "Installed via cargo at ${cargo_bin}. Ensure it is on PATH."
        "$cargo_bin" --version || true
    else
        echo "cargo completed, but ${cargo_bin} was not found; ensure ~/.cargo/bin is on PATH." >&2
    fi
}

if [[ "$USE_CARGO" == "1" ]]; then
    install_from_cargo
    exit $?
fi

if install_from_tarball; then
    exit 0
fi

echo "Falling back to cargo install ..." >&2
install_from_cargo
