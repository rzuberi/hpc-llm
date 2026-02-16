#!/usr/bin/env bash
set -euo pipefail

TARGET_BIN="$HOME/.local/bin/ollama"
LOCAL_PREFIX="$HOME/.local"

pick_default_models_dir() {
  local user_name group_name
  user_name="$(whoami)"
  group_name="$(id -gn 2>/dev/null || true)"
  local base
  local candidate

  if [ -n "${SCRATCH:-}" ]; then
    candidate="$SCRATCH/.ollama/models"
    if mkdir -p "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  fi

  for base in /mnt/scratchc /scratchc /mnt/scratch /scratch; do
    [ -d "$base" ] || continue

    for candidate in \
      "$base/$user_name/.ollama/models" \
      "$base/$group_name/$user_name/.ollama/models"; do
      if mkdir -p "$candidate" >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    done
  done

  candidate="$HOME/.ollama/models"
  mkdir -p "$candidate"
  echo "$candidate"
}

if command -v ollama >/dev/null 2>&1; then
  echo "ollama already available at: $(command -v ollama)"
elif [ -x "$TARGET_BIN" ]; then
  echo "ollama already installed at: $TARGET_BIN"
else
  mkdir -p "$LOCAL_PREFIX"

  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64)
      url="https://ollama.com/download/ollama-linux-amd64.tar.zst"
      ;;
    aarch64|arm64)
      url="https://ollama.com/download/ollama-linux-arm64.tar.zst"
      ;;
    *)
      echo "ERROR: unsupported architecture '$arch'" >&2
      exit 1
      ;;
  esac

  if ! command -v zstd >/dev/null 2>&1; then
    echo "ERROR: zstd is required to extract the Ollama tar.zst archive." >&2
    exit 1
  fi

  tmpdir="$(mktemp -d)"
  archive="$tmpdir/ollama.tar.zst"
  cleanup() {
    rm -rf "$tmpdir"
  }
  trap cleanup EXIT

  echo "Downloading ollama from $url"
  curl -fsSL "$url" -o "$archive"

  if tar --help 2>/dev/null | grep -q -- '--zstd'; then
    tar --zstd -xf "$archive" -C "$LOCAL_PREFIX"
  else
    unzstd -c "$archive" | tar -xf - -C "$LOCAL_PREFIX"
  fi

  if [ ! -x "$TARGET_BIN" ]; then
    echo "ERROR: expected binary not found after extraction: $TARGET_BIN" >&2
    exit 1
  fi
  chmod +x "$TARGET_BIN"
  echo "Installed ollama to $TARGET_BIN"

  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *)
      echo "NOTE: add '$HOME/.local/bin' to PATH to use ollama globally."
      ;;
  esac
  echo "If needed for shared libs, export: LD_LIBRARY_PATH=\"$HOME/.local/lib/ollama:\${LD_LIBRARY_PATH:-}\""
fi

MODELS_DIR="${OLLAMA_MODELS:-$(pick_default_models_dir)}"
if ! mkdir -p "$MODELS_DIR" >/dev/null 2>&1; then
  echo "WARN: could not create OLLAMA_MODELS path '$MODELS_DIR', falling back to $HOME/.ollama/models" >&2
  MODELS_DIR="$HOME/.ollama/models"
  mkdir -p "$MODELS_DIR"
fi

echo "Model directory: $MODELS_DIR"
if [ -z "${OLLAMA_MODELS:-}" ]; then
  echo "To pin this location, export: OLLAMA_MODELS=$MODELS_DIR"
fi
