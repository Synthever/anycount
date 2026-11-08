#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  export NVM_DIR="$HOME/.nvm"
  . "$NVM_DIR/nvm.sh"
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "[1/2] Building React frontend (web/dist)..."
cd "$PROTO_DIR/web"
npm ci --prefer-offline || npm install
npm run build

echo "[2/2] Building Rust single-binary release..."
cd "$PROTO_DIR"
cargo build --release
ls -lh "$PROTO_DIR/target/release/anycount-prototype"
