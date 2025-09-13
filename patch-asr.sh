#!/usr/bin/env bash
# patch-asr.sh — Patch your existing asr.js in-place to use env vars and /tmp.
# Usage: bash patch-asr.sh /path/to/asr.js
set -euo pipefail

JS="${1:-}"
[[ -n "$JS" && -f "$JS" ]] || { echo "Usage: $0 /path/to/asr.js" >&2; exit 1; }

TMP="$(mktemp)"
cp "$JS" "$TMP"

# Build header
read -r -d "" HEADER <<"HDR"
// --- asr runtime dirs (injected) ---
import os from "os";
import path from "path";
const SECRETS_DIR = process.env.ASR_SECRETS_DIR || path.resolve("secrets");
const TMP_BASE = process.env.ASR_TMPDIR || process.env.TMPDIR || process.env.TMP || process.env.TEMP || os.tmpdir();
// -----------------------------------
HDR

# Insert header if not already present
if ! grep -q "asr runtime dirs (injected)" "$TMP"; then
  if head -1 "$TMP" | grep -q '^#!'; then
    { head -1 "$TMP"; echo "$HEADER"; tail -n +2 "$TMP"; } > "${TMP}.new"
  else
    { echo "$HEADER"; cat "$TMP"; } > "${TMP}.new"
  fi
  mv "${TMP}.new" "$TMP"
fi

# Replace secrets path usages
sed -i -E "s|path\\.resolve\\((['\"])secrets/mistral-asr-key\\.txt\\1\\)|path.join(SECRETS_DIR,'mistral-asr-key.txt')|g" "$TMP"
sed -i -E "s|path\\.resolve\\((['\"])secrets\\1\\)|SECRETS_DIR|g" "$TMP"

# Replace script-relative tmp with TMP_BASE
sed -i -E "s|path\\.join\\(__dirname\\s*,\\s*(['\"])tmp\\1\\s*,|path.join(TMP_BASE,|g" "$TMP"
sed -i -E "s|path\\.join\\(__dirname\\s*,\\s*(['\"])tmp\\1\\)|TMP_BASE|g" "$TMP"

# Save backup and replace original
cp "$JS" "$JS.bak.$(date +%s)"
mv "$TMP" "$JS"

echo "Patched $JS"
echo "Backup saved as $JS.bak.<timestamp>"
