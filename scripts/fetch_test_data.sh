#!/usr/bin/env bash
# Fetch Databento's public 4-record MBO sample (Apache-2.0, databento-python test suite), pinned
# to one upstream commit and checked against its SHA-256. The decode test skips without it.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=130cb8a576a7a039f5e14528f13d2da09dcf0725
SHA256=0218cbb0ac4b2fb885cba798b94b06f404992ea8c20457c77d1d04bdeb91af69
OUT=tests/data/databento_sample_GLBX_mbo.dbn.zst
mkdir -p tests/data
curl -fsSL -o "$OUT" \
  "https://raw.githubusercontent.com/databento/databento-python/$COMMIT/tests/data/GLBX.MDP3/test_data.mbo.dbn.zst"
if command -v sha256sum >/dev/null; then got=$(sha256sum "$OUT" | cut -d' ' -f1); else got=$(shasum -a 256 "$OUT" | cut -d' ' -f1); fi
if [ "$got" != "$SHA256" ]; then
  rm -f "$OUT"
  echo "SHA-256 mismatch for $OUT: got $got" >&2
  exit 1
fi
echo "fetched $OUT ($SHA256)"
