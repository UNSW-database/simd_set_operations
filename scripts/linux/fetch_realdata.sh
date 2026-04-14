#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS_DIR="${1:-$ROOT_DIR/datasets-linux}"
WEBDOCS_URL="${2:-http://fimi.uantwerpen.be/data/webdocs.dat.gz}"
EXPERIMENT_TOML="${EXPERIMENT_TOML:-$ROOT_DIR/experiment.toml}"

mkdir -p "${DATASETS_DIR}"

echo "[realdata] fetching WebDocs source into ${DATASETS_DIR}"
"${ROOT_DIR}/scripts/realdata/fetch_webdocs.bash" "${DATASETS_DIR}" "${WEBDOCS_URL}"

echo "[realdata] generating benchmark datasets from experiment.toml"
cargo run --release --bin generate -- \
  --experiment "${EXPERIMENT_TOML}" \
  --datasets "${DATASETS_DIR}"

cat <<EOF

[done] WebDocs is ready under:
  ${DATASETS_DIR}/webdocs.dat
  ${DATASETS_DIR}/webdocs.cache
  ${DATASETS_DIR}/webdocs/

You can now run:
  ${ROOT_DIR}/scripts/linux/run_paper_benchmarks.sh realdata
EOF
