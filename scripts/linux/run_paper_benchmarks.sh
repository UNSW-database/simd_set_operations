#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_TOML="${EXPERIMENT_TOML:-$ROOT_DIR/experiment.toml}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT_DIR/datasets-linux}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results/linux}"
THREADS="${RAYON_NUM_THREADS:-16}"
MODE="${1:-all}"

export RAYON_NUM_THREADS="$THREADS"

HOST_TAG="$(hostname -s 2>/dev/null || hostname)"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RESULTS_ROOT}/${HOST_TAG}-${STAMP}"
mkdir -p "$RUN_DIR"

has_cpu_flag() {
  local flag="$1"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | tr '[:upper:]' '[:lower:]' | grep -qE "(flags:|features:).*(^| )${flag}( |$)"
  else
    grep -qiE "(^| )${flag}( |$)" /proc/cpuinfo
  fi
}

HAS_AVX2=0
HAS_AVX512=0
if has_cpu_flag "avx2"; then
  HAS_AVX2=1
fi
if has_cpu_flag "avx512f"; then
  HAS_AVX512=1
fi

write_machine_info() {
  {
    echo "timestamp=${STAMP}"
    echo "host=${HOST_TAG}"
    echo "root_dir=${ROOT_DIR}"
    echo "datasets_dir=${DATASETS_DIR}"
    echo "results_dir=${RUN_DIR}"
    echo "rayon_threads=${RAYON_NUM_THREADS}"
    echo "has_avx2=${HAS_AVX2}"
    echo "has_avx512=${HAS_AVX512}"
  } > "${RUN_DIR}/run-meta.txt"

  uname -a > "${RUN_DIR}/uname.txt" 2>&1 || true
  if command -v lscpu >/dev/null 2>&1; then
    lscpu > "${RUN_DIR}/lscpu.txt" 2>&1 || true
  fi
  if command -v rustc >/dev/null 2>&1; then
    rustc -Vv > "${RUN_DIR}/rustc-version.txt" 2>&1 || true
  fi
  if command -v cargo >/dev/null 2>&1; then
    cargo -V > "${RUN_DIR}/cargo-version.txt" 2>&1 || true
  fi
}

ensure_build() {
  cargo build --release --bin generate --bin benchmark
}

ensure_datasets() {
  if [[ ! -d "${DATASETS_DIR}/2set_vary_selectivity" ]]; then
    echo "[setup] datasets not found under ${DATASETS_DIR}, generating once..."
    cargo run --release --bin generate -- \
      --experiment "${EXPERIMENT_TOML}" \
      --datasets "${DATASETS_DIR}"
  else
    echo "[setup] datasets already exist under ${DATASETS_DIR}, skipping generation."
  fi
}

run_group() {
  local name="$1"
  local count_only="$2"
  shift 2
  local experiments=("$@")
  local out="${RUN_DIR}/${name}.json"

  if [[ "${#experiments[@]}" -eq 0 ]]; then
    echo "[skip] ${name}: no experiments selected for this machine."
    return 0
  fi

  echo "[run] ${name}"
  echo "      output: ${out}"
  echo "      experiments: ${experiments[*]}"

  local args=(
    run --release --bin benchmark -- 
    --experiment "${EXPERIMENT_TOML}"
    --datasets "${DATASETS_DIR}"
    --out "${out}"
    --stage-stats-mode separate
    --tag "${HOST_TAG}-${name}"
  )

  if [[ "${count_only}" == "1" ]]; then
    args+=(--count_only)
  fi

  cargo "${args[@]}" "${experiments[@]}"
}

setup_only() {
  write_machine_info
  ensure_build
  ensure_datasets
}

run_verify() {
  cargo test -p setops
  cargo test -p benchmark
}

run_low_density() {
  local exps=("2set_vary_selectivity_tods_sse")
  if [[ "${HAS_AVX2}" == "1" ]]; then
    exps+=("2set_vary_selectivity_tods_avx2")
  fi
  if [[ "${HAS_AVX512}" == "1" ]]; then
    exps+=("2set_vary_selectivity_tods_avx512")
  fi
  run_group "low-density" 0 "${exps[@]}"
}

run_stage1() {
  local exps=("2set_vary_skew_tods_sse")
  if [[ "${HAS_AVX2}" == "1" ]]; then
    exps+=("2set_vary_skew_tods_avx2")
  fi
  if [[ "${HAS_AVX512}" == "1" ]]; then
    exps+=("2set_vary_skew_tods_avx512")
  fi
  run_group "stage1-skew" 0 "${exps[@]}"
}

run_stage2() {
  local exps=(
    "compare_bmiss"
    "compare_bmiss_sttni"
    "compare_qfilter"
  )
  if [[ "${HAS_AVX2}" == "1" ]]; then
    exps+=(
      "lbk_prefilter_ablation"
      "lbk_prefilter_ablation_skew"
    )
  fi
  run_group "stage2-filtering" 0 "${exps[@]}"
}

run_stage3() {
  local exps=(
    "compare_shuffling_sse"
    "compare_broadcast_sse"
  )
  if [[ "${HAS_AVX2}" == "1" ]]; then
    exps+=(
      "compare_shuffling_avx2"
      "compare_broadcast_avx2"
    )
  fi
  if [[ "${HAS_AVX512}" == "1" ]]; then
    exps+=(
      "compare_shuffling_avx512"
      "compare_broadcast_avx512"
      "compare_vp2intersect_emulation"
      "compare_conflict_intersect"
    )
  fi
  run_group "stage3-kernels" 0 "${exps[@]}"
}

run_stage3_count_only() {
  local exps=(
    "compare_shuffling_sse"
    "compare_broadcast_sse"
  )
  if [[ "${HAS_AVX2}" == "1" ]]; then
    exps+=(
      "compare_shuffling_avx2"
      "compare_broadcast_avx2"
    )
  fi
  if [[ "${HAS_AVX512}" == "1" ]]; then
    exps+=(
      "compare_shuffling_avx512"
      "compare_broadcast_avx512"
      "compare_vp2intersect_emulation"
      "compare_conflict_intersect"
    )
  fi
  run_group "stage3-kernels-count-only" 1 "${exps[@]}"
}

run_boundary() {
  local exps=()
  if [[ "${HAS_AVX2}" == "1" ]]; then
    exps+=(
      "2set_vary_density_tods_avx2"
      "2set_vary_density_tods_roaring"
      "2set_vary_density_tods_fesia"
    )
  fi
  if [[ "${HAS_AVX512}" == "1" ]]; then
    exps+=("2set_vary_density_tods_avx512")
  fi
  run_group "boundary-density" 0 "${exps[@]}"
}

run_realdata() {
  local exps=("webdocs_tods_sse")
  if [[ "${HAS_AVX2}" == "1" ]]; then
    exps+=("webdocs_tods_avx2" "webdocs_tods_others")
  fi
  if [[ "${HAS_AVX512}" == "1" ]]; then
    exps+=("webdocs_tods_avx512")
  fi
  run_group "boundary-realdata" 0 "${exps[@]}"
}

print_summary() {
  cat <<EOF

[done] results are under:
  ${RUN_DIR}

Important files to send back:
  ${RUN_DIR}/*.json
  ${RUN_DIR}/run-meta.txt
  ${RUN_DIR}/uname.txt
  ${RUN_DIR}/lscpu.txt
  ${RUN_DIR}/rustc-version.txt

If you want to compress them on Linux:
  tar -czf ${HOST_TAG}-${STAMP}-paper-results.tar.gz -C "${RESULTS_ROOT}" "$(basename "${RUN_DIR}")"
EOF
}

case "${MODE}" in
  setup)
    setup_only
    ;;
  verify)
    run_verify
    ;;
  lowd)
    setup_only
    run_low_density
    ;;
  stage1)
    setup_only
    run_stage1
    ;;
  stage2)
    setup_only
    run_stage2
    ;;
  stage3)
    setup_only
    run_stage3
    ;;
  stage3-count)
    setup_only
    run_stage3_count_only
    ;;
  boundary)
    setup_only
    run_boundary
    ;;
  realdata)
    setup_only
    run_realdata
    ;;
  all)
    setup_only
    run_low_density
    run_stage1
    run_stage2
    run_stage3
    run_stage3_count_only
    run_boundary
    run_realdata
    ;;
  *)
    echo "usage: $0 {setup|verify|lowd|stage1|stage2|stage3|stage3-count|boundary|realdata|all}" >&2
    exit 1
    ;;
esac

print_summary
