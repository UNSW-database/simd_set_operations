set -e
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin=benchmark

OUT=gcp-results
mkdir -p $OUT

for size in l1 l3 mem
do
    sudo ./target/release/benchmark \
	--out=$OUT/tods-compare-lowd-$size.json \
	compare_shuffling_sse_selectivity_${size} \
	compare_broadcast_sse_selectivity_${size} \
	compare_bmiss_selectivity_${size} \
	compare_bmiss_sttni_selectivity_${size} \
	compare_qfilter_selectivity_${size} \
	compare_shuffling_avx2_selectivity_${size} \
	compare_broadcast_avx2_selectivity_${size} \
	compare_shuffling_avx512_selectivity_${size} \
	compare_broadcast_avx512_selectivity_${size} \
	compare_conflict_intersect_selectivity_${size} \
	compare_vp2intersect_emulation_selectivity_${size}
done

for size in l1 l3 mem
do
    sudo ./target/release/benchmark \
	--out=$OUT/tods-compare-highd-${size}.json \
	compare_shuffling_sse_bsr_density_${size} \
	compare_broadcast_sse_bsr_density_${size} \
	compare_qfilter_bsr_density_${size} \
	compare_shuffling_avx2_bsr_density_${size} \
	compare_broadcast_avx2_bsr_density_${size} \
	compare_shuffling_avx512_bsr_density_${size} \
	compare_broadcast_avx512_bsr_density_${size}
done

