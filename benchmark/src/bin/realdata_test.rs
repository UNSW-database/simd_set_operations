#![feature(portable_simd)]

// On non-x86 targets, provide a stub main to avoid build failures.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {
    eprintln!("realdata_test is only supported on x86/x86_64 targets");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod imp {
    use std::{
        ops::BitAnd,
        path::PathBuf,
        simd::{cmp::*, *},
    };

    use benchmark::{realdata, util};
    use clap::Parser;
    use rand::{distributions::Uniform, thread_rng, Rng};
    use setops::{
        bsr::{BsrVec, Intersect2Bsr},
        intersect::{
            self, fesia::*, run_2set, run_2set_bsr, run_kset, run_svs, Intersect2, IntersectK,
        },
        visitor::VecWriter,
        Set,
    };

    #[derive(Parser)]
    #[command(author, version, about, long_about = None)]
    struct Cli {
        #[arg(default_value = "datasets/", long)]
        datasets: PathBuf,
        #[arg(default_value = "10000", long)]
        test_count: u32,
    }

    type TwoSetAlgorithm = (Intersect2<[i32], VecWriter<i32>>, &'static str);
    type TwoSetBsrAlgorithm = (Intersect2Bsr, &'static str);

    pub fn main() {
        let cli = Cli::parse();

        let real_datasets = [
            "webdocs",
            "twitter",
            "as-skitter",
            "census1881",
            "census-income",
        ];

        for real_dataset in real_datasets {
            if let Err(s) = test_on_dataset(&cli, real_dataset) {
                eprintln!("error: {}", s);
            };
        }
    }

    fn test_on_dataset(cli: &Cli, real_dataset: &str) -> Result<(), String> {
        let all_sets = realdata::load_sets(&cli.datasets, real_dataset)?;

        let min_len = all_sets.iter().map(|s| s.len()).min().unwrap();
        let max_len = all_sets.iter().map(|s| s.len()).max().unwrap();

        let total_len: usize = all_sets.iter().map(|s| s.len()).sum();
        let avg_len = total_len as f64 / all_sets.len() as f64;

        println!(
            "{}: set lengths: avg {:.2}, min {}, max {}",
            real_dataset, avg_len, min_len, max_len
        );

        let mut twoset_array_algorithms: Vec<TwoSetAlgorithm> = TWOSET.into();
        twoset_array_algorithms.extend_from_slice(&TWOSET_SSE);
        twoset_array_algorithms.extend_from_slice(&TWOSET_AVX2);
        twoset_array_algorithms.extend_from_slice(&TWOSET_AVX512);

        let mut twoset_bsr_algorithms: Vec<TwoSetBsrAlgorithm> = TWOSET_BSR.into();
        twoset_bsr_algorithms.extend_from_slice(&TWOSET_BSR_SSE);
        twoset_bsr_algorithms.extend_from_slice(&TWOSET_BSR_AVX2);
        twoset_bsr_algorithms.extend_from_slice(&TWOSET_BSR_AVX512);

        println!("2-set:");
        run_twoset_tests(
            &all_sets,
            cli.test_count,
            &twoset_array_algorithms,
            &twoset_bsr_algorithms,
        );

        println!("k-set svs:");
        run_kset_tests(&all_sets, cli.test_count);
        Ok(())
    }

    fn run_twoset_tests(
        sets: &Vec<Vec<i32>>,
        test_count: u32,
        algorithms: &[TwoSetAlgorithm],
        bsr_algorithms: &[TwoSetBsrAlgorithm],
    ) {
        run_twoset_array_tests(sets, test_count, algorithms);
        run_twoset_bsr_tests(sets, test_count, bsr_algorithms);
    }

    fn run_twoset_array_tests(sets: &Vec<Vec<i32>>, test_count: u32, algorithms: &[TwoSetAlgorithm]) {
        let mut rng = thread_rng();
        let distribution = Uniform::new(0, sets.len());

        for (intersect, name) in algorithms {
            println!("  {}", name);
            for _ in 0..test_count {
                let i = rng.sample(distribution);
                let j = rng.sample(distribution);
                let left = &sets[i];
                let right = &sets[j];
                let expected = run_2set(left, right, intersect::naive_merge);
                let actual = run_2set(left, right, *intersect);
                assert!(actual == expected);
            }
        }
    }

    fn run_twoset_bsr_tests(
        sets: &Vec<Vec<i32>>,
        test_count: u32,
        algorithms: &[TwoSetBsrAlgorithm],
    ) {
        let sets: Vec<BsrVec> = sets.iter().map(|s| BsrVec::from_sorted(s.as_slice())).collect();

        let mut rng = thread_rng();
        let distribution = Uniform::new(0, sets.len());

        for (intersect, name) in algorithms {
            println!("  {}", name);
            for _ in 0..test_count {
                let i = rng.sample(distribution);
                let j = rng.sample(distribution);
                let left = &sets[i];
                let right = &sets[j];

                let expected =
                    run_2set_bsr(left.bsr_ref(), right.bsr_ref(), intersect::branchless_merge_bsr);

                let mut actual = BsrVec::new();
                intersect(left.bsr_ref(), right.bsr_ref(), &mut actual);

                if !expected.equal_set(actual.bsr_ref()) {
                    panic!("expected {:?}\nactual {:?}", expected, actual);
                }
            }
        }
    }

    fn run_kset_tests(sets: &Vec<Vec<i32>>, test_count: u32) {
        let mut rng = thread_rng();
        let distribution = Uniform::new(0, sets.len());

        for k in 2..=4 {
            let mut kset: Vec<&Vec<i32>> = Vec::with_capacity(k);

            for _ in 0..test_count {
                let small_idx = rng.sample(distribution);
                kset.push(&sets[small_idx]);
                for _ in 0..k - 1 {
                    let large_idx = rng.sample(distribution);
                    kset.push(&sets[large_idx]);
                }

                let expected = run_svs(kset.as_slice(), intersect::branchless_merge);
                let actual = run_kset(kset.as_slice(), intersect::small_adaptive);
                let actual_sorted = util::sorted(actual);
                assert!(expected == actual_sorted);
                kset.clear();
            }
        }
    }

    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    fn run_fesia_realdata(
        sets: &[Vec<i32>],
        test_count: u32,
        twoset: &[(Intersect2<[i32], VecWriter<i32>>, &'static str)],
    ) {
        let mut rng = thread_rng();
        let distribution = Uniform::new(0, sets.len());

        let fesia_method = FesiaTwoSetMethod {
            hash_scale: 0.01,
            simd_type: SimdType::Sse,
            intersect_method: SegmentIntersectSse::Intersect,
            match_method: SegmentMatch::Simd,
        };

        let fesia_b_method = FesiaTwoSetMethod {
            hash_scale: 0.01,
            simd_type: SimdType::Sse,
            intersect_method: SegmentIntersectSse::Intersect,
            match_method: SegmentMatch::BitAnd,
        };

        let mut i = 0;

        while i < test_count {
            i += 1;
            let small = rng.sample(distribution);
            let large = rng.sample(distribution);
            let small_set = &sets[small];
            let large_set = &sets[large];

            let mut writer: VecWriter<i32> = VecWriter::with_capacity(small_set.len());
            let mut counter = 0;

            let fesia_b = FesiaIntersect::from_array(&fesia_b_method, small_set, large_set);
            let fesia_a = FesiaIntersect::from_array(&fesia_method, large_set, small_set);
            fesia_a.intersect::<VecWriter<i32>, SegmentIntersectSse>(&fesia_b, &mut writer);

            let expected = run_2set(
                small_set.as_slice(),
                large_set.as_slice(),
                intersect::branchless_merge,
            );

            let counter_method = FesiaTwoSetMethod {
                hash_scale: 0.01,
                simd_type: SimdType::Sse,
                intersect_method: SegmentIntersectSse::Intersect,
                match_method: SegmentMatch::Count,
            };
            let fesia_a = FesiaIntersect::from_array(&fesia_method, large_set, small_set);
            let fesia_b = FesiaIntersect::from_array(&counter_method, small_set, large_set);
            fesia_a.intersect::<VecWriter<i32>, SegmentIntersectSse>(&fesia_b, &mut counter);

            writer.visit(counter as i32);

            twoset.iter().for_each(|(f, name)| {
                if !intersection_matches(small_set.as_slice(), large_set.as_slice(), *f, &writer) {
                    println!("failed for {}", name);
                }
            });
        }
    }

    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    fn intersection_matches(
        set_a: &[i32],
        set_b: &[i32],
        intersect: Intersect2<[i32], VecWriter<i32>>,
        intersect_expected: &VecWriter<i32>,
    ) -> bool {
        let mut intersection_actual = Vec::new();
        intersect(set_a, set_b, &mut intersection_actual);
        intersection_actual.sort();
        let intersect_expected = intersect_expected.to_owned();
        intersection_actual == util::sorted(intersect_expected)
    }

    const TWOSET: [TwoSetAlgorithm; 3] = [
        (intersect::naive_merge, "naive_merge"),
        (intersect::branchless_merge, "branchless_merge"),
        (intersect::baezayates, "baezayates"),
    ];

    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    const TWOSET_SSE: [TwoSetAlgorithm; 7] = [
        (intersect::shuffling_sse, "shuffling_sse"),
        (intersect::broadcast_sse, "broadcast_sse"),
        (intersect::galloping_sse, "galloping_sse"),
        (intersect::bmiss, "bmiss"),
        (intersect::bmiss_sttni, "bmiss_sttni"),
        (intersect::qfilter, "qfilter"),
        (intersect::qfilter_v1, "qfilter_v1"),
    ];
    #[cfg(not(all(feature = "simd", target_feature = "ssse3")))]
    const TWOSET_SSE: [TwoSetAlgorithm; 0] = [];

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    const TWOSET_AVX2: [TwoSetAlgorithm; 6] = [
        (intersect::shuffling_avx2, "shuffling_avx2"),
        (intersect::broadcast_avx2, "broadcast_avx2"),
        (intersect::galloping_avx2, "galloping_avx2"),
        (intersect::lbk_v1x8_avx2, "lbk_v1x8_avx2"),
        (intersect::lbk_v1x16_avx2, "lbk_v1x16_avx2"),
        (intersect::lbk_v3_avx2, "lbk_v3_avx2"),
    ];
    #[cfg(not(all(feature = "simd", target_feature = "avx2")))]
    const TWOSET_AVX2: [TwoSetAlgorithm; 0] = [];

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    const TWOSET_AVX512: [TwoSetAlgorithm; 6] = [
        (intersect::shuffling_avx512, "shuffling_avx512"),
        (intersect::broadcast_avx512, "broadcast_avx512"),
        (intersect::galloping_avx512, "galloping_avx512"),
        (intersect::lbk_v1x32_avx512, "lbk_v1x32_avx512"),
        (intersect::lbk_v1x16_avx512, "lbk_v1x16_avx512"),
        (intersect::lbk_v3_avx512, "lbk_v3_avx512"),
    ];
    #[cfg(not(all(feature = "simd", target_feature = "avx512f")))]
    const TWOSET_AVX512: [TwoSetAlgorithm; 0] = [];

    const TWOSET_BSR: [TwoSetBsrAlgorithm; 1] =
        [(intersect::branchless_merge_bsr, "branchless_merge_bsr")];

    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    const TWOSET_BSR_SSE: [TwoSetBsrAlgorithm; 4] = [
        (intersect::shuffling_sse_bsr, "shuffling_sse_bsr"),
        (intersect::broadcast_sse_bsr, "broadcast_sse_bsr"),
        (intersect::galloping_sse_bsr, "galloping_sse_bsr"),
        (intersect::qfilter_bsr, "qfilter_bsr"),
    ];
    #[cfg(not(all(feature = "simd", target_feature = "ssse3")))]
    const TWOSET_BSR_SSE: [TwoSetBsrAlgorithm; 0] = [];

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    const TWOSET_BSR_AVX2: [TwoSetBsrAlgorithm; 3] = [
        (intersect::shuffling_avx2_bsr, "shuffling_avx2_bsr"),
        (intersect::broadcast_avx2_bsr, "broadcast_avx2_bsr"),
        (intersect::galloping_avx2_bsr, "galloping_avx2_bsr"),
    ];
    #[cfg(not(all(feature = "simd", target_feature = "avx2")))]
    const TWOSET_BSR_AVX2: [TwoSetBsrAlgorithm; 0] = [];

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    const TWOSET_BSR_AVX512: [TwoSetBsrAlgorithm; 3] = [
        (intersect::shuffling_avx512_bsr, "shuffling_avx512_bsr"),
        (intersect::broadcast_avx512_bsr, "broadcast_avx512_bsr"),
        (intersect::galloping_avx512_bsr, "galloping_avx512_bsr"),
    ];
    #[cfg(not(all(feature = "simd", target_feature = "avx512f")))]
    const TWOSET_BSR_AVX512: [TwoSetBsrAlgorithm; 0] = [];
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    imp::main();
}
