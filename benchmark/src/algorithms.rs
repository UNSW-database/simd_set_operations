use setops::intersect::{merge::*, svs::*};
use paste::paste;
use std::hint::black_box;

//
// === FUNCTION COMPOSITION ===
//

macro_rules! twoset_to_kset_u32_fn {
    ($outer_func:ident, $inner_func:ident) => {
        paste! {
            fn [<$outer_func _ $inner_func>](sets: &[&[u32]], out: &mut [u32], buf: &mut [u32]) -> usize {
                $outer_func($inner_func, sets, out, buf)
            }
        }
    };
}

twoset_to_kset_u32_fn!(svs, zipper);
twoset_to_kset_u32_fn!(svs, zipper_branch_optimized);
twoset_to_kset_u32_fn!(svs, zipper_branch_optimized_branchless);
twoset_to_kset_u32_fn!(svs, zipper_branch_loop_optimized);

//
// === LOOKUP MAP ===
//

pub struct AlgorithmInfo {
    pub atype: AlgorithmType,
    pub index: usize,
}

pub enum AlgorithmType {
    TwoSet,
    KSetBuf,
}

pub fn algorithm_info_from_name(algorithm_name: &str) -> Option<AlgorithmInfo> {
    return Some(match algorithm_name {
        "zipper"                             => AlgorithmInfo {atype: AlgorithmType::TwoSet,  index: 0},
        "zipper_branch_optimized"            => AlgorithmInfo {atype: AlgorithmType::TwoSet,  index: 1},
        "zipper_branch_optimized_branchless" => AlgorithmInfo {atype: AlgorithmType::TwoSet,  index: 2},
        "zipper_branch_loop_optimized"       => AlgorithmInfo {atype: AlgorithmType::TwoSet,  index: 3},

        "svs_zipper"                             => AlgorithmInfo {atype: AlgorithmType::KSetBuf, index: 0},
        "svs_zipper_branch_optimized"            => AlgorithmInfo {atype: AlgorithmType::KSetBuf, index: 1},
        "svs_zipper_branch_loop_optimized"       => AlgorithmInfo {atype: AlgorithmType::KSetBuf, index: 2},
        "svs_zipper_branch_optimized_branchless" => AlgorithmInfo {atype: AlgorithmType::KSetBuf, index: 3},

        _ => return None,
    })
}

pub fn twoset_u32(algorithm_index: usize, sets: (&[u32], &[u32]), out: &mut [u32]) -> usize {
    return match algorithm_index {
        0 => black_box(zipper(black_box(sets), black_box(out))),
        1 => black_box(zipper_branch_optimized(black_box(sets), black_box(out))),
        2 => black_box(zipper_branch_optimized_branchless(black_box(sets), black_box(out))),
        3 => black_box(zipper_branch_loop_optimized(black_box(sets), black_box(out))),
        _ => unreachable!(),
    }
}

pub fn kset_buf_u32(algorithm_index: usize, sets: &[&[u32]], out: &mut [u32], buf: &mut [u32]) -> usize {
    return match algorithm_index {
        0 => black_box(svs_zipper(black_box(sets), black_box(out), black_box(buf))),
        1 => black_box(svs_zipper_branch_optimized(black_box(sets), black_box(out), black_box(buf))),
        2 => black_box(svs_zipper_branch_optimized_branchless(black_box(sets), black_box(out), black_box(buf))),
        3 => black_box(svs_zipper_branch_loop_optimized(black_box(sets), black_box(out), black_box(buf))),
        _ => unreachable!(),
    }
}

