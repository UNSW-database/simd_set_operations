use paste::paste;
use phf::phf_map;
use setops::intersect::{svs::*, zipper::*, *};
use std::fmt::Display;

//
// === TYPES ===
//

#[derive(Debug, Clone, Copy)]
pub enum Algorithm<T> {
    TwoSet(TwoSetAlgorithmFnGeneric<T>),
    KSetBuf(KSetAlgorithmBufFnGeneric<T>),
    ConstantTimeDummy(usize),
}

impl<T> Algorithm<T> {
    pub fn is_valid(&self, set_count: usize) -> bool {
        match &self {
            Algorithm::KSetBuf(_) => true,
            Algorithm::TwoSet(_) => set_count == 2,
            Algorithm::ConstantTimeDummy(_) => true,
        }
    }
}

//
// === FUNCTION COMPOSITION ===
//

macro_rules! twoset_to_kset_generic_fn {
    ($outer_func:ident, $inner_func:ident) => {
        paste! {
            fn [<$outer_func _ $inner_func>]<T: Ord + Copy + Display>(sets: &[&[T]], out: &mut [T], buf: &mut [T]) -> usize {
                $outer_func::<T>($inner_func::<T, true>, sets, out, buf)
            }
        }
    };
}

twoset_to_kset_generic_fn!(svs, zipper_ref);
twoset_to_kset_generic_fn!(svs, zipper_branch_optimized_ref);
twoset_to_kset_generic_fn!(svs, zipper_loop_optimized_ref);

macro_rules! twoset_to_kset_fn_typed {
    ($outer_func:ident, $type:ident, $inner_func:ident) => {
        paste! {
            fn [<$outer_func _ $inner_func>](sets: &[&[$type]], out: &mut [$type], buf: &mut [$type]) -> usize {
                $outer_func::<$type>($inner_func, sets, out, buf)
            }
        }
    };
}

twoset_to_kset_fn_typed!(svs, u32, zipper_u32);
twoset_to_kset_fn_typed!(svs, u32, zipper_branch_optimized_u32);
twoset_to_kset_fn_typed!(svs, u32, zipper_branchless_u32);
twoset_to_kset_fn_typed!(svs, u32, zipper_loop_optimized_u32);
twoset_to_kset_fn_typed!(svs, u32, zipper_noindex_u32);
twoset_to_kset_fn_typed!(svs, u32, zipper_bad_branch_u32);


//
// === TRAITS ===
//

pub trait IntersectionAlgorithmLookup: Sized where Self: 'static {
    const INTERSECTION_ALGORITHMS: phf::Map<&'static str, Algorithm<Self>>;

    fn get_2set(name: &str) -> &TwoSetAlgorithmFnGeneric<Self> {
        match Self::INTERSECTION_ALGORITHMS.get(name).unwrap() {
            Algorithm::TwoSet(x) => x,
            _ => panic!(),
        }
    }

    fn get_kset_buf(name: &str) -> &KSetAlgorithmBufFnGeneric<Self> {
        match Self::INTERSECTION_ALGORITHMS.get(name).unwrap() {
            Algorithm::KSetBuf(x) => x,
            _ => panic!(),
        }
    }
}


//                     //
// === TRAIT IMPLS === //
//                     //

impl IntersectionAlgorithmLookup for u32 {
    const INTERSECTION_ALGORITHMS: phf::Map<&'static str, Algorithm<Self>> = phf_map! {
        "zipper_asm"                  => Algorithm::TwoSet(zipper_u32),
        "zipper_branch_optimized_asm" => Algorithm::TwoSet(zipper_branch_optimized_u32),
        "zipper_branchless_asm"       => Algorithm::TwoSet(zipper_branchless_u32),
        "zipper_loop_optimized_asm"   => Algorithm::TwoSet(zipper_loop_optimized_u32),
        "zipper_noindex_asm"          => Algorithm::TwoSet(zipper_noindex_u32),
        "zipper_bad_branch_asm"       => Algorithm::TwoSet(zipper_bad_branch_u32),
        "zipper_ref"                  => Algorithm::TwoSet(zipper_ref::<u32, true>),
        "zipper_branch_optimized_ref" => Algorithm::TwoSet(zipper_branch_optimized_ref::<u32, true>),
        "zipper_loop_optimized_ref"   => Algorithm::TwoSet(zipper_loop_optimized_ref::<u32, true>),
        "svs_zipper_asm"                  => Algorithm::KSetBuf(svs_zipper_u32),
        "svs_zipper_branch_optimized_asm" => Algorithm::KSetBuf(svs_zipper_branch_optimized_u32),
        "svs_zipper_branchless_asm"       => Algorithm::KSetBuf(svs_zipper_branchless_u32),
        "svs_zipper_loop_optimized_asm"   => Algorithm::KSetBuf(svs_zipper_loop_optimized_u32),
        "svs_zipper_noindex_asm"          => Algorithm::KSetBuf(svs_zipper_noindex_u32),
        "svs_zipper_bad_branch_asm"           => Algorithm::KSetBuf(svs_zipper_bad_branch_u32),
        "svs_zipper_ref"                  => Algorithm::KSetBuf(svs_zipper_ref::<u32>),
        "svs_zipper_branch_optimized_ref" => Algorithm::KSetBuf(svs_zipper_branch_optimized_ref::<u32>),
        "svs_zipper_loop_optimized_ref"   => Algorithm::KSetBuf(svs_zipper_loop_optimized_ref::<u32>),
    };
}

impl IntersectionAlgorithmLookup for i32 {
    const INTERSECTION_ALGORITHMS: phf::Map<&'static str, Algorithm<Self>> = phf_map! {
        "zipper_ref"                  => Algorithm::TwoSet(zipper_ref::<i32, true>),
        "zipper_branch_optimized_ref" => Algorithm::TwoSet(zipper_branch_optimized_ref::<i32, true>),
        "zipper_loop_optimized_ref"   => Algorithm::TwoSet(zipper_loop_optimized_ref::<i32, true>),
        "svs_zipper_ref"                  => Algorithm::KSetBuf(svs_zipper_ref::<i32>),
        "svs_zipper_branch_optimized_ref" => Algorithm::KSetBuf(svs_zipper_branch_optimized_ref::<i32>),
        "svs_zipper_loop_optimized_ref"   => Algorithm::KSetBuf(svs_zipper_loop_optimized_ref::<i32>),
    };
}

impl IntersectionAlgorithmLookup for u64 {
    const INTERSECTION_ALGORITHMS: phf::Map<&'static str, Algorithm<Self>> = phf_map! {
        "zipper_ref"                  => Algorithm::TwoSet(zipper_ref::<u64, true>),
        "zipper_branch_optimized_ref" => Algorithm::TwoSet(zipper_branch_optimized_ref::<u64, true>),
        "zipper_loop_optimized_ref"   => Algorithm::TwoSet(zipper_loop_optimized_ref::<u64, true>),
        "svs_zipper_ref"                  => Algorithm::KSetBuf(svs_zipper_ref::<u64>),
        "svs_zipper_branch_optimized_ref" => Algorithm::KSetBuf(svs_zipper_branch_optimized_ref::<u64>),
        "svs_zipper_loop_optimized_ref"   => Algorithm::KSetBuf(svs_zipper_loop_optimized_ref::<u64>),
    };
}

impl IntersectionAlgorithmLookup for i64 {
    const INTERSECTION_ALGORITHMS: phf::Map<&'static str, Algorithm<Self>> = phf_map! {
        "zipper_ref"                  => Algorithm::TwoSet(zipper_ref::<i64, true>),
        "zipper_branch_optimized_ref" => Algorithm::TwoSet(zipper_branch_optimized_ref::<i64, true>),
        "zipper_loop_optimized_ref"   => Algorithm::TwoSet(zipper_loop_optimized_ref::<i64, true>),
        "svs_zipper_ref"                  => Algorithm::KSetBuf(svs_zipper_ref::<i64>),
        "svs_zipper_branch_optimized_ref" => Algorithm::KSetBuf(svs_zipper_branch_optimized_ref::<i64>),
        "svs_zipper_loop_optimized_ref"   => Algorithm::KSetBuf(svs_zipper_loop_optimized_ref::<i64>),
    };
}

