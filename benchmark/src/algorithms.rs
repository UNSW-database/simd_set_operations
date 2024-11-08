use paste::paste;
use phf::phf_map;
use setops::intersect::{svs::*, zipper::*, *};

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
    pub fn has_output(&self) -> bool {
        match self {
            Algorithm::KSetBuf(_) => true,
            Algorithm::TwoSet(_)  => true,
            Algorithm::ConstantTimeDummy(_) => false,
        }
    }
}

//
// === FUNCTION COMPOSITION ===
//

macro_rules! twoset_to_kset_generic_fn {
    ($outer_func:ident, $inner_func:ident) => {
        paste! {
            fn [<$outer_func _ $inner_func>]<T: Ord + Copy>(sets: &[&[T]], out: &mut [T], buf: &mut [T]) -> usize {
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

pub fn intersect<T: Ord + Copy>(sets: &[&[T]], out: &mut [T], buf: &mut [T]) -> usize {
    return svs_zipper_ref(sets, out, buf);
}


//
// === TRAITS ===
//

pub trait IntersectionAlgorithmLookup: Sized where Self: 'static + Copy {
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

    fn algorithms_from_names(r_names: &[impl AsRef<str>]) -> Result<Vec<Algorithm<Self>>, String> {
        let mut algorithms = Vec::<Algorithm<Self>>::with_capacity(r_names.len());
        for r_name in r_names {
            let r_name_ref = r_name.as_ref();
            match Self::INTERSECTION_ALGORITHMS.get(r_name_ref) {
                Some(rv) => algorithms.push(*rv),
                None => return Err(format!("Algorithm {} not available for type {}.", r_name_ref, std::any::type_name::<Self>())),
            }
        }
        return Ok(algorithms);
    }
}

// This will run to within a handful of cycles of dummy_counts on most
// architectures, though there are some recent intel architectures where it
// may run twice as fast. This doesn't matter hugely as long as it runs
// consistently.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn constant_time_dummy(dummy_counts: usize) {
    use std::arch::asm;
    unsafe {
        asm!(
            "2:",
            "sub {val}, 1",
            "jne 2b",
            val = in(reg) dummy_counts,
        )
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
        "svs_zipper_bad_branch_asm"       => Algorithm::KSetBuf(svs_zipper_bad_branch_u32),
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

