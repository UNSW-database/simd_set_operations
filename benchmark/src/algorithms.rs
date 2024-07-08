use std::fmt::Display;
use concat_idents::concat_idents;
use setops::intersect::{*, merge::*, svs::*};
use phf::phf_map;

//
// === TYPES ===
//

#[derive(Debug)]
pub enum Algorithm {
    TwoSet(TwoSetAlgorithm),
    KSetBuf(KSetAlgorithmBuf),
}

macro_rules! algorithm_struct {
    ($struct_name:ident, $func_name:ident) => {
        concat_idents!(InnerStructName = $struct_name, Inner {
            #[derive(Debug)]
            pub struct $struct_name {
                pub out: InnerStructName,
                pub count: InnerStructName,
            }

            #[derive(Debug)]
            pub struct InnerStructName {
                pub u32: Option<$func_name<u32>>,
                pub i32: Option<$func_name<i32>>,
                pub u64: Option<$func_name<u64>>,
                pub i64: Option<$func_name<i64>>,
            }
        });
    };
}

algorithm_struct!(TwoSetAlgorithm, TwoSetAlgorithmFnGeneric);
algorithm_struct!(KSetAlgorithmBuf, KSetAlgorithmBufFnGeneric);

//
// === FUNCTION COMPOSITION ===
//

macro_rules! twoset_to_kset_generic_fn {
    ($outer_func:ident, $inner_func:ident) => {
        concat_idents!(func_name = $outer_func, _, $inner_func {
            fn func_name<T: Ord + Copy + Display>(sets: &[&[T]], out: &mut [T], buf: &mut [T]) -> usize {
                $outer_func::<T>($inner_func::<T, true>, sets, out, buf)
            }
        });
    };
}

twoset_to_kset_generic_fn!(svs, zipper);
twoset_to_kset_generic_fn!(svs, zipper_branch_optimized);
twoset_to_kset_generic_fn!(svs, zipper_branch_loop_optimized);

//
// === LOOKUP MAP ===
//

macro_rules! twoset_generic {
    ($func_name:ident) => {
        Algorithm::TwoSet(
            TwoSetAlgorithm {
                out: TwoSetAlgorithmInner {
                    u32: Some($func_name::<u32, true>),
                    i32: Some($func_name::<i32, true>),
                    u64: Some($func_name::<u64, true>),
                    i64: Some($func_name::<i64, true>),
                },
                count: TwoSetAlgorithmInner {
                    u32: Some($func_name::<u32, false>),
                    i32: Some($func_name::<i32, false>),
                    u64: Some($func_name::<u64, false>),
                    i64: Some($func_name::<i64, false>),
                },
            }
        )
    }
}

// Twoset to kset approaches only support OUT = true as they must calculate intersections for intermediate steps
macro_rules! twoset_to_kset_buf_generic {
    ($func_name:ident) => {
        Algorithm::KSetBuf(
            KSetAlgorithmBuf {
                out: KSetAlgorithmBufInner {
                    u32: Some($func_name::<u32>),
                    i32: Some($func_name::<i32>),
                    u64: Some($func_name::<u64>),
                    i64: Some($func_name::<i64>),
                },
                count: KSetAlgorithmBufInner {
                    u32: None,
                    i32: None,
                    u64: None,
                    i64: None,
                }
            }
        )
    }
}

pub const ALGORITHMS: phf::Map<&'static str, Algorithm> = phf_map! {
    "zipper" => twoset_generic!(zipper),
    "zipper_branch_optimized" => twoset_generic!(zipper_branch_optimized),
    "zipper_branch_loop_optimized" => twoset_generic!(zipper_branch_loop_optimized),
    "svs_zipper" => twoset_to_kset_buf_generic!(svs_zipper),
    "svs_zipper_branch_optimized" => twoset_to_kset_buf_generic!(svs_zipper_branch_optimized),
    "svs_zipper_branch_loop_optimized" => twoset_to_kset_buf_generic!(svs_zipper_branch_loop_optimized),
};

pub fn get_2set(name: &str) -> &TwoSetAlgorithm {
    match ALGORITHMS.get(name).unwrap() {
        Algorithm::TwoSet(x) => x,
        _ => panic!(),
    }
}

pub fn get_kset_buf(name: &str) -> &KSetAlgorithmBuf {
    match ALGORITHMS.get(name).unwrap() {
        Algorithm::KSetBuf(x) => x,
        _ => panic!(),
    }
}
