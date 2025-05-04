use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::intrinsics::assert_inhabited;
use crate::intersect::small_adaptive;
use crate::visitor::Visitor;

#[cfg(target_feature = "avx2")]
pub fn Gather<T, S, V>(sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug + Into<i32>,
    S: AsRef<[T]>,
    V: Visitor<T>,
{
    let mut vec: Vec<T> = vec![];
    vec.reserve_exact(sets[0].as_ref().len());
    for set in sets[0].as_ref().iter() {
        vec.push(*set);
    }
    GatherRec(&vec, &sets[1..], visitor)
}
#[cfg(target_feature = "avx2")]
pub fn GatherRec<T, S, V>(toCompare: &Vec<T>, sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug + Into<i32>,
    S: AsRef<[T]>,
    V: Visitor<T>,
{
    assert_eq!(size_of::<T>(), size_of::<i32>());
    match sets.len() {
        n if n < 4 => {
            let mut vecs: Vec<Vec<T>> = vec![];
            for set in sets {
                vecs.push(set.as_ref().to_vec())
            }
            vecs.push(toCompare.to_vec());
            small_adaptive(&vecs, visitor);
            return;
        },
        _ => {
            unsafe {
                use std::arch::x86_64::*;
                let intersecting = &sets[0..4];
                let mut writing: std::vec::Vec<T> = vec![];
                let rest = &sets[4..];
                let startingPointers = [sets[0].as_ref().as_ptr(), sets[1].as_ref().as_ptr(), sets[2].as_ref().as_ptr(), sets[3].as_ref().as_ptr()];
                let endingPointers = [sets[0].as_ref().as_ptr().add(sets[0].as_ref().len()),  sets[1].as_ref().as_ptr().add(sets[1].as_ref().len()),
                    sets[2].as_ref().as_ptr().add(sets[2].as_ref().len()), sets[3].as_ref().as_ptr().add(sets[3].as_ref().len())];
                let mut pointers: __m256i = _mm256_loadu_si256(startingPointers.as_ptr() as *mut __m256i);
                let endPointers: __m256i = _mm256_loadu_si256(endingPointers.as_ptr() as *mut __m256i);
                let mut i = 0;
                while (i < toCompare.len()) && _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi32(endPointers, pointers))) == 0xF {
                    let val = toCompare[i];
                    let vals = _mm256_i64gather_epi32::<1>(std::ptr::null(), pointers);
                    let vec_val = _mm_set1_epi32(val.into());
                    if _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(vals, vec_val))) == 0xF {
                        i += 1;
                        pointers = _mm256_add_epi64(pointers, _mm256_set1_epi64x(8));
                        writing.push(val);
                    }
                    let mut lessThan = _mm_cmpgt_epi32(vec_val, vals);
                    lessThan = _mm_and_si128(lessThan, _mm_set1_epi32(1));
                    let zero = _mm_setzero_si128();
                    let low = _mm_unpacklo_epi32(lessThan, zero);
                    let high = _mm_unpackhi_epi32(lessThan,zero); 
                    let toAdd = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high);
                    pointers = _mm256_add_epi64(pointers, toAdd);
                }
                GatherRec(&writing, rest, visitor);
            }
        }
    }
}
