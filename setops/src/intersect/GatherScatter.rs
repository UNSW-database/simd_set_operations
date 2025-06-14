use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::intrinsics::assert_inhabited;
use crate::intersect::{broadcast_avx2, small_adaptive};
use crate::KSetInput::KSetInput;
use crate::visitor;
use crate::visitor::{VecWriter, Visitor};
#[cfg(target_feature = "avx2")]
pub fn Gather<V>(initial: &Vec<i32>, additional: &KSetInput, visitor: &mut V)
where
    V: Visitor<i32> + visitor::SimdVisitor8,
{
    GatherRec(initial, additional, 0, visitor);
}
pub fn GatherRec<V>(initial: &Vec<i32>, additional: &KSetInput, index: u32, visitor: &mut V)
where
    V: Visitor<i32> + visitor::SimdVisitor8,
{
    match additional.getSize() - index  {
        0 => {
            broadcast_avx2(initial, initial, visitor);
        }
        1 => {
            broadcast_avx2(initial,additional.getSlice(index), visitor);
        }
        n if n < 8 => {
            let mut writer: visitor::VecWriter<i32> = VecWriter::with_capacity(initial.len());
            broadcast_avx2(initial, additional.getSlice(index), &mut writer);
            let newInit: Vec<i32> = writer.into();
            GatherRec(&newInit, additional, index + 1, visitor);
        }
        _ => {
            unsafe {
                use std::arch::x86_64::*;
                let mut pointersArr = vec![0; 8];
                let mut pointersArrEnd = vec![0; 8];
                for j in 0..8u32 {
                    pointersArr[j as usize] = additional.getRange(index + j).start;
                    pointersArrEnd[j as usize] = additional.getRange(index + j).end;
                }
                let mut indexes = _mm256_loadu_si256(pointersArr.as_ptr().cast());
                let ends = _mm256_loadu_si256(pointersArrEnd.as_ptr().cast());

                let mut i = 0;
                let offset = additional.getIntial();
                let end = initial.len() as u32;
                let mut resultsVec: Vec<i32> = vec![];
                while (i < end && _mm256_movemask_epi8(_mm256_cmpgt_epi32(ends, indexes)) == 0xFFFFFFFFu32 as i32) {
                    let val = _mm256_set1_epi32(*initial.get_unchecked(i as usize));
                    let vals = _mm256_i32gather_epi32::<4>(offset, indexes);
                    if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(val, vals)) == 0xFFFFFFFFu32 as i32) {
                        resultsVec.push(*initial.get_unchecked(i as usize));
                        i += 1;
                        indexes = _mm256_add_epi32(indexes, _mm256_set1_epi32(1));
                        continue;
                    } 
                    if _mm256_movemask_epi8(_mm256_cmpgt_epi32(vals, val)) != 0 {
                        i+=1;
                        continue;
                    }
                    let cmp = _mm256_and_si256(_mm256_cmpgt_epi32(val, vals),  _mm256_set1_epi32(1));
                    indexes = _mm256_add_epi32(indexes, cmp);
                }
                GatherRec(&resultsVec, additional, index + 8, visitor);
            }
        }
    }
}
#[cfg(target_feature = "avx2")]
pub fn Gather0<T, S, V>(sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug + Into<i32>,
    S: AsRef<[T]>,
    V: Visitor<T> + visitor::SimdVisitor8,
{
    let mut vec: Vec<T> = vec![];
    vec.reserve_exact(sets[0].as_ref().len());
    for set in sets[0].as_ref().iter() {
        vec.push(*set);
    }
    GatherRec0(&vec, &sets[1..], visitor)
}
#[cfg(target_feature = "avx2")]
pub fn GatherRec0<T, S, V>(toCompare: &Vec<T>, sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug + Into<i32>,
    S: AsRef<[T]>,
    V: Visitor<T> + visitor::SimdVisitor8,
{
    assert_eq!(size_of::<T>(), size_of::<i32>());
    match sets.len() {
        0 => {
            broadcast_avx2(toCompare, toCompare, visitor);
        }
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
                        continue;
                    }
                    let mut lessThan = _mm_cmpgt_epi32(vec_val, vals);
                    lessThan = _mm_and_si128(lessThan, _mm_set1_epi32(1));
                    if _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(vals, vec_val))) != 0 {
                        i += 1;
                        continue;
                    }
                    let zero = _mm_setzero_si128();
                    let low = _mm_unpacklo_epi32(lessThan, zero);
                    let high = _mm_unpackhi_epi32(lessThan,zero); 
                    let toAdd = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high);
                    pointers = _mm256_add_epi64(pointers, toAdd);
                }
                GatherRec0(&writing, rest, visitor);
            }
        }
    }
}
