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
pub fn Gather<V>(initial: &Vec<i32>, additional: &KSetInput, visitor: &mut V)
where
    V: Visitor<i32> + visitor::SimdVisitor8,
{
}
#[cfg(target_feature = "avx2")]
pub fn GatherRec<V>(starting: &Vec<i32>, additional: &KSetInput, mut index: u32, visitor: &mut V)
where
    V: Visitor<i32> + visitor::SimdVisitor8,
{
    let mut initial = starting.clone();
    while true {
        match additional.getSize() - index  {
            0 => {
                broadcast_avx2(&initial, &initial, visitor);
                return;
            }
            1 => {
                broadcast_avx2(&initial,additional.getSlice(index), visitor);
                return;
            }
            n if n < 8 => {
                let mut writer: visitor::VecWriter<i32> = VecWriter::with_capacity(initial.len());
                broadcast_avx2(&initial, additional.getSlice(index), &mut writer);
                initial = writer.into();
                index += 1;
                // GatherRec(&newInit, additional, index + 1, visitor);
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
                    initial = resultsVec;
                    index += 8;
                    // GatherRec(initial, additional, index + 8, visitor);
                }
            }
        }
    }
}

