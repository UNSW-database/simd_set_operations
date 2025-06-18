use crate::intersect::{broadcast_avx2, GatherRec};
use crate::KSetInput::KSetInput;
use crate::visitor;
use crate::visitor::{VecWriter, Visitor};

#[cfg(target_feature = "avx2")]
pub fn BroadcastK<V>(initial: &Vec<i32>, additional: &KSetInput, visitor: &mut V)
where
    V: Visitor<i32> + visitor::SimdVisitor8,
{
    BroadcastKRec(initial, additional, 0, visitor);
}
#[cfg(target_feature = "avx2")]
fn BroadcastKRec<V>(initial: &Vec<i32>, additional: &KSetInput, index: u32, visitor: &mut V)
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
        _ => {
            let mut writer = VecWriter::<i32>::with_capacity(initial.len());
            broadcast_avx2(initial, additional.getSlice(index), &mut writer);
            let vec: Vec<i32>= writer.into();
            BroadcastKRec(&vec, additional, index + 1, visitor);
        }
    }
}
