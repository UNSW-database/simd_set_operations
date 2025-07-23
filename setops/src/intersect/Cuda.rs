use std::ops::Bound::Included;
use crate::intersect::{broadcast_avx2, broadcast_avx512};
use crate::KSetInput::KSetInput;
use crate::visitor;
use crate::visitor::Visitor;

#[repr(C)]
pub struct RangedData {
    pub first: IntSpan,
    pub second: RangeSpan,
}
#[repr(C)]
pub struct RangeSpan {
    pub ptr: *const Range,
    pub size: usize,
}
#[repr(C)]
pub struct IntSpan {
    pub ptr: *const i32,
    pub size: usize,
}
#[repr(C)]
pub struct Range {
    pub start: usize,
    pub end: usize,
}
pub fn convert(input : &KSetInput) -> (RangedData, Vec<Range>) {
    let ptr = input.getVec().as_ptr();
    let intSize = input.getVec().len();
    let mut vec: Vec<Range> = vec![];
    for mut r in input.getRanges() {
        vec.push(Range{start: r.start as usize,end: r.end as usize });
    }
    (RangedData{first : IntSpan{ptr: input.getVec().as_ptr(), size: input.getVec().len()}, second: RangeSpan{ptr: vec.as_ptr(), size: input.getRanges().len()}, }, vec)
}

extern "C" {
    fn cudaGatherWrapperC(input: RangedData, output: RangedData);
}

pub fn cudaGatherWarp<V>(input: RangedData, visitor: &mut V)
where
    V: Visitor<i32> + visitor::SimdVisitor8,
{
    let outSize = unsafe {(*(input.second.ptr)).end - (*input.second.ptr).start};
    let outputVec = vec![0; outSize];
    let range = Range{start: 0, end: 0};
    let output = RangedData{first: IntSpan{ptr: outputVec.as_ptr(), size: outSize}, second: RangeSpan{ptr: &range, size: 1}};
    unsafe {
        cudaGatherWrapperC(input, output);
    }
    let outputSlice = &outputVec[..range.end];
    // for outp in outputSlice {
    //     println!("{}",(*outp));
    // }
    broadcast_avx2(outputSlice, outputSlice, visitor);
}
