pub struct KSetInput {
    data: Vec<i32>,
    ranges: Vec<std::ops::Range<u32>>,
}
impl KSetInput {
   pub fn new(vecs: &[Vec<i32>]) -> KSetInput {
       let mut i = 0;
       let mut data: Vec<i32> = Vec::new();
       let mut ranges: Vec<std::ops::Range<u32>> = Vec::new();
       for vec in vecs {
           let len = vec.len() as u32;
           data.extend(vec);
           ranges.push(std::ops::Range { start: i, end: i + len });
           i += len;
       }
       KSetInput {
           data,
           ranges,
       }
   }
    pub unsafe fn getIntial(&self) -> *const i32 {
         self.data.as_ptr()
    }
    pub fn getRange(&self, index: u32) -> &std::ops::Range<u32> {
        &self.ranges[index as usize]
    }
    pub fn getSlice(&self, index: u32) -> &[i32] {
        let slice = &self.data[self.getRange(index).start as usize..self.getRange(index).end as usize];
        slice
    }
    pub fn getSize(&self) -> u32 {
        self.ranges.len() as u32
    }
    pub fn getVec(&self) -> &Vec<i32> {
        &self.data
    }
    pub fn getRanges(&self) -> &Vec<std::ops::Range<u32>> {
        &self.ranges
    }
}