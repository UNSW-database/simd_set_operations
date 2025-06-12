pub struct KSetInput {
    data: Vec<u32>,
    ranges: Vec<std::ops::Range<u32>>,
}
impl KSetInput {
   pub fn new(vecs: Vec<Vec<u32>>) -> KSetInput {
       let mut i = 0;
       let mut data: Vec<u32> = Vec::new();
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
    pub unsafe fn getIntial(&self) -> *const u32 {
         self.data.as_ptr()
    }
    pub fn getRange(&self, index: u32) -> &std::ops::Range<u32> {
        &self.ranges[index as usize]
    }
}