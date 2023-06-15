use crate::visitor::Visitor;


pub fn simd_shuffling<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    const SIMD_WIDTH: usize = 4;

    let count = 0;
    let mut i_a = 0;
    let mut i_b = 0;

    while i_a < set_a.len() && i_b < set_b.len() {
        
        

        i_a += SIMD_WIDTH;
        i_a += SIMD_WIDTH;
    }

}
