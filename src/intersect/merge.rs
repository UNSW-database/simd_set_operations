use crate::visitor::Visitor;

/// Classical set intersection via merge. Original author unknown.
// Inspired by https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
pub fn naive_merge<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    let mut idx_a = 0;
    let mut idx_b = 0;

    while idx_a < set_a.len() && idx_b < set_b.len() {
        let value_a = set_a[idx_a];
        let value_b = set_b[idx_b];
        if value_a < value_b {
            idx_a += 1;
        } else if value_b < value_a {
            idx_b += 1;
        } else {
            visitor.visit(value_a);
            idx_a += 1;
            idx_b += 1;
        }
    }
}

/// Removes hard-to-predict 'less than' branch.
/// From [BMiss](http://www.vldb.org/pvldb/vol8/p293-inoue.pdf) paper.
// Faster Set Intersection with SIMD instructions by Reducing Branch Mispredictions
// H. Inoue, M. Ohara, K. Taura, 2014
pub fn branchless_merge<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    let mut idx_a = 0;
    let mut idx_b = 0;

    while idx_a < set_a.len() && idx_b < set_b.len() {
        let value_a = set_a[idx_a];
        let value_b = set_b[idx_b];

        if value_a == value_b {
            visitor.visit(value_a);
            idx_a += 1;
            idx_b += 1;
        } else {
            idx_a += (value_a < value_b) as usize;
            idx_b += (value_b < value_a) as usize;
        }
    }
}
