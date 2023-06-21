use crate::{visitor::Visitor, intersect::branchless_merge};


#[inline(never)]
pub fn bmiss_scalar_3x<T, V>(mut left: &[T], mut right: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    const S: usize = 3;

    while left.len() >= S && right.len() >= S {

        if left[0] == right[0] || left[0] == right[1] || left[0] == right[2] {
            visitor.visit(left[0]);
        }
        if left[1] == right[0] || left[1] == right[1] || left[1] == right[2] {
            visitor.visit(left[1]);
        }
        if left[2] == right[0] || left[2] == right[1] || left[2] == right[2] {
            visitor.visit(left[2]);
        }


        if left[S-1] == right[S-1] {
            left = &left[S..];
            right = &right[S..];
        }
        else {
            let lt = (left[S-1] < right[S-1]) as usize;
            left = &left[S * lt..];
            right = &right[S * (lt^1)..];
        }
    }

    branchless_merge(left, right, visitor)
}

#[inline(never)]
pub fn bmiss_scalar_4x<T, V>(mut left: &[T], mut right: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    const S: usize = 4;

    while left.len() >= S && right.len() >= S {

        if left[0] == right[0] || left[0] == right[1] ||
            left[0] == right[2] || left[0] == right[3]
        {
            visitor.visit(left[0]);
        }
        if left[1] == right[0] || left[1] == right[1] ||
            left[1] == right[2] || left[1] == right[3]
        {
            visitor.visit(left[1]);
        }
        if left[2] == right[0] || left[2] == right[1] ||
            left[2] == right[2] || left[2] == right[3]
        {
            visitor.visit(left[2]);
        }
        if left[3] == right[0] || left[3] == right[1] ||
            left[3] == right[2] || left[3] == right[3]
        {
            visitor.visit(left[3]);
        }

        if left[S-1] == right[S-1] {
            left = &left[S..];
            right = &right[S..];
        }
        else {
            let lt = (left[S-1] < right[S-1]) as usize;
            left = &left[S * lt..];
            right = &right[S * (lt^1)..];
        }
    }

    branchless_merge(left, right, visitor)
}

pub fn bmiss_mono(left: &[i32], right: &[i32], visitor: &mut crate::visitor::VecWriter<i32>) {
    bmiss_scalar_3x(left, right, visitor);
    bmiss_scalar_4x(left, right, visitor)
}

