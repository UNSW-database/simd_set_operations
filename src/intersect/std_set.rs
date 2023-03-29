use crate::{visitor::Visitor, CustomSet};
use std::{
    collections::{BTreeSet, HashSet},
    hash,
};

impl<T> CustomSet<T> for HashSet<T>
where
    T: Copy + Eq + hash::Hash,
{
    fn from_sorted(sorted: &[T]) -> Self {
        let mut set = HashSet::with_capacity(sorted.len());
        for item in sorted {
            set.insert(item.clone());
        }
        set
    }
}

pub fn hash_set_intersect<T>(
    set_a: &HashSet<T>,
    set_b: &HashSet<T>,
    visitor: &mut impl Visitor<T>,
) -> usize
where
    T: Copy + Eq + hash::Hash,
{
    let mut count = 0;
    for item in set_a.intersection(set_b) {
        visitor.visit(*item);
        count += 1;
    }

    count
}

impl<T> CustomSet<T> for BTreeSet<T>
where
    T: Ord + Copy,
{
    fn from_sorted(sorted: &[T]) -> Self {
        let mut set = BTreeSet::new();
        for item in sorted {
            set.insert(item.clone());
        }
        set
    }
}

pub fn btree_set_intersect<T: Ord + Copy>(
    set_a: &BTreeSet<T>,
    set_b: &BTreeSet<T>,
    visitor: &mut impl Visitor<T>,
) -> usize {
    let mut count = 0;
    for item in set_a.intersection(set_b) {
        visitor.visit(*item);
        count += 1;
    }

    count
}
