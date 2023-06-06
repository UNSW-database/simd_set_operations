use crate::{visitor::Visitor, Set};
use std::{
    collections::{BTreeSet, HashSet},
    hash,
};

impl<T> Set<T> for HashSet<T>
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
    visitor: &mut impl Visitor<T>)
where
    T: Copy + Eq + hash::Hash,
{
    for item in set_a.intersection(set_b) {
        visitor.visit(*item);
    }
}

impl<T> Set<T> for BTreeSet<T>
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
    visitor: &mut impl Visitor<T>)
{
    for item in set_a.intersection(set_b) {
        visitor.visit(*item);
    }
}
