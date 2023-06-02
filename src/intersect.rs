mod merge;
mod search;
mod svs;
mod adaptive;
mod std_set;

pub use merge::*;
pub use search::{galloping, galloping_inplace};
pub use adaptive::*;
pub use std_set::*;

pub type Intersect2<I, V> = fn(a: &I, b: &I, visitor: &mut V) -> usize;
pub type IntersectK<I, V> = fn(sets: &[&I], visitor: &mut V) -> usize;
