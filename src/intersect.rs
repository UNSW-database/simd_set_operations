mod merge;
mod search;
mod svs;
mod adaptive;
mod std_set;
mod simd_shuffling;

pub use merge::*;
pub use search::{galloping, galloping_inplace};
pub use adaptive::*;
pub use std_set::*;
pub use svs::*;

#[cfg(feature = "simd")]
pub use simd_shuffling::*;

pub type Intersect2<I, V> = fn(a: &I, b: &I, visitor: &mut V);
pub type IntersectK<S, V> = fn(sets: &[S], visitor: &mut V);
