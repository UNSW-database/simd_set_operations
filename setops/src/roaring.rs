use crate::Set;
use roaring::RoaringBitmap;

impl Set<i32> for RoaringBitmap {
    fn from_sorted(sorted: &[i32]) -> Self {
        let mut bitmap = Self::new();
        bitmap.append(sorted.iter().map(|&i| i as u32)).unwrap();
        bitmap
    }
}

impl Set<u32> for RoaringBitmap {
    fn from_sorted(sorted: &[u32]) -> Self {
        let mut bitmap = Self::new();
        bitmap.append(sorted.iter().copied()).unwrap();
        bitmap
    }
}

pub fn roaring_intersect<V>(set_a: &RoaringBitmap, set_b: &RoaringBitmap, _visitor: &mut V) {
    let _ = set_a & set_b;
}
