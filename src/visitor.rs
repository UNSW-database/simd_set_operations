// Inspired by roaring-rs.
pub trait Visitor<T> {
    fn preallocate(cardinality: usize) -> Self;
    fn visit(&mut self, value: T);
}

pub struct VecWriter<T> {
    data: Vec<T>,
}

impl<T> Into<Vec<T>> for VecWriter<T> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> Visitor<T> for VecWriter<T> {
    fn preallocate(cardinality: usize) -> Self {
        Self {
            data: Vec::with_capacity(cardinality),
        }
    }

    fn visit(&mut self, value: T) {
        self.data.push(value);
    }
}
