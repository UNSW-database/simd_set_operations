// Inspired by roaring-rs.
pub trait Visitor<T> {
    fn visit(&mut self, value: T);
    fn clear(&mut self);
}

pub struct VecWriter<T> {
    data: Vec<T>,
}

impl<T> VecWriter<T> {
    pub fn with_capacity(cardinality: usize) -> Self {
        Self {
            data: Vec::with_capacity(cardinality),
        }
    }
}

impl<T> AsRef<[T]> for VecWriter<T> {
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T> Into<Vec<T>> for VecWriter<T> {
    fn into(self) -> Vec<T> {
        self.data
    }
}

impl<T> Default for VecWriter<T> {
    fn default() -> Self {
        Self { data: Vec::default() }
    }
}

impl<T> Visitor<T> for VecWriter<T> {
    fn visit(&mut self, value: T) {
        self.data.push(value);
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}



pub struct SliceWriter<'a, T> {
    data: &'a mut[T],
    position: usize,
}

impl<'a, T> SliceWriter<'a, T> {
    fn position(&self) -> usize {
        self.position
    }
}

impl<'a, T> From<&'a mut[T]> for SliceWriter<'a, T> {
    fn from(data: &'a mut[T]) -> Self {
        Self {
            data: data,
            position: 0,
        }
    }
}

impl<'a, T> Visitor<T> for SliceWriter<'a, T> {
    fn visit(&mut self, value: T) {
        self.data[self.position] = value;
        self.position += 1;
    }

    fn clear(&mut self) {
        self.position = 0;
    }
}
