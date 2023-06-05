/// Used to receive set intersection results in a generic way. Inspired by
/// roaring-rs.
pub trait Visitor<T> {
    fn visit(&mut self, value: T);
    fn clear(&mut self);
}

/// Counts intersection size without storing result.
pub struct Counter {
    count: usize,
}

impl<T> Visitor<T> for Counter {
    fn visit(&mut self, _value: T) {
        self.count += 1;
    }

    fn clear(&mut self) {
        self.count = 0;
    }
}

impl Counter {
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Stores intersection result in a vector.
pub struct VecWriter<T> {
    items: Vec<T>,
}

impl<T> VecWriter<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }

    pub fn with_capacity(cardinality: usize) -> Self {
        Self {
            items: Vec::with_capacity(cardinality),
        }
    }
}

impl<T> AsRef<[T]> for VecWriter<T> {
    fn as_ref(&self) -> &[T] {
        &self.items
    }
}

impl<T> Into<Vec<T>> for VecWriter<T> {
    fn into(self) -> Vec<T> {
        self.items
    }
}

impl<T> Default for VecWriter<T> {
    fn default() -> Self {
        Self { items: Vec::default() }
    }
}

impl<T> Visitor<T> for VecWriter<T> {
    fn visit(&mut self, value: T) {
        self.items.push(value);
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

/// Writes intersection result to provided array slice.
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
