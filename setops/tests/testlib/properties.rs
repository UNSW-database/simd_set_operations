
pub fn prop_intersection_correct<S, T>(result: Vec<T>, sets: &[S]) -> bool
where
    S: AsRef<[T]>,
    T: Ord + Copy,
{
    prop_strictly_increasing(&result) &&
    prop_result_items_all_common(&result, &sets) &&
    prop_all_common_items_in_result(&result, &sets)
}

pub fn prop_strictly_increasing<T>(result: &Vec<T>) -> bool
where
    T: Ord + Copy,
{
    result.windows(2).all(|w| w[0] < w[1])
}

// If an item is in the result, then it is a common item.
pub fn prop_result_items_all_common<S, T>(result: &Vec<T>, sets: &[S]) -> bool
where
    S: AsRef<[T]>,
    T: Ord + Copy,
{
    result.iter().all(|result_item| {
        sets.iter().all(|input_set| {
            input_set.as_ref().contains(&result_item)
        })
    })
}

// If an item is common, then it is in the result.
pub fn prop_all_common_items_in_result<S, T>(
    result: &Vec<T>,
    sets: &[S]) -> bool
where
    S: AsRef<[T]>,
    T: Ord + Copy,
{
    for item in sets[0].as_ref() {
        if sets.iter().skip(1).all(|set|
            set.as_ref().contains(item)
        ) {
            if !result.contains(item) {
                return false;
            }
        }
    }
    true
}
