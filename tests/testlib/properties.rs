use super::SetCollection;


pub fn prop_intersection_correct(
    result: Vec<u32>,
    sets: SetCollection) -> bool
{
    prop_strictly_increasing(&result) &&
    prop_result_items_all_common(&result, &sets) &&
    prop_all_common_items_in_result(&result, &sets)
}

pub fn prop_strictly_increasing(result: &Vec<u32>) -> bool {
    result.windows(2).all(|w| w[0] < w[1])
}

// If an item is in the result, then it is a common item.
pub fn prop_result_items_all_common(
    result: &Vec<u32>,
    sets: &SetCollection) -> bool
{
    result.iter().all(|result_item| {
        sets.sets().iter().all(|input_set| {
            input_set.as_slice().contains(&result_item)
        })
    })
}

// If an item is common, then it is in the result.
pub fn prop_all_common_items_in_result(
    result: &Vec<u32>,
    sets: &SetCollection) -> bool
{
    let sets_vec = sets.sets();

    for item in sets_vec[0].as_slice() {
        if sets_vec.iter().skip(1).all(|set|
            set.as_slice().contains(&item)
        ) {
            if !result.contains(&item) {
                return false;
            }
        }
    }
    true
}
