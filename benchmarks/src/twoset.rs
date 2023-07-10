use crate::schema::SetInfo;

use rand::{distributions::Uniform, thread_rng, Rng, seq::SliceRandom};

pub fn gen_twoset(props: &SetInfo) -> (Vec<i32>, Vec<i32>) {
    // Gen array of random numbers of size shared + 2*unshared

    // Array of 1000 elements, density of 1%, max = 1000/0.01
    let density = props.density as f64 / 1000.0;
    let selectivity = props.selectivity as f64 / 1000.0;
    let small_len = props.size as usize;
    let large_len = (props.size * props.skew) as usize;
    let max = (large_len as f64 / density) as i32;

    let shared_count = (selectivity * small_len as f64) as usize;
    let different_count = small_len + large_len - shared_count;
    let gen_count = shared_count + different_count;

    let rng = &mut thread_rng();
    let dist = Uniform::from(0..max);

    let mut items: Vec<i32> = Vec::new();
    while items.len() < gen_count {
        let need = gen_count - items.len();
        items.extend(rng.sample_iter(dist).take(need * 2));
        items.sort_unstable();
        items.dedup();
    }
    items.shuffle(rng);

    let mut small = items[0..shared_count].to_vec();
    let mut large = items[0..shared_count].to_vec();
    small.extend_from_slice(&items[shared_count..small_len]);
    large.extend_from_slice(&items[small_len..gen_count]);
    small.sort_unstable();
    large.sort_unstable();

    assert!(small.len() == small_len);
    assert!(large.len() == large_len);

    (small, large)
}
