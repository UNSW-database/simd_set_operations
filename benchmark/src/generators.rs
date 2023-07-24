use crate::schema::SetInfo;

use rand::{distributions::Uniform, thread_rng, Rng, seq::SliceRandom};

pub fn gen_twoset(props: &SetInfo) -> (Vec<i32>, Vec<i32>) {
    // Array of 1000 elements, density of 1%, max = 1000/0.01
    let density = props.density as f64 / 1000.0;
    let target_selectivity = props.selectivity as f64 / 1000.0;
    let size = 1 << props.size;
    let skew = 1 << (props.skew - 1);

    let small_len = size as usize;
    let large_len = (size * skew) as usize;
    let max = (large_len as f64 / density) as i32;

    let (target_shared_count, target_gen_count) =
        get_gen_counts(target_selectivity, small_len, large_len);

    let (shared_count, gen_count) = if target_gen_count as i32 > max {
        let shared_count = small_len + large_len - max as usize;
        //let actual_selectivity = shared_count as f64 / small_len as f64;
        //println!(
        //    "\nwarning: target selectivity {:.2} \
        //    is unachievable with density {:.2}\n(minimum selectivity {:.2})",
        //    target_selectivity, density, actual_selectivity
        //);
        (shared_count, max as usize)
    }
    else {
        (target_shared_count, target_gen_count)
    };
    
    //let max_selectivity = max_selectivity(max as usize, small_len, large_len);
    //let real_selectivity = 
    //if target_selectivity <= max_selectivity {
    //    max_selectivity
    //}
    //else {
    //    target_selectivity
    //};


    let rng = &mut thread_rng();
    let dist = Uniform::from(0..max);

    let items = if density < 0.2 {
        let mut items: Vec<i32> = Vec::new();
        while items.len() < gen_count {
            let need = gen_count - items.len();
            items.extend(rng.sample_iter(dist).take(need * 2));
            items.sort_unstable();
            items.dedup();
        }
        items.shuffle(rng);
        items.truncate(gen_count);
        items
    }
    else {
        let mut everything: Vec<i32> = (0..max).collect();
        everything.shuffle(rng);
        everything.truncate(gen_count);
        everything
    };

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

fn get_gen_counts(
    selectivity: f64,
    small_len: usize,
    large_len: usize) -> (usize, usize)
{
    let shared_count = (selectivity * small_len as f64) as usize;
    let different_count = small_len + large_len - 2*shared_count;
    let gen_count = shared_count + different_count;
    (shared_count, gen_count)
}
