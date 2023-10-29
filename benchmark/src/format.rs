use crate::schema::*;

pub fn format_x(x: u32, info: &SyntheticDataset) -> String {
    match info.vary {
        Parameter::Density | Parameter::Selectivity =>
            format!("{:.2}", x as f64 / PERCENT_F),
        Parameter::Size => format_size(x),
        Parameter::Skew => if info.intersection.set_count == 2 {
            let skew = f64::powf(2.0, x as f64 / PERCENT_F) as usize;
            format!("1:{}", skew)
        } else {
            format!("f={}", x as f64 / PERCENT_F)
        },
        Parameter::SetCount => x.to_string()
    }
}

pub fn format_size(size: u32) -> String {
    match size {
        0..=9   => format!("{}", 1 << size),
        10..=19 => format!("{}Ki", 1 << (size - 10)),
        20..=29 => format!("{}Mi", 1 << (size - 20)),
        30..=39 => format!("{}Gi", 1 << (size - 30)),
        _ => size.to_string(),
    }
}

pub fn format_time(nanos: u64) -> String {
    if nanos < 10u64.pow(3) {
        format!("{}ns", nanos)
    }
    else if nanos < 10u64.pow(6) {
        format!("{}µs", nanos as f64 / 10usize.pow(3) as f64)
    }
    else if nanos < 10u64.pow(9) {
        format!("{}ms", nanos as f64 / 10usize.pow(6) as f64)
    }
    else {
        format!("{}s", nanos as f64 / 10usize.pow(9) as f64)
    }
}

pub fn format_xlabel(parameter: Parameter) -> &'static str {
    match parameter {
        Parameter::Density => "density",
        Parameter::Selectivity => "selectivity",
        Parameter::Size => "size",
        Parameter::Skew => "skew",
        Parameter::SetCount => "set count",
    }
}
