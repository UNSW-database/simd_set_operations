use crate::schema::*;

pub fn format_x(x: u32, info: &DatasetInfo) -> String {
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
        0..=9   => format!("{size}"),
        10..=19 => format!("{}KiB", 1 << (size - 10)),
        20..=29 => format!("{}MiB", 1 << (size - 20)),
        30..=39 => format!("{}GiB", 1 << (size - 30)),
        _ => size.to_string(),
    }
}

pub fn format_time(micros: f64) -> String {
    format!("{:.0}", micros)
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
