//! Module for working with the Time Stamp Counter (TSC)
//!
//! Timestamp reading code is based on:
//!
//! Paoloni, Gabriele. "How to benchmark code execution times on Intel IA-32 and IA-64
//! instruction set architectures." Intel Corporation 123.170 (2010).
//!
use crate::util::{large_median, median3_u64, small_median};
use serde::{Deserialize, Serialize};
use std::arch::asm;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct TSCCharacteristics {
    pub frequency: u64,
    pub overhead: u64,
    pub resolution: u64,
    pub error: (u64, u64),
}

/// Checks if the CPU has a TSC and that it supports the features required for our use case.
pub fn is_valid() -> bool {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::__cpuid;

    const CPUID_EXTENDED: u32 = 1u32 << 31;
    const CPUID_FEATURE_FLAGS: u32 = 0x1;
    const CPUID_EXTENDED_FEATURE_FLAGS: u32 = CPUID_EXTENDED | 0x1u32;
    const CPUID_POWER_MANAGEMENT: u32 = CPUID_EXTENDED | 0x7u32;

    {
        // Check that we can query the required CPUID leafs
        let cpuid_highest_param = unsafe { __cpuid(0) }.eax;
        let cpuid_highest_extended_param = unsafe { __cpuid(CPUID_EXTENDED) }.eax;

        if cpuid_highest_param < CPUID_FEATURE_FLAGS
            || cpuid_highest_extended_param < CPUID_POWER_MANAGEMENT
        {
            return false;
        }
    }

    {
        // Check that we support TSC
        let cpuid_feature_flags = unsafe { __cpuid(CPUID_FEATURE_FLAGS) }.edx;

        const TSC_MASK: u32 = 1u32 << 4;
        if cpuid_feature_flags & TSC_MASK == 0 {
            return false;
        }
    }

    {
        // Check that we support the RDTSCP instruction
        let cpuid_extended_feature_flags = unsafe { __cpuid(CPUID_EXTENDED_FEATURE_FLAGS) }.edx;

        const RDTSCP_MASK: u32 = 1u32 << 27;
        if cpuid_extended_feature_flags & RDTSCP_MASK == 0 {
            return false;
        }
    }

    {
        // Check that we have an invariant TSC
        let cpuid_power_management = unsafe { __cpuid(CPUID_POWER_MANAGEMENT) }.edx;

        const INVARIANT_TSC_MASK: u32 = 1u32 << 8;
        if cpuid_power_management | INVARIANT_TSC_MASK == 0 {
            return false;
        }
    }

    true
}

/// Read the TSC value at the start of a measurement.
///
/// This function reads the TSC with the RDTSC instruction and calculates the
/// 64-bit timestamp from the two 32-bit output registers. It also performas necessary
/// serializaiton of the CPU to prevent out-of-order execution affecting readings.
///
/// It should be used at the start of a measurement as reading the TSC at the start
/// and end of a measurement require different stategies to minimise measurement variance.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn start() -> u64 {
    let out: u64;
    unsafe {
        asm!(
            // manually handle rbx clobbering because LLVM reserves it
            "mov {tmp}, rbx",
            // clear rax for CPUID call
            "xor rax, rax",
            // serialize the processor before we read the TSC
            "cpuid",
            // read the TSC, we don't serialize after because the
            // inconsistency added by CPUID seems to be worse than potentially
            // out-of-order execution issues on current CPUs
            "rdtsc",
            // restore rbx immediately as LLV may use it in a generic register
            "mov rbx, {tmp}",
            // calculate 64-bit count
            "shl rdx, 32",
            "or rax, rdx",
            "mov {out}, rax",
            // outputs
            out = out(reg) out,
            // clobbers
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            tmp = out(reg) _,
        )
    }
    out
}

/// Read the TSC value at the end of a measurement.
///
/// This function reads the TSC with the RDTSCP instruction and calculates the
/// 64-bit timestamp from the two 32-bit output registers. It also performas necessary
/// serializaiton of the CPU to prevent out-of-order execution affecting readings.
///
/// It should be used at the end of a measurement as reading the TSC at the start
/// and end of a measurement require different stategies to minimise measurement variance.
///
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn end() -> u64 {
    let out: u64;
    unsafe {
        asm!(
            // read time stamp counter, rdtscp partially serializes the processor
            // so we don't need a preceeding CPUID call
            "rdtscp",
            // move the measured values out before CPUID as it clobbers the
            // same registers
            "mov {out}, rax",
            "mov {upp}, rdx",
            // manually handle rbx clobbering because LLVM reserves it
            "mov {tmp}, rbx",
            // clear rax register for CPUID call
            "xor rax, rax",
            // serialize the CPU before we run any non-dependent instructions to
            // ensure that rdtscp is called at the correct time
            "cpuid",
            // restore rbx immediately as LLV may use it in a generic register
            "mov rbx, {tmp}",
            // calculate 64-bit count
            "shl {upp}, 32",
            "or {out}, {upp}",
            // outputs
            out = out(reg) out,
            // clobbers
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            tmp = out(reg) _,
            upp = out(reg) _,
        )
    }
    out
}

pub fn characterise() -> TSCCharacteristics {
    let frequency = estimate_frequency();

    // Collect and sort enough control times to analyze overhead, resolution, and error
    let times: Vec<u64> = {
        let mut raw_times: Vec<u64> = std::iter::repeat_with(|| control()).take(10001).collect();
        raw_times.sort_unstable();

        let median = raw_times[raw_times.len() / 2];

        const MAX_DIFF: u64 = 100;
        raw_times
            .into_iter()
            .filter(|&t| t.abs_diff(median) <= MAX_DIFF)
            .collect()
    };

    // Estimate overhead as sample median
    let overhead = times[times.len() / 2];

    // Estimate resolution as minimum difference between any two times
    let resolution = times
        .iter()
        .map_windows(|&[&p, &n]| n - p)
        .filter(|&d| d != 0)
        .min()
        .unwrap();

    // Collect values for calculating error
    let (min, max) = times.iter().fold((u64::MAX, u64::MIN), |(min, max), &t| {
        (min.min(t), max.max(t))
    });

    // Estimate per-side error as maximum of full range and resolution
    let error = (
        resolution.max(min.abs_diff(overhead)),
        resolution.max(max.abs_diff(overhead)),
    );

    TSCCharacteristics {
        frequency,
        overhead,
        resolution,
        error,
    }
}

/// Estimate the frequency of the TSC
///
/// Estimates the frequency of the TSC by using the Rust's std::time::Instant as
/// a lower accuracy but accurate measurement of time. This works as we can assume
/// that the TSC operates at some multiple of 1 MHz and thus we only need a timer
/// with single-digit-millisecond precision to accurately estimate the frequency
/// of the TSC.
///
fn estimate_frequency() -> u64 {
    let instant_start = std::time::Instant::now();
    let tsc_start = start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let instant_end = std::time::Instant::now();
    let tsc_end = end();

    let instant_diff = instant_end.duration_since(instant_start).as_secs_f64();
    let tsc_diff = tsc_end - tsc_start;

    let freq_f64 = tsc_diff as f64 / instant_diff;
    // assume it will be a multiple of 1 MHz
    let freq = (freq_f64 / 1_000_000.0).round() as u64 * 1_000_000;

    freq
}

fn control() -> u64 {
    let start = start();
    let end = end();
    end - start
}

/// Measure the CPU frequency with the TSC
///
/// Repeatedly measures the runtime of a set of instructions with known
/// cycle count. We then take the median of these measurements and calculate:
///
/// freq_CPU = freq_TSC * cycles / TSC_count
///
pub fn measure_cpu_frequency<const CYCLES: u64, const TRIALS: usize>(
    tsc: TSCCharacteristics,
) -> u64 {
    assert!(TRIALS > 0 && CYCLES > 0);

    let mut buf = [0u64; TRIALS];
    for slot in buf.iter_mut() {
        *slot = trial::<CYCLES>()
    }

    let median = match TRIALS {
        1..=2 => buf[0],
        3 => median3_u64(&buf),
        4..=100 => small_median(&buf),
        _ => large_median(&mut buf),
    };

    (tsc.frequency * CYCLES) / (median - tsc.overhead)
}

#[cfg(target_arch = "x86_64")]
fn trial<const CYCLES: u64>() -> u64 {
    let mut sum: u64 = 0;

    let start = start();
    for _ in 0..CYCLES {
        // Inline assembly seems to stop constant folding,
        // however this may become an issue in the future
        unsafe {
            asm!(
                "add {val}, 1",
                val = inout(reg) sum,
            )
        }
    }
    let end = end();

    end - start
}
