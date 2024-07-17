//! Module for working with the Time Stamp Counter (TSC)
//!
//! Timestamp reading code is based on:
//!
//! Paoloni, Gabriele. "How to benchmark code execution times on Intel IA-32 and IA-64
//! instruction set architectures." Intel Corporation 123.170 (2010).
//!
use std::arch::asm;

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

/// Estimate the frequency of the TSC
/// 
/// Estimates the frequency of the TSC by using the Rust's std::time::Instant as
/// a lower accuracy but accurate measurement of time. This works as we can assume
/// that the TSC operates at some multiple of 1 MHz and thus we only need a timer
/// with single-digit-millisecond precision to accurately estimate the frequency
/// of the TSC.
/// 
pub fn estimate_frequency() -> u64 {
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

/// Measure the overhead of a TSC measurement
/// 
/// Finds the median of a large sample of TSC measurements with nothing between
/// [start]/[end]. Median preferred over mean as measurements may be binned (such as 
/// on certain Ryzen processor where the TSC is always measured at some multiple
/// of the usually 100 MHz referenc clock).
/// 
/// This value should be subtracted from final measurement times made with 
/// [start]/[end]. This may result in underflows/negative values if you're trying
/// to measure very small time periods, though in such situations this means that 
/// such measurements are completely within the error of the TSC and are not
/// worth much anyway.
pub fn measure_overhead() -> u64 {
    let mut times: Vec<u64> = std::iter::repeat_with(|| control()).take(10001).collect();

    times.sort_unstable();

    times[times.len() / 2]
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
pub fn measure_cpu_frequency(tsc_frequency: u64, tsc_overhead: u64) -> u64 {
    const CYCLES: u64 = 10000;
    const SAMPLES: usize = 31;

    let mut counts: Vec<u64> = std::iter::repeat_with(|| trial(CYCLES) - tsc_overhead)
        .take(SAMPLES)
        .collect();

    counts.sort_unstable();

    (tsc_frequency * CYCLES) / counts[counts.len() / 2]
}

#[cfg(target_arch = "x86_64")]
fn trial(cycles: u64) -> u64 {
    let mut sum: u64 = 0;

    let start = start();
    for _ in 0..cycles {
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
