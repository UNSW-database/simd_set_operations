use std::arch::asm;

#[inline(always)]
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
			// restore rbx
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

#[inline(always)]
pub fn end() -> u64 {
    let out: u64;
    unsafe {
        asm!(
            // manually handle rbx clobbering because LLVM reserves it
            "mov {tmp}, rbx",
			// read time stamp counter, rdtscp partially serializes the processor
			// so we don't need a preceeding CPUID call
            "rdtscp",
            // move the measured values out before CPUID as it clobbers the
            // same registers
            "mov {out}, rax",
            "mov {upp}, rdx",
			// clear rax register for CPUID call
			"xor rax, rax",
			// serialize the CPU before we run any non-dependent instructions to 
			// ensure that rdtscp is called at the correct time
            "cpuid",
			// restore rbx
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

pub fn estimate_tsc_freq() -> u64 {
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
