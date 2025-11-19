use std::{
    cell::Cell,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Default, Clone, Copy)]
pub struct Stage1Counters {
    pub linear_steps: u64,
    pub advance_a: u64,
    pub advance_b: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Stage2Counters {
    pub lowbyte_probes: u64,
    pub lowbyte_hits: u64,
    pub lowbyte_skipped: u64,
    pub bytegate_probes: u64,
    pub bytegate_hits: u64,
    pub bytegate_skipped: u64,
}

impl Stage1Counters {
    pub const fn new() -> Self {
        Self {
            linear_steps: 0,
            advance_a: 0,
            advance_b: 0,
        }
    }
}

impl Stage2Counters {
    pub const fn new() -> Self {
        Self {
            lowbyte_probes: 0,
            lowbyte_hits: 0,
            lowbyte_skipped: 0,
            bytegate_probes: 0,
            bytegate_hits: 0,
            bytegate_skipped: 0,
        }
    }
}

static STAGE_STATS_ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static STAGE1_COUNTERS: Cell<Stage1Counters> = Cell::new(Stage1Counters::new());
    static STAGE2_COUNTERS: Cell<Stage2Counters> = Cell::new(Stage2Counters::new());
}

fn stats_enabled() -> bool {
    STAGE_STATS_ENABLED.load(Ordering::Relaxed)
}

pub fn set_stage_stats_enabled(enabled: bool) {
    STAGE_STATS_ENABLED.store(enabled, Ordering::Relaxed);
}

fn with_stage1_counters<F: FnOnce(&mut Stage1Counters)>(f: F) {
    STAGE1_COUNTERS.with(|cell| {
        let mut current = cell.get();
        f(&mut current);
        cell.set(current);
    });
}

fn with_stage2_counters<F: FnOnce(&mut Stage2Counters)>(f: F) {
    STAGE2_COUNTERS.with(|cell| {
        let mut current = cell.get();
        f(&mut current);
        cell.set(current);
    });
}

pub fn reset_stage_counters() {
    if !stats_enabled() {
        return;
    }
    STAGE1_COUNTERS.with(|cell| cell.set(Stage1Counters::new()));
    STAGE2_COUNTERS.with(|cell| cell.set(Stage2Counters::new()));
}

pub fn take_stage1_counters() -> Stage1Counters {
    if !stats_enabled() {
        return Stage1Counters::new();
    }
    STAGE1_COUNTERS.with(|cell| {
        let counters = cell.get();
        cell.set(Stage1Counters::new());
        counters
    })
}

pub fn take_stage2_counters() -> Stage2Counters {
    if !stats_enabled() {
        return Stage2Counters::new();
    }
    STAGE2_COUNTERS.with(|cell| {
        let counters = cell.get();
        cell.set(Stage2Counters::new());
        counters
    })
}

pub fn record_stage1_linear_step(advance_a: bool, advance_b: bool) {
    if !stats_enabled() {
        return;
    }
    with_stage1_counters(|c| {
        c.linear_steps += 1;
        c.advance_a += advance_a as u64;
        c.advance_b += advance_b as u64;
    });
}

pub fn record_lowbyte_prefilter(probes: u64, hits: u64) {
    if probes == 0 || !stats_enabled() {
        return;
    }
    with_stage2_counters(|c| {
        c.lowbyte_probes += probes;
        c.lowbyte_hits += hits;
        c.lowbyte_skipped += probes.saturating_sub(hits);
    });
}

pub fn record_bytegate_prefilter(probes: u64, hits: u64) {
    if probes == 0 || !stats_enabled() {
        return;
    }
    with_stage2_counters(|c| {
        c.bytegate_probes += probes;
        c.bytegate_hits += hits;
        c.bytegate_skipped += probes.saturating_sub(hits);
    });
}
