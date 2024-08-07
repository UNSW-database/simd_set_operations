#[cfg(target_os = "linux")]
use perf_event;
use crate::schema;

#[derive(Debug)]
pub struct PerfResults {
    pub l1d: CacheResult,
    pub l1i: CacheResult,
    pub ll: CacheResult,

    pub branches: Option<u64>,
    pub branch_misses: Option<u64>,

    pub cpu_stalled_front: Option<u64>,
    pub cpu_stalled_back: Option<u64>,
    pub instructions: Option<u64>,
    pub cpu_cycles: Option<u64>,
    pub cpu_cycles_ref: Option<u64>,
}

#[derive(Debug)]
pub struct CacheResult {
    pub rd_access: Option<u64>,
    pub rd_miss: Option<u64>,
    pub wr_access: Option<u64>,
    pub wr_miss: Option<u64>,
}

#[cfg(target_os = "linux")]
pub struct PerfCounters {
    group: perf_event::Group,
    l1d: CacheCounters,
    l1i: CacheCounters,
    ll: CacheCounters,
    branches: Option<perf_event::Counter>,
    branch_misses: Option<perf_event::Counter>,
    cpu_stalled_front: Option<perf_event::Counter>,
    cpu_stalled_back: Option<perf_event::Counter>,
    instructions: Option<perf_event::Counter>,
    cpu_cycles: Option<perf_event::Counter>,
    cpu_cycles_ref: Option<perf_event::Counter>,
}

#[cfg(not(target_os = "linux"))]
pub struct PerfCounters {
}

#[cfg(target_os = "linux")]
pub struct CacheCounters {
    pub rd_access: Option<perf_event::Counter>,
    pub rd_miss: Option<perf_event::Counter>,
    pub wr_access: Option<perf_event::Counter>,
    pub wr_miss: Option<perf_event::Counter>,
}

#[cfg(target_os = "linux")]
impl PerfCounters {
    pub fn new() -> Self {
        use perf_event::{*, events::*, events::Hardware};
        let mut group = Group::new().unwrap();
        
        let l1d = Self::cache_group(CacheId::L1D, &mut group);
        let l1i = Self::cache_group(CacheId::L1I, &mut group);
        // let ll = Self::cache_group(CacheId::LL, &mut group);
        let branches = group.add(&Builder::new(Hardware::BRANCH_INSTRUCTIONS)).ok();
        let branch_misses = group.add(&Builder::new(Hardware::BRANCH_MISSES)).ok();
        // let cpu_stalled_front = group.add(&Builder::new(Hardware::STALLED_CYCLES_FRONTEND)).ok();
        // let cpu_stalled_back = group.add(&Builder::new(Hardware::STALLED_CYCLES_BACKEND)).ok();
        let instructions = group.add(&Builder::new(Hardware::INSTRUCTIONS)).ok();
        let cpu_cycles = group.add(&Builder::new(Hardware::CPU_CYCLES)).ok();
        let cpu_cycles_ref = group.add(&Builder::new(Hardware::REF_CPU_CYCLES)).ok();

        // let lld = CacheCounters{ rd_access: None, rd_miss: None, wr_access: None, wr_miss: None};
        // let l1i = CacheCounters{ rd_access: None, rd_miss: None, wr_access: None, wr_miss: None};
        let ll = CacheCounters{ rd_access: None, rd_miss: None, wr_access: None, wr_miss: None};
        // let branches = None;
        // let branch_misses = None;
        let cpu_stalled_front = None;
        let cpu_stalled_back = None;
        // let cpu_cycles = None;
        // let cpu_cycles_ref = None;
        Self {
            group, l1d, l1i, ll, branches, branch_misses,
            cpu_stalled_front, cpu_stalled_back, instructions, cpu_cycles, cpu_cycles_ref
        }
    }

    pub fn summarise(&self) {
        use colored::Colorize;
        let convert = |c: &Option<perf_event::Counter>|
            c.as_ref().map_or("disabled".yellow(), |_| "enabled".green());

        println!("=== CPU Performance Counters ===");

        println!("l1d.rd_access: {}", convert(&self.l1d.rd_access));
        println!("l1d.rd_miss: {}", convert(&self.l1d.rd_miss));
        println!("l1d.wr_access: {}", convert(&self.l1d.wr_access));
        println!("l1d.wr_miss: {}", convert(&self.l1d.wr_miss));

        println!("l1i.rd_access: {}", convert(&self.l1i.rd_access));
        println!("l1i.rd_miss: {}", convert(&self.l1i.rd_miss));
        println!("l1i.wr_access: {}", convert(&self.l1i.wr_access));
        println!("l1i.wr_miss: {}", convert(&self.l1i.wr_miss));

        println!("ll.rd_access: {}", convert(&self.ll.rd_access));
        println!("ll.rd_miss: {}", convert(&self.ll.rd_miss));
        println!("ll.wr_access: {}", convert(&self.ll.wr_access));
        println!("ll.wr_miss: {}", convert(&self.ll.wr_miss));

        println!("branches: {}", convert(&self.branches));
        println!("branch_misses: {}", convert(&self.branch_misses));

        println!("cpu_stalled_front: {}", convert(&self.cpu_stalled_front));
        println!("cpu_stalled_back: {}", convert(&self.cpu_stalled_back));
        println!("instructions: {}", convert(&self.instructions));
        println!("cpu_cycles: {}", convert(&self.cpu_cycles));
        println!("cpu_cycles_ref: {}", convert(&self.cpu_cycles_ref));

        println!("================================");
    }

    pub fn enable(&mut self) {
        self.group.reset().unwrap();
        self.group.enable().expect("Failed to enable group");
    }

    pub fn disable(&mut self) {
        self.group.disable().expect("Failed to disable group");
    }

    pub fn results(&mut self) -> PerfResults {
        let counts = self.group.read().unwrap();
        PerfResults {
            l1d: Self::cache_results(&self.l1d, &counts),
            l1i: Self::cache_results(&self.l1i, &counts),
            ll: Self::cache_results(&self.ll, &counts),
            branches: self.branches.as_ref().map(|c| counts[c]),
            branch_misses: self.branch_misses.as_ref().map(|c| counts[c]),
            cpu_stalled_front: self.cpu_stalled_front.as_ref().map(|c| counts[c]),
            cpu_stalled_back: self.cpu_stalled_back.as_ref().map(|c| counts[c]),
            instructions: self.instructions.as_ref().map(|c| counts[c]),
            cpu_cycles: self.cpu_cycles.as_ref().map(|c| counts[c]),
            cpu_cycles_ref: self.cpu_cycles_ref.as_ref().map(|c| counts[c]),
        }
    }

    pub fn new_result_run(&self, x: u32) -> schema::ResultRun {
        schema::ResultRun {
            x: x,
            times: Vec::default(),
            l1d: Self::new_cache_run(&self.l1d),
            l1i: Self::new_cache_run(&self.l1i),
            ll: Self::new_cache_run(&self.ll),
            branches: self.branches.as_ref().map(|_| Vec::new()),
            branch_misses: self.branch_misses.as_ref().map(|_| Vec::new()),
            cpu_stalled_front: self.cpu_stalled_front.as_ref().map(|_| Vec::new()),
            cpu_stalled_back: self.cpu_stalled_back.as_ref().map(|_| Vec::new()),
            instructions: self.instructions.as_ref().map(|_| Vec::new()),
            cpu_cycles: self.cpu_cycles.as_ref().map(|_| Vec::new()),
            cpu_cycles_ref: self.cpu_cycles_ref.as_ref().map(|_| Vec::new()),
            bytes: Vec::default(),
        }
    }


    fn cache_group(which: perf_event::events::CacheId, group: &mut perf_event::Group) -> CacheCounters {
        use perf_event::{*, events::*};
        CacheCounters {
            rd_access: group.add(&Builder::new(Cache{ which: which, operation: CacheOp::READ, result: CacheResult::ACCESS })).ok(),
            rd_miss:   group.add(&Builder::new(Cache{ which: which, operation: CacheOp::READ, result: CacheResult::MISS })).ok(),
            wr_access: group.add(&Builder::new(Cache{ which: which, operation: CacheOp::WRITE, result: CacheResult::ACCESS })).ok(),
            wr_miss:   group.add(&Builder::new(Cache{ which: which, operation: CacheOp::WRITE, result: CacheResult::MISS })).ok(),
        }
    }

    fn cache_results(counters: &CacheCounters, counts: &perf_event::GroupData) -> CacheResult {
        CacheResult {
            rd_access: counters.rd_access.as_ref().map(|c| counts[c]),
            rd_miss: counters.rd_miss.as_ref().map(|c| counts[c]),
            wr_access: counters.wr_access.as_ref().map(|c| counts[c]),
            wr_miss: counters.wr_miss.as_ref().map(|c| counts[c]),
        }
    }

    fn new_cache_run(counters: &CacheCounters) -> schema::CacheRun {
        schema::CacheRun {
            rd_access: counters.rd_access.as_ref().map(|_| Vec::default()),
            rd_miss: counters.rd_miss.as_ref().map(|_| Vec::default()),
            wr_access: counters.wr_access.as_ref().map(|_| Vec::default()),
            wr_miss: counters.wr_miss.as_ref().map(|_| Vec::default()),
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl PerfCounters {
    pub fn new() -> Self {
        Self {}
    }

    pub fn summarise(&self) {
        println!("CPU performance counters disabled on non-linux platforms");
    }

    pub fn enable(&mut self) {
    }

    pub fn disable(&mut self) {
    }

    pub fn results(&mut self) -> PerfResults {
        PerfResults {
            l1d: CacheResult { rd_access: None, rd_miss: None, wr_access: None, wr_miss: None },
            l1i: CacheResult { rd_access: None, rd_miss: None, wr_access: None, wr_miss: None },
            ll: CacheResult { rd_access: None, rd_miss: None, wr_access: None, wr_miss: None },
            branches: None,
            branch_misses: None,
            cpu_stalled_front: None,
            cpu_stalled_back: None,
            instructions: None,
            cpu_cycles: None,
            cpu_cycles_ref: None,
        }
    }

    pub fn new_result_run(&self, x: u32) -> schema::ResultRun {
        schema::ResultRun {
            x: x,
            times: Vec::default(),
            l1d: Self::new_cache_run(),
            l1i: Self::new_cache_run(),
            ll: Self::new_cache_run(),
            branches: None,
            branch_misses: None,
            cpu_stalled_front: None,
            cpu_stalled_back: None,
            instructions: None,
            cpu_cycles: None,
            cpu_cycles_ref: None,
            bytes: Vec::default(),
        }
    }

    fn new_cache_run() -> schema::CacheRun {
        schema::CacheRun {
            rd_access: None,
            rd_miss: None,
            wr_access: None,
            wr_miss: None,
        }
    }
}
