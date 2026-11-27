//! GPU emulation front end for the PS5 (AGC): submission queues, the gfx10
//! PM4 command processor (`cp`) and flips.
//!
//! Command buffers submitted through libSceAgcDriver are queued per hardware
//! queue (graphics, or a compute queue id) and executed in submission order
//! by one command-processor thread per queue, concurrently with the CPU, as
//! on the console. Dispatches run on Vulkan with `DS_GPU=1` (`exec`); draws
//! are decoded but not rendered yet (docs/STATUS.md).

pub mod cpu_profile;
pub mod record_opts;
pub mod capture;
pub mod nan_probe;
pub mod cp;
pub mod desc;
pub mod emucheck;
pub mod drawring;
pub mod frameprobe;
pub mod slotcensus;
pub mod hitch;
pub mod runprobe;
pub mod smalltrack;
/// Run-time switches for A/B measurements within one run (`tune <name>
/// <value>`; slowly varying load from other GPU users cancels out), each
/// seeded from its environment variable.
pub mod tune {
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Tunable {
        name: &'static str,
        env: &'static str,
        default: u64,
        value: AtomicU64,
    }

    const UNSET: u64 = u64::MAX;

    static ALL: [Tunable; 190] = [
        Tunable { name: "promises", env: "DS_PROMISES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lazy", env: "DS_LAZY_BARRIERS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_min", env: "DS_GPU_SHADOW_MIN", default: 64 << 10, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_vram", env: "DS_RING_VRAM", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "snap_index", env: "DS_SNAP_INDEX", default: 4 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_lo", env: "DS_RING_LO", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ring_hi", env: "DS_RING_HI", default: u64::MAX >> 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_keep", env: "DS_SHADOW_KEEP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "block_chain", env: "DS_BLOCK_CHAIN", default: 3, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_indirect", env: "DS_GPU_INDIRECT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lazy_wait_submit", env: "DS_LAZY_WAIT_SUBMIT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "slots", env: "DS_GPU_SLOTS", default: 4, value: AtomicU64::new(UNSET) },
        Tunable { name: "lod_stats", env: "DS_LOD_STATS", default: 3, value: AtomicU64::new(UNSET) },
        Tunable { name: "cmp_emulate", env: "DS_CMP_EMULATE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "tex_waits", env: "DS_TEX_WAITS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_max", env: "DS_PRE_MAX", default: 8 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "scratch_writes", env: "DS_SCRATCH_WRITES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "trace_sync", env: "DS_TRACE_SYNC_N", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "homes", env: "DS_HOMES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "reg_vars", env: "DS_REG_VARS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "map_ranges", env: "DS_MAP_RANGES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_cs", env: "DS_PRE_CS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par", env: "DS_PAR", default: 1000, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share", env: "DS_PRE_SHARE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pending_set", env: "DS_PENDING_SET", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "tex_scrub", env: "DS_TEX_SCRUB", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_min", env: "DS_PRE_SHARE_MIN", default: 16384, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_share_mib", env: "DS_PRE_SHARE_MIB", default: 4096, value: AtomicU64::new(UNSET) },
        Tunable { name: "submit_thread", env: "DS_SUBMIT_THREAD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "push_desc", env: "DS_PUSH_DESC", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_regs", env: "DS_LEAN_REGS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "span_cache", env: "DS_SPAN_CACHE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "fill_dma", env: "DS_FILL_DMA", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_cmd", env: "DS_LEAN_CMD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "exact_big", env: "DS_EXACT_BIG", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean_notes", env: "DS_LEAN_NOTES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_keep", env: "DS_PRE_KEEP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "clean_pages", env: "DS_CLEAN_PAGES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "upload_parts", env: "DS_UPLOAD_PARTS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_priority", env: "DS_CP_PRIORITY", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_cpus", env: "DS_CP_CPUS", default: u64::MAX - 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "guest_cpus", env: "DS_GUEST_CPUS", default: u64::MAX - 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "push_partial", env: "DS_PUSH_PARTIAL", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "per_lane", env: "DS_PER_LANE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "copy_queue", env: "DS_COPY_QUEUE_USE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_read_waits", env: "DS_SKIP_READ_WAITS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_split", env: "DS_PAR_SPLIT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_weight", env: "DS_PAR_WEIGHT", default: 500, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_barriers", env: "DS_SKIP_BARRIERS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "exact_indirect", env: "DS_EXACT_INDIRECT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "merge_barriers", env: "DS_MERGE_BARRIERS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "feedback_exact", env: "DS_FEEDBACK_EXACT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_classes", env: "DS_CP_CLASSES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "race_checks", env: "DS_RACE_CHECKS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "data_snapshots", env: "DS_DATA_SNAPSHOTS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "forward_copies", env: "DS_FORWARD_COPIES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "sync_cs", env: "DS_SYNC_CS_ON", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_min", env: "DS_PAR_MIN", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "label_early", env: "DS_LABEL_EARLY", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_min_compute", env: "DS_PAR_MIN_COMPUTE", default: 192, value: AtomicU64::new(UNSET) },
        Tunable { name: "chain_memo", env: "DS_CHAIN_MEMO", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "wait_yields", env: "DS_WAIT_YIELDS", default: 800, value: AtomicU64::new(UNSET) },
        Tunable { name: "timer_sleep", env: "DS_TIMER_SLEEP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cp_classes_acb", env: "DS_CP_CLASSES_ACB", default: 0x50, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_args", env: "DS_SHADOW_ARGS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_copies", env: "DS_SKIP_COPIES", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "zero_fill_clears", env: "DS_ZERO_FILL_CLEARS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "flip_async", env: "DS_FLIP_ASYNC", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "wait_ahead", env: "DS_WAIT_AHEAD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "marks_unlocked", env: "DS_MARKS_UNLOCKED", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lock_relief", env: "DS_LOCK_RELIEF", default: 3, value: AtomicU64::new(UNSET) },
        Tunable { name: "prep_pool", env: "DS_PREP_POOL", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "ww_probe", env: "DS_WW_PROBE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cmdbuf_merge", env: "DS_CMDBUF_MERGE", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_spins", env: "DS_PAR_SPINS", default: 512, value: AtomicU64::new(UNSET) },
        Tunable { name: "slots_gfx", env: "DS_GPU_SLOTS_GFX", default: 8, value: AtomicU64::new(UNSET) },
        Tunable { name: "split_every", env: "DS_SPLIT_EVERY", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "plan_check", env: "DS_PLAN_CHECK", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "homes_watch", env: "DS_HOMES_WATCH", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "image_homes", env: "DS_IMAGE_HOMES", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "attachless_draws", env: "DS_ATTACHLESS_DRAWS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "block_tails", env: "DS_BLOCK_TAILS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "mip_views", env: "DS_MIP_VIEWS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "compile_ahead", env: "DS_COMPILE_AHEAD", default: COMPILE_AHEAD_AUTO, value: AtomicU64::new(UNSET) },
        Tunable { name: "label_latency", env: "DS_LABEL_LATENCY", default: 2000, value: AtomicU64::new(UNSET) },
        Tunable { name: "precompile", env: "DS_PRECOMPILE", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "precompile_workers", env: "DS_PRECOMPILE_WORKERS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "depth_views", env: "DS_DEPTH_VIEWS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pre_trust", env: "DS_PRE_TRUST", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "draw_repeat", env: "DS_DRAW_REPEAT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "cpu_args_base", env: "DS_CPU_ARGS_BASE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "sgpr_pc", env: "DS_SGPR_PC", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "fill_skip", env: "DS_FILL_SKIP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "image_evict", env: "DS_IMAGE_EVICT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "vram_high", env: "DS_VRAM_HIGH", default: 85, value: AtomicU64::new(UNSET) },
        Tunable { name: "vram_low", env: "DS_VRAM_LOW", default: 75, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_cmds", env: "DS_SKIP_CMDS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "xwait_skip", env: "DS_XWAIT_SKIP", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "batch_cs", env: "DS_BATCH_CS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "job_idle", env: "DS_JOB_IDLE", default: crate::jobidle::DEFAULT_IDLE, value: AtomicU64::new(UNSET) },
        Tunable { name: "job_sleep", env: "DS_JOB_SLEEP", default: crate::jobidle::DEFAULT_SLEEP, value: AtomicU64::new(UNSET) },
        Tunable { name: "job_watch", env: "DS_JOB_WATCH", default: crate::jobidle::DEFAULT_WATCH, value: AtomicU64::new(UNSET) },
        Tunable { name: "sgpr_ubo", env: "DS_SGPR_UBO", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "power_optout", env: "DS_POWER_OPTOUT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "flip_grid", env: "DS_FLIP_GRID", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "present_fit", env: "DS_PRESENT_FIT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "frame_cap", env: "DS_FRAME_CAP", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "scrub_ms", env: "DS_SCRUB_MS", default: 10, value: AtomicU64::new(UNSET) },
        Tunable { name: "skip_steps", env: "DS_SKIP_STEPS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "stencil_export", env: "DS_STENCIL_EXPORT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_share_min", env: "DS_SHADOW_SHARE_MIN", default: 16384, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_runs", env: "DS_PAST_RUNS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_drop", env: "DS_PAST_DROP", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "acm_reverb", env: "DS_ACM_REVERB", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "trace_waits", env: "DS_TRACE_WAITS_N", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "plan_fast", env: "DS_PLAN_FAST", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "async_particles", env: "DS_ASYNC_PARTICLES", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_pre", env: "DS_SHADOW_PRE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_read_min", env: "DS_HOME_READ_MIN", default: 1 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "shadow_max", env: "DS_SHADOW_MAX", default: 4 << 20, value: AtomicU64::new(UNSET) },
        Tunable { name: "acb_cpus", env: "DS_ACB_CPUS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "host_census", env: "DS_HOST_CENSUS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "placement", env: "DS_PLACEMENT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "acb50_cpus", env: "DS_ACB50_CPUS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_check_unlocked", env: "DS_HOME_CHECK_UNLOCKED", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "home_check_win", env: "DS_HOME_CHECK_WIN", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_chains", env: "DS_GPU_CHAINS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "forward_earlier", env: "DS_FORWARD_EARLIER", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "forward_held", env: "DS_FORWARD_HELD", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_backoff", env: "DS_PAST_BACKOFF", default: 16, value: AtomicU64::new(UNSET) },
        Tunable { name: "submit_coalesce", env: "DS_SUBMIT_COALESCE", default: 16, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_check", env: "DS_PAST_CHECK", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_hot", env: "DS_PAST_HOT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_conflict_hold", env: "DS_PAST_CONFLICT_HOLD", default: 600, value: AtomicU64::new(UNSET) },
        Tunable { name: "fe2_wide_pages", env: "DS_FE2_WIDE_PAGES", default: 64, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_floor", env: "DS_PAR_FLOOR", default: 48, value: AtomicU64::new(UNSET) },
        Tunable { name: "fe2_cpw_close", env: "DS_FE2_CPW_CLOSE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "fe2_cpw_waited", env: "DS_FE2_CPW_WAITED", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "early_auto", env: "DS_EARLY_AUTO", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "write_marks", env: "DS_WRITE_MARKS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lod_late", env: "DS_LOD_LATE", default: 8, value: AtomicU64::new(UNSET) },
        Tunable { name: "lod_skip", env: "DS_LOD_SKIP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "gpu_detile", env: "DS_GPU_DETILE", default: 8, value: AtomicU64::new(UNSET) },
        Tunable { name: "mark_touch", env: "DS_MARK_TOUCH", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "upload_batch", env: "DS_UPLOAD_BATCH", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "record_defer", env: "DS_RECORD_DEFER", default: 5, value: AtomicU64::new(UNSET) },
        Tunable { name: "fill_partial", env: "DS_FILL_PARTIAL", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "fe2_back_granule", env: "DS_FE2_BACK_GRANULE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ballast_ns", env: "DS_BALLAST_NS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "ballast_at", env: "DS_BALLAST_AT", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "async_compute", env: "DS_ASYNC_COMPUTE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "vram_others", env: "DS_VRAM_OTHERS", default: 512, value: AtomicU64::new(UNSET) },
        Tunable { name: "mirror_cold", env: "DS_MIRROR_COLD", default: 3600, value: AtomicU64::new(UNSET) },
        Tunable { name: "buffer_delta", env: "DS_BUFFER_DELTA", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lod_cs", env: "DS_LOD_CS", default: 8, value: AtomicU64::new(UNSET) },
        Tunable { name: "tr_ballast", env: "DS_TR_BALLAST", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "prep_memo", env: "DS_PREP_MEMO", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "fe3_scan", env: "DS_FE3_SCAN", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "uncapped", env: "DS_UNCAPPED", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "cpw_lazy", env: "DS_CPW_LAZY", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "wb_probe", env: "DS_WB_PROBE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "wb_probe_site", env: "DS_WB_PROBE_SITE", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "unb_bound", env: "DS_UNB_BOUND", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "kco", env: "DS_KCO", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "r2", env: "DS_R2", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_subs", env: "DS_PAST_SUBS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "par_compute", env: "DS_PAR_COMPUTE", default: 1000, value: AtomicU64::new(UNSET) },
        Tunable { name: "submit_fuse", env: "DS_SUBMIT_FUSE", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "behind_quiet", env: "DS_BEHIND_QUIET", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "behind_us", env: "DS_BEHIND_US", default: 300, value: AtomicU64::new(UNSET) },
        Tunable { name: "async_gfx", env: "DS_ASYNC_GFX", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "async_gfx_workers", env: "DS_ASYNC_GFX_WORKERS", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "held_runs", env: "DS_HELD_RUNS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "past_repair", env: "DS_PAST_REPAIR", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "held_min", env: "DS_HELD_MIN", default: 192, value: AtomicU64::new(UNSET) },
        Tunable { name: "async_gfx_inline", env: "DS_ASYNC_GFX_INLINE", default: 12, value: AtomicU64::new(UNSET) },
        Tunable { name: "zk", env: "DS_ZK", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "xq_pre_c", env: "DS_XQ_PRE_C", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lean", env: "DS_LEAN", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "lane", env: "DS_LANE", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "join", env: "DS_JOIN", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "copy_snap", env: "DS_COPY_SNAP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "headless_present", env: "DS_HEADLESS_PRESENT", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pace_step", env: "DS_PACE_STEP", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "pace_smooth", env: "DS_PACE_SMOOTH", default: 8, value: AtomicU64::new(UNSET) },
        Tunable { name: "pace_phys", env: "DS_PACE_PHYS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "vrr", env: "DS_VRR", default: 2, value: AtomicU64::new(UNSET) },
        Tunable { name: "tick_lead", env: "DS_TICK_LEAD", default: 0, value: AtomicU64::new(UNSET) },
        Tunable { name: "layer_clears", env: "DS_LAYER_CLEARS", default: 1, value: AtomicU64::new(UNSET) },
        Tunable { name: "arm_inline_max", env: "DS_ARM_INLINE_MAX", default: 1024, value: AtomicU64::new(UNSET) },
    ];

    /// A switch's value: one relaxed load once set (inlined into every
    /// caller; the per-command paths read dozens of switches).
    #[inline(always)]
    fn get(i: usize) -> u64 {
        let v = ALL[i].value.load(Ordering::Relaxed);
        if v != UNSET {
            return v;
        }
        first_get(i)
    }

    /// One step of a countdown switch (`tune trace_waits <n>` and the
    /// like): true while it was above 0. Found by name: removing a switch
    /// renumbers the positions `get` reads (a literal position here once
    /// counted down `shadow_pre` instead of `trace_waits`).
    #[cold]
    fn count_down(name: &str) -> bool {
        ALL.iter().find(|t| t.name == name).is_some_and(|t| t.value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| if v != UNSET && v > 0 { Some(v - 1) } else { None }).is_ok())
    }

    /// The first read of a switch: its environment variable or default.
    #[cold]
    #[inline(never)]
    fn first_get(i: usize) -> u64 {
        let t = &ALL[i];
        let v = std::env::var(t.env).ok().and_then(|x| x.trim().parse().ok()).unwrap_or(t.default);
        t.value.store(v, Ordering::Relaxed);
        v
    }

    /// Label promises (`Cp::wait`).
    #[inline]
    pub fn promises() -> bool {
        get(0) != 0
    }

    /// Dispatches ordered by the title's synchronization only.
    #[inline]
    pub fn lazy_barriers() -> bool {
        get(1) != 0
    }

    /// Written buffers below this size skip VRAM shadows (unless atomic).
    #[inline]
    pub fn shadow_min() -> u64 {
        get(2)
    }

    /// Which stages read their SGPR data from video memory instead of system
    /// memory (it changes timing, not results): 0 none, 1 all (the default
    /// since 2026-11-12, in 512 KiB segments, at most 24 MiB of the
    /// CPU-visible heap for the graphics queue's executors and 16 MiB for the
    /// compute queues': `exec::RING_VRAM_SEG`), 2 the graphics queue's, 3
    /// the compute queues', 4 shaders at addresses in [`ring_lo`, `ring_hi`).
    #[inline]
    pub fn ring_vram() -> u64 {
        get(3)
    }

    #[inline]
    pub fn ring_range() -> std::ops::Range<u64> {
        get(5)..get(8)
    }

    /// Index buffers up to this size are snapshotted into video memory when
    /// recorded (0: off).
    #[inline]
    pub fn snap_index() -> u64 {
        get(4)
    }

    /// VRAM shadows stay resident for the rest of their batch (merged back
    /// once) instead of being merged after each dispatch.
    #[inline]
    pub fn shadow_keep() -> bool {
        get(7) != 0
    }

    /// Structured control flow instead of the translator's dispatcher: 3 (the
    /// default since 2026-11-12 night) for every shader with each block
    /// nested in its dominator's region and joins unguarded
    /// (`rdna::translate::Spec::dom_nest`), 1 for every shader as a flat
    /// chain of guarded blocks (before), 2 for loop-free ones only, 0 never;
    /// switching translates shaders anew.
    #[inline]
    pub fn block_chain() -> u64 {
        get(8)
    }

    /// Read-only buffer bindings and index buffers up to this size are read
    /// from a copy in video memory that the GPU makes at the start of their
    /// batch (`Exec::pre_copy`; 0: off) instead of across PCIe (vertex data,
    /// the per-item culls' meshes): draws took ~45% less GPU time and the
    /// courtyard ran ~10% faster (in-run A/B, run 20261011-063258). 8 MiB
    /// since 2026-10-12: the foliage draws' 6.8 MB instance buffer was read in
    /// place (~9 ms of draw time per courtyard frame); GPU time per frame
    /// ~11% lower in each of three A/B cycles, rate median +3.9% (run
    /// 20261012-132127; 16 MiB was worse earlier).
    #[inline]
    pub fn pre_max() -> u64 {
        get(15)
    }

    /// Diagnosis only (1; breaks rendering): written buffer bindings too large
    /// for a VRAM shadow store into a video-memory scratch buffer and their
    /// results are lost. Measures what GPU stores to imported host memory
    /// cost (the per-item culls' index output).
    #[inline]
    pub fn scratch_writes() -> bool {
        get(16) != 0
    }

    /// Diagnosis only (2; breaks rendering): every written buffer binding
    /// stores into the scratch buffer, none shadowed, merged or written in
    /// guest memory: the upper bound of keeping GPU-written data in video
    /// memory (residency design stage 3), its readers left as they are.
    #[inline]
    pub fn scratch_all() -> bool {
        get(16) == 2
    }

    /// Which textures a draw or dispatch waits for GPU writes to (their guest
    /// memory): 1 those a coarse bound finds written, then only the ones read
    /// from guest memory now, with their exact span (`Exec::sync_textures`);
    /// 0 every texture with the coarse bound of 16 bytes per texel (A/B).
    #[inline]
    pub fn tex_waits() -> u64 {
        get(14)
    }

    /// Indirect dispatches whose arguments GPU work still writes read them on
    /// the GPU (else the command processor waits for that work first).
    #[inline]
    pub fn gpu_indirect() -> bool {
        get(9) != 0
    }

    /// A label wait submits the recorded work only when it has to poll (a wait
    /// that passes at once does not end the batch).
    #[inline]
    pub fn lazy_wait_submit() -> bool {
        get(10) != 0
    }

    /// Batch slots a queue cycles through (`exec::MAX_SLOTS` at most): it
    /// records one batch while the batches of the others execute.
    #[inline]
    pub fn slots() -> u64 {
        get(11)
    }

    /// GetLodStats (hardware LOD statistics, the title's texture streaming
    /// feedback; `lodstats`): 0 leaves the buffer as the title wrote it (no
    /// streaming of high mips), 1 reports every counter bank as sampled at
    /// mip 0, 2 the banks of textures bound since the last report (at mip
    /// 0), 3 what the sampling instructions measured on the GPU (finest mip
    /// and lane count per bank; translations change with it).
    #[inline]
    pub fn lod_stats() -> u64 {
        get(12)
    }

    /// Depth-comparison sampling of color images is emulated in the
    /// translation (Vulkan compares only on depth formats); 0 compares in
    /// hardware everywhere (undefined on color formats; A/B).
    #[inline]
    pub fn cmp_emulate() -> bool {
        get(13) != 0
    }

    /// Index buffers that GPU work wrote through a binding too large for a
    /// VRAM shadow get a home in video memory that later GPU uses of the
    /// range bind (`gpu::homes`): 3.29 against 2.26 flips/s in the
    /// courtyard (in-run A/B, run 20261011-072603). 0 writes the homes back
    /// and drops them.
    #[inline]
    pub fn homes() -> bool {
        get(18) != 0
    }

    /// Translations keep each register in its own variable (1) or the
    /// register files in arrays (0, the earlier form): see
    /// `rdna::translate::Spec::reg_vars`. Switching translates programs anew.
    #[inline]
    pub fn reg_vars() -> bool {
        get(19) != 0
    }

    /// The per-thread caches of readable guest ranges (`mem::readable_span`)
    /// keep the runs a mapping change did not touch; 0 empties them at every
    /// change (the earlier behavior, for A/B).
    #[inline]
    pub fn map_ranges() -> bool {
        get(20) != 0
    }

    /// Read-only buffer bindings of dispatches copied into video memory at
    /// the batch's start like those of draws (`Exec::pre_copy`); 0: compute
    /// shaders read them from guest memory in place.
    #[inline]
    pub fn pre_cs() -> bool {
        get(21) != 0
    }

    /// Worker command processors that record chunks of long
    /// synchronization-free runs in parallel (`gpu::par`); 0: off; 1000 (the
    /// default): half the logical CPUs, 2 to 8 (`par::workers`).
    #[inline]
    pub fn par() -> u64 {
        get(22)
    }

    /// Batch-start copies of the graphics queue shared across its batches
    /// (`gpu::precache`); 0: each batch copies what it binds.
    #[inline]
    pub fn pre_share() -> bool {
        get(23) != 0
    }

    /// Draws check pending fast clears and overwrites of their images in a
    /// published set (no image-cache lock); 0: under the lock (A/B).
    #[inline]
    pub fn pending_set() -> bool {
        get(24) != 0
    }

    /// Uploaded textures checked for CPU writes by a background thread
    /// (`images::scrubbing`): 2 by their write-watch marks (exact; the pages
    /// go to the write tracker, and re-uploads leave out parts whose bytes
    /// are unchanged: the title rewrites texture memory with the same data),
    /// 1 by sampled hashes (the thread's CPU 5.7-6.4 against 2.3-2.5 ms per
    /// frame, CPU per command +0.4 to +4.4%, run 20261013-083150); 0: by the
    /// first draw binding each per frame (sampled).
    #[inline]
    pub fn tex_scrub() -> bool {
        get(25) != 0
    }

    /// `tex_scrub`'s mode.
    #[inline]
    pub fn tex_scrub_mode() -> u64 {
        get(25)
    }

    /// Smallest batch-start copy kept in the shared video-memory chunks
    /// (`gpu::precache`), bytes.
    #[inline]
    pub fn pre_share_min() -> u64 {
        get(26)
    }

    /// Video memory for shared batch-start copies, MiB.
    #[inline]
    pub fn pre_share_mib() -> u64 {
        get(27)
    }

    /// Batches are submitted to Vulkan by a thread of their own
    /// (`exec::push_batch`) instead of by the command processors.
    #[inline]
    pub fn submit_thread() -> bool {
        get(28) != 0
    }

    /// Pipelines created from now on take their descriptors pushed into the
    /// command buffer (VK_KHR_push_descriptor) when their set layout fits
    /// maxPushDescriptors, instead of a descriptor set allocated, written
    /// and bound per command; pipelines are cached per setting (an in-run
    /// A/B compiles both).
    #[inline]
    pub fn push_desc() -> bool {
        get(29) != 0
    }

    /// Command-processor waits for GPU work still to be traced (`tune
    /// trace_sync <n>`: the next n are logged with the read that collided,
    /// its writer and the shader).
    #[inline]
    pub fn get_trace_sync() -> bool {
        get(17) > 0
    }

    /// Takes one of the waits still to be traced.
    #[inline]
    pub fn take_trace_sync() -> bool {
        get(17) > 0 && count_down("trace_sync")
    }

    /// Register-state snapshots of parallel chunks (and replays) hold only
    /// the registers written so far (`cp::HIGH`); 0 copies whole register
    /// files (100 KiB each), for A/B.
    #[inline]
    pub fn lean_regs() -> bool {
        get(30) != 0
    }

    /// The shader-input reader keeps the mappings of its last four reads and
    /// looks a miss up once; 0: the last read's mapping, two lookups per miss
    /// (the earlier behavior, for A/B).
    #[inline]
    pub fn span_cache() -> bool {
        get(31) != 0
    }

    /// Dispatches of the title's fill kernel over a whole range are buffer
    /// fills of guest memory in place (`verify::full_fill`), without the
    /// kernel and its VRAM shadow; 0: the kernel runs (for A/B).
    #[inline]
    pub fn fill_dma() -> bool {
        get(32) != 0
    }

    /// Per-command steps without needless work: the main command processor
    /// spins on a flag for its workers, and a compute pipeline already bound
    /// in the command buffer is not bound again; 0: as before (for A/B).
    #[inline]
    pub fn lean_cmd() -> bool {
        get(33) != 0
    }

    /// Range unions keep ranges over 64 pages exactly beside their page
    /// filter (`ranges::Ranges`); 0: any such range switches the filter off
    /// (the earlier behavior, for A/B).
    #[inline]
    pub fn exact_big() -> bool {
        get(34) != 0
    }

    /// A dispatch writing the same ranges as its executor's previous command
    /// skips the global write notes (image cache, write tracker, copy cache)
    /// while their generations are unchanged (`Exec::last_notes`); 0: every
    /// dispatch notes them (for A/B).
    #[inline]
    pub fn lean_notes() -> bool {
        get(35) != 0
    }

    /// The shared batch-start copies evict the oldest chunk no batch binds
    /// (not the oldest, often bound by every frame's batches), and a range
    /// another batch's unsubmitted prologue copies is copied privately
    /// instead of replacing that entry; 0: as before (for A/B).
    #[inline]
    pub fn pre_keep() -> bool {
        get(36) != 0
    }

    /// The command processors' read checks skip reads in 64 KiB pages found
    /// clean (no pending or in-flight write, no stale home part) since the
    /// write generations last changed (`Exec::clean_pages`); 0: every read is
    /// checked (for A/B).
    #[inline]
    pub fn clean_pages() -> bool {
        get(37) != 0
    }


    /// Texture re-uploads upload only the (mip, layer)s that the written
    /// ranges (mappings, file reads) or the CPU's writes since the last
    /// upload (write-watch marks, taken through the copy cache) overlap: the
    /// title streams texture arrays a layer at a time (1024x1024x64 BC7,
    /// 1024x1024x256), and every layer it mapped re-uploaded the whole array.
    /// Upload time -32% and the level load to the gate 18.2 -> 13.5 s (runs
    /// 20261013-073959 / -073718); DS_UPLOAD_CHECK=1 found no changed part
    /// left out (run -072814). 0: whole re-uploads (for A/B).
    #[inline]
    pub fn upload_parts() -> bool {
        get(38) != 0
    }

    /// Host priority of the GPU front end's threads (`prio`): 0 normal,
    /// 1 above normal (default since 2026-10-13: on 12 of the 5900X's 24
    /// logical CPUs 11.3-11.5 flips/s instead of 2.3, the same at 24; run
    /// 20261013-124535), 2 highest, 3 above normal for the graphics command
    /// processor only (the others at normal: busy front-end threads above
    /// the title's starve its jobs, and World 8-3 loads into a different
    /// state; FINDINGS.md 2026-10-14).
    #[inline]
    pub fn cp_priority() -> u64 {
        get(39)
    }

    /// Logical CPUs the GPU front end's threads may run on (`prio`; a
    /// decimal mask, 0 = all of the process's; default `prio::CPUS_AUTO`:
    /// one last-level-cache group when there are several, `prio::auto_cpus`).
    #[inline]
    pub fn cp_cpus() -> u64 {
        get(40)
    }

    /// Logical CPUs the title's threads may run on (`thread::apply_guest_cpus`;
    /// a decimal mask, 0 = all of the process's; AUTO, the default: the
    /// CPUs the front end leaves, `prio::auto_guest_cpus`).
    #[inline]
    pub fn guest_cpus() -> u64 {
        match get(41) {
            super::prio::CPUS_AUTO => super::prio::auto_guest_cpus(),
            m => m,
        }
    }

    /// A command with the same pipeline layout as the previous push and no
    /// pipeline bound since pushes only the descriptors that changed
    /// (`exec::bind_descriptors`).
    #[inline]
    pub fn push_partial() -> bool {
        get(42) != 0
    }

    /// The listed compute programs are translated per lane
    /// (`shaders::PER_LANE_PROGRAMS`, `Spec::per_lane`): 1 all, 2 those
    /// without subgroup-decided EXEC branches (`Spec::lane_votes`), 0 none;
    /// switching translates them anew.
    #[inline]
    pub fn per_lane() -> u64 {
        get(43)
    }

    /// Batch-start copies of ranges no submitted batch writes run on the
    /// copy queue when there is one (`DS_COPY_QUEUE=0` at boot: none; 0
    /// here: all in the prologues on the main queue).
    #[inline]
    pub fn copy_queue() -> bool {
        get(44) != 0
    }

    /// Diagnosis (timing only, wrong results): stage preparations do not
    /// wait for GPU work writing what they read (they use memory as it is).
    #[inline]
    pub fn skip_read_waits() -> bool {
        get(45) != 0
    }

    /// Chunks per recorder a parallel run is split into (`gpu::par`): the
    /// workers take them in order and submit each when its turn comes; 1:
    /// one each.
    #[inline]
    pub fn par_split() -> u64 {
        get(46)
    }

    /// The main command processor's chunk of a parallel run relative to a
    /// worker's, in 1/1000 (0: `par_balance` decides): half a worker's, as it
    /// records the cull run slower per command than they do (+1.0-1.4%,
    /// FINDINGS.md 2026-10-14).
    #[inline]
    pub fn par_weight() -> u64 {
        get(47)
    }

    /// Diagnosis (timing only, wrong results): full barriers left out, by
    /// site: 1 after a render pass, 2 before an indirect dispatch, 4 around
    /// a CP memory write the GPU performs, 8 around a GPU timestamp store,
    /// 16 the title's synchronization (partial flushes, cache actions,
    /// end-of-pipe events) before the next command, 32 between dispatches
    /// and the draws after them, 64 a batch's opening barrier, 128 the
    /// prologue's barrier after earlier work, 256 the barrier before an
    /// end-of-pipe CP write (light end-of-pipe writes).
    #[inline]
    pub fn skip_barriers() -> u64 {
        get(48)
    }

    /// An indirect dispatch whose arguments the GPU reads waits behind a full
    /// barrier only when its batch wrote them since its last one (0: always).
    #[inline]
    pub fn exact_indirect() -> bool {
        get(49) != 0
    }

    /// A render pass ended right before a full barrier takes none of its own
    /// (0: one each).
    #[inline]
    pub fn merge_barriers() -> bool {
        get(50) != 0
    }

    /// A draw that samples an attachment of the current rendering instance
    /// ends it only when the instance writes that attachment (or the draw
    /// does, or stores to it); 0: whenever it samples one.
    #[inline]
    pub fn feedback_exact() -> bool {
        get(51) != 0
    }

    /// The graphics command processor's time by packet class, waits apart
    /// (profile report line; costs two clock reads per packet).
    #[inline]
    pub fn cp_classes() -> bool {
        get(52) != 0
    }

    /// Diagnosis: read-only data the shaders read on the GPU (4 KiB or less
    /// per binding) is compared, once its batch completed, with what guest
    /// memory held when the command was recorded (`exec::check_races`).
    #[inline]
    pub fn race_checks() -> bool {
        get(53) != 0
    }

    /// The largest range the race checks follow: `tune race_checks 1` 4 KiB
    /// (words kept), a larger value that many bytes (ranges over 4 KiB are
    /// compared by hash).
    #[inline]
    pub fn race_max() -> u64 {
        match get(53) {
            0 => 0,
            1 => 4096,
            v => v,
        }
    }

    /// Data-only loads read on the GPU (`Spec::gpu_loads`) of data no GPU work
    /// still writes are bound to a snapshot taken when the command is
    /// recorded (0: guest memory as the batch executes).
    #[inline]
    pub fn data_snapshots() -> bool {
        get(54) != 0
    }

    /// The per-view copy kernel's result forwarded to the graphics queue's
    /// later reads of its destination instead of waiting for the copy
    /// (`gpu::forward`).
    #[inline]
    pub fn forward_copies() -> bool {
        get(55) != 0
    }

    /// Dispatches of the listed compute shaders (`exec::sync_cs`; by default
    /// two load-time kernels) are completed right after they are recorded
    /// (0: not; switchable while running).
    #[inline]
    pub fn sync_cs() -> bool {
        get(56) != 0
    }

    /// The fewest commands a run is split into parallel chunks at (chunks
    /// at least a third of it, at least 16 commands); 0: automatic
    /// (`par::min_run`).
    #[inline]
    pub fn par_min() -> u64 {
        get(57)
    }

    /// A GPU-side label write recorded while another queue blocks on that
    /// label submits its batch at the next packet (0: when another queue
    /// asks before the write is recorded, or after `label_latency`).
    #[inline]
    pub fn label_early() -> bool {
        get(58) != 0
    }

    /// `par_min` for the compute queues' runs.
    #[inline]
    pub fn par_min_compute() -> u64 {
        get(59)
    }

    /// Stage preparations resolve each of their plan's distinct pointer
    /// chains once (`Plan::chain_ids`; 0: looked up by hash per use).
    #[inline]
    pub fn chain_memo() -> bool {
        get(60) != 0
    }

    /// A command processor's polling wait (labels, REWIND) yields its
    /// time slice this many times after 200 spins before it sleeps 100 us
    /// between checks.
    #[inline]
    pub fn wait_yields() -> u64 {
        get(61)
    }

    /// Short sleeps (polling waits, the title's usleep and nanosleep) on a
    /// waitable timer each thread keeps (`thread::short_sleep`; 0: std's
    /// sleep, which makes and closes a timer per call).
    #[inline]
    pub fn timer_sleep() -> bool {
        get(62) != 0
    }

    /// The compute queue (its id, e.g. 0x50) whose command processor's time
    /// `cp_classes` also gives per packet opcode.
    #[inline]
    pub fn cp_classes_acb() -> u64 {
        get(63)
    }

    /// An indirect dispatch whose arguments a resident shadow holds (an
    /// earlier dispatch of the batch wrote them in its working copy) reads
    /// them there instead of merging the shadow back first: 1 (default since
    /// 2026-10-27: +1.6% at the vista in an in-run A/B) on every executor, 2
    /// on the graphics queue's (its command processor and parallel workers)
    /// only, 0 off.
    #[inline]
    pub fn shadow_args() -> u64 {
        get(64)
    }

    /// Diagnosis (timing only, wrong results): copies between guest memory
    /// and video memory left out, as if the data lived in video memory: 1
    /// the copies into new shadows (use only with 2: the merges would store
    /// garbage), 2 the merges of shadows back to guest memory (guest memory
    /// keeps older values), 4 the batch-start copies (read-only data), 8 the
    /// fill kernel's large zero fills stored in place (guest memory keeps
    /// older values; images they cover are still cleared), 16 / 32 / 64 the
    /// merges of windows of at least 1 MiB / 256 KiB to 1 MiB / under 256 KiB;
    /// windows under 256 KiB written back by a whole copy instead of the
    /// merge kernel (128; CPU-written words in them are lost) or merged into
    /// video memory instead of guest memory (256) or into host memory
    /// nothing reads (2048), into our own write-watched imported memory
    /// nothing reads at one place (4096) or at each window's guest address
    /// modulo 64 MiB (8192); merged into guest memory and then once more into
    /// guest memory (32768) or into the spread scratch (65536) (results
    /// correct: what the stores cost when later work reads the same data;
    /// the others leave GPU-written data in guest memory stale, which can
    /// change the title's GPU-driven work); 512 the title's timestamps stored into video
    /// memory; 1024 GPU-performed end-of-pipe label writes made twice.
    #[inline]
    pub fn skip_copies() -> u64 {
        get(65)
    }

    /// GPU-owned images whose memory the fill kernel zeroes (recognized
    /// when recorded, `verify::full_fill`) are cleared right after the fill
    /// in its batch (`Images::zero_filled`), instead of their next user
    /// waiting for the GPU and scanning their bytes.
    #[inline]
    pub fn zero_fill_clears() -> bool {
        get(66) != 0
    }

    /// A GPU queue's FLIP packet is queued at once and shown after the GPU
    /// work before it (a deferred effect: `videoout::reserve_flip`,
    /// `commit_flip`) instead of the command processor waiting for that work
    /// (~0.7 ms per flip on the graphics command processor).
    #[inline]
    pub fn flip_async() -> bool {
        get(67) != 0
    }

    /// A command processor blocked on a label wait prepares the stages of
    /// the commands after it meanwhile (`Cp::scan_through`,
    /// `ahead::prepare_here`; compute queue 0x50 waits ~40 ms per frame on
    /// graphics labels, then records ~800 commands the graphics command
    /// processor waits for), checked when used as any preparation made
    /// ahead (`ahead::take`): 1 the compute queues' command processors, 2
    /// the graphics one too (it wakes later from its waits, ~0.3 ms).
    #[inline]
    pub fn wait_ahead_for(compute: bool) -> bool {
        match get(68) {
            0 => false,
            1 => compute,
            _ => true,
        }
    }

    /// The batch-start copy cache takes write-watch marks without holding
    /// its lock during the query (`precache::take_unlocked`; 0: under the
    /// lock, as before 2026-10-15).
    #[inline]
    pub fn marks_unlocked() -> bool {
        get(69) != 0
    }

    /// Bit 0: writing dispatches skip the image cache's lock when no page
    /// they write holds a GPU-owned image or fast-clear metadata
    /// (`images::may_note`); bit 1: the write logs' lookups share their lock
    /// (`exec::RangeLog`, a reader-writer lock; else taken exclusively).
    #[inline]
    pub fn note_filter() -> bool {
        get(70) & 1 != 0
    }

    #[inline]
    pub fn shared_log_reads() -> bool {
        get(70) & 2 != 0
    }

    /// Stage preparations take their vectors from a per-thread pool that
    /// finished preparations refill (`shaders::pool`; 0: allocate each).
    #[inline]
    pub fn prep_pool() -> bool {
        get(71) != 0
    }


    /// Write-mark takes query first without resetting (one page at most):
    /// a range found clean needs no resetting query (a query that resets
    /// nothing costs 0.4-0.8 us, the game's takes average 3.8 us; resets
    /// interrupt the process's other threads with TLB shootdowns). Live A/B
    /// 2026-10-15 night: +1.5% flips/s, -1% CPU per command; default 1.
    #[inline]
    pub fn ww_probe() -> bool {
        get(72) != 0
    }

    /// The graphics command processor keeps its batch open at the end of a
    /// title command buffer when another follows at once (more buffers in
    /// the submission, or the next submission queued): ~22 of its ~75
    /// batches per frame end there, each a batch begin (~60 us with slot
    /// waits) and a costlier first command (~100 us) more. Live A/B
    /// 2026-10-15 late night: +3.8% flips/s (1.023-1.050); default 1.
    /// 2 (default): the compute queues too (+1.4% flips/s, -1.3% CPU per
    /// command against 1 in a live A/B).
    #[inline]
    pub fn cmdbuf_merge() -> u64 {
        get(73)
    }

    /// Spins (`pause`) of a parallel run's waits (a worker for its chunk's
    /// turn, the main command processor for a chunk) before they block on
    /// the run's condition variable (512 since 2026-10-16, 4,096 before:
    /// -5% CPU per command with the front end on one CCD).
    #[inline]
    pub fn par_spins() -> u64 {
        get(74)
    }

    /// Batch slots of the graphics queue's command processor (`slots` for
    /// the others): its batches wait for a slot when the GPU is behind. 8
    /// against 4: level until 2026-11-12 (TRIED.md); 8 since then, once the
    /// SGPR rings moved to video memory (Castlewall Front x1.008, the swamp
    /// x1.017, FINDINGS.md 2026-11-12 midday).
    #[inline]
    pub fn slots_gfx() -> u64 {
        get(75)
    }

    /// Diagnostics: the graphics queue's command processor ends its batch
    /// after every N commands it records serially (0: off): the marginal
    /// cost of a batch boundary (`cp::SPLITS`; the thread CPU window line).
    #[inline]
    pub fn split_every() -> u64 {
        get(76)
    }

    /// Diagnostics: the parallel-run planner also computes each chunk's
    /// start state a register at a time (`Cp::apply_state` for every packet)
    /// and logs where it differs from the one-copy register ranges
    /// (`put_regs`).
    #[inline]
    pub fn plan_check() -> bool {
        get(77) != 0
    }

    /// Every video-memory home made from now on is write-watched like the
    /// indirect-argument homes: once per synchronization epoch the pages the
    /// CPU wrote are compared word by word with what the home last took in,
    /// and the words guest code changed are copied in before its next GPU
    /// use (logged). Without it (0; the default until 2026-10-17) a home
    /// sees only the CPU-side writes we make (CP writes, file reads): guest
    /// code writing into its range is not seen and GPU work keeps reading
    /// the home's older copy (level loads writing the new area's meshes into
    /// a homed buffer: the load-time fault class, misplaced and stretched
    /// meshes; -1.8% at the gate).
    #[inline]
    pub fn homes_watch() -> bool {
        get(78) != 0
    }


    /// A GPU-owned image overwritten through a buffer is reloaded from guest
    /// memory after the GPU-written parts of video-memory homes over it are
    /// written back (0: as before, for A/B: the reload reads older bytes
    /// where the writers bound a home).
    #[inline]
    pub fn image_homes() -> bool {
        get(79) != 0
    }

    /// Draws with neither color nor depth attachments whose shaders write
    /// memory (buffer or image stores) are recorded in a rendering instance
    /// without attachments, its area limited by the draw's scissors (0:
    /// skipped, as before 2026-10-16: one each per recording thread in every
    /// session, `draw with only memory writes`).
    #[inline]
    pub fn attachless_draws() -> bool {
        get(80) != 0
    }

    /// Block-compressed textures that GPU work transcodes: the mip tail the
    /// title stores through a view of the tail block alone (a single-level
    /// view one swizzle block in size at the surface's base) is copied into
    /// the tail mips (0: only the view of the whole chain is copied, as
    /// before 2026-10-16: the tail mips kept whatever that view held).
    #[inline]
    pub fn block_tails() -> bool {
        get(81) != 0
    }

    /// Storage images whose programs select the mip level they store to or
    /// load from (image_store_mip / image_load_mip: the title's mip-chain
    /// generators, several levels per dispatch) are bound one view per
    /// level (`rdna::translate::Spec::mip_views`; 0: one view of the T#'s
    /// base level that receives every level, as before 2026-10-16: the
    /// character creator's face, generated mips over the base level).
    /// Switching translates programs anew.
    #[inline]
    pub fn mip_views() -> bool {
        get(82) != 0
    }

    /// Worker threads that compile the pipelines of the commands after a
    /// first use while its command processor compiles its own
    /// (`gpu::compile`; 0: each when execution reaches it. Its timing made
    /// World 8-2's homes race spoil the load until `homes_watch` closed
    /// it). By default (`COMPILE_AHEAD_AUTO`, since 2026-10-17 morning) a
    /// third of the logical CPUs, 2 to 12 (8 on the 24 of the development
    /// machine): first sessions reached the courtyard after 71-73 s with 8
    /// and 70 s with 12 against 77-81 s with 4 (FINDINGS.md 2026-10-17).
    #[inline]
    pub fn compile_ahead() -> u64 {
        match get(83) {
            COMPILE_AHEAD_AUTO => {
                static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                *N.get_or_init(|| (std::thread::available_parallelism().map_or(6, |n| n.get()) as u64 / 3).clamp(2, 12))
            }
            n => n,
        }
    }

    /// `compile_ahead`'s default: workers by the logical CPUs.
    pub const COMPILE_AHEAD_AUTO: u64 = 99;

    /// A batch holding GPU-side label writes is submitted once it has
    /// recorded this long, us (CPU threads of the title may poll them;
    /// another queue that waits for one asks at once).
    #[inline]
    pub fn label_latency() -> std::time::Duration {
        std::time::Duration::from_micros(get(84))
    }

    /// Pipelines made ahead from learned specs (`gpu::precompile`): 1 learns
    /// each compute program's specs at its first translations and each
    /// graphics pipeline's stages and state, with the programs' code (and
    /// compares the guess from a compute object, statistics); 2 (default
    /// since 2026-10-17) also makes the learned pipelines on
    /// `precompile_workers` threads at the lowest priority, from the start
    /// of a session or when the title creates the programs' objects; 3 a
    /// compute object's guess too; 0 off.
    #[inline]
    pub fn precompile() -> u64 {
        get(85)
    }

    /// Threads that make pipelines from shader objects (`precompile` 2; at
    /// the lowest priority). 0: as many as `compile_ahead`'s workers.
    #[inline]
    pub fn precompile_workers() -> u64 {
        match get(86) {
            0 => compile_ahead(),
            n => n,
        }
    }

    /// Depth attachments over memory that GPU work wrote through a color
    /// view (the half-resolution depth the title's compute shader builds
    /// and its particles test against) get that view's data copied in
    /// before their draws (`exec::sync_depth_views`; 0: they keep their own
    /// contents, as before 2026-10-18: particles showed through walls).
    #[inline]
    pub fn depth_views() -> bool {
        get(87) != 0
    }

    /// Diagnosis only (stale copies): a shared batch-start copy found clean
    /// by a write-mark take is used without another take for this many
    /// flips (0: checked once per synchronization epoch, as it must be).
    /// Bounds what the copy cache's checks cost the frame.
    #[inline]
    pub fn pre_trust() -> u64 {
        get(88)
    }

    /// An indexed indirect draw whose arguments the CPU read, following one
    /// like it with no other packet between them (the culled clusters' draws:
    /// ~3,700 of the gate frame's ~5,600 indirect draws come in such runs),
    /// is recorded as one more draw on the state the previous draw left
    /// (pipeline, descriptors, dynamic state, index buffer, rendering
    /// instance), without preparing or binding its stages again
    /// (`Exec::draw_repeat_cpu`, with `cpu_args_base`).
    #[inline]
    pub fn draw_repeat() -> bool {
        get(89) != 0
    }

    /// Indexed indirect draws whose arguments the CPU reads hand the start
    /// instance to the vertex stage as Vulkan's
    /// first instance, the stage taking that user-data word from
    /// `gl_BaseInstance` (`DrawArgs::base_inst`), instead of in its register
    /// (which changes the stage's SGPR data per draw). Draws like the one
    /// before can then be repeated (`draw_repeat`) without GPU-read
    /// arguments; stages that need the word on the CPU keep the register.
    #[inline]
    pub fn cpu_args_base() -> bool {
        get(90) != 0
    }

    /// Stages read their SGPR data at an offset they take from a push
    /// constant (`Spec::sgpr_pc`): the data binding covers the batch's whole
    /// ring and stays the same from command to command, so a command whose
    /// resources equal the one before pushes no descriptors. Read when
    /// stages are specialized: set it at boot (`DS_SGPR_PC`; switched later,
    /// every stage is translated and its pipelines made again). Off: on
    /// the RTX 4090 driver it pushed 17% fewer descriptors per push yet
    /// cost 1.8% more CPU per command (FINDINGS.md 2026-10-19).
    #[inline]
    pub fn sgpr_pc() -> bool {
        get(91) != 0
    }

    /// A zero fill (the fill kernel over a whole range) of 256 KiB or more
    /// whose range is all the memory of GPU-written images it clears
    /// (`Images::fill_covered`) clears only the images: their guest memory
    /// is not kept current, and the store across PCIe cost the GPU ~0.7 ms
    /// per gate frame (an 18 MB layered render target each frame).
    #[inline]
    pub fn fill_skip() -> bool {
        get(92) != 0
    }

    /// Images are evicted when video memory runs short (`images::maybe_evict`):
    /// once this process uses more than `vram_high` percent of the device-local
    /// heap's budget (VK_EXT_memory_budget), the least recently used images
    /// go until it is under `vram_low` percent. 0: images are kept for the
    /// whole session (the earlier behavior: six level loads filled a 24 GB
    /// GPU, 2026-10-19).
    #[inline]
    pub fn image_evict() -> bool {
        get(93) != 0
    }

    #[inline]
    pub fn vram_high() -> u64 {
        get(94).clamp(10, 100)
    }

    #[inline]
    pub fn vram_low() -> u64 {
        get(95).clamp(5, 100)
    }

    /// Measurement only: draws (bit 0) and dispatches (bit 1) are not
    /// recorded while the command processors still execute their packets,
    /// labels and writes: the title's own frame without our GPU work. Wrong
    /// pictures and GPU-written
    /// data the title reads stay stale.
    #[inline]
    pub fn skip_cmds() -> u64 {
        get(96)
    }

    /// Measurement only (unsafe): the graphics CP skips its waits for labels
    /// compute queue 0x50 wrote last (`Cp::wait`; 1); queue 0x50 skips its
    /// waits for labels other queues' GPU work wrote last (2).
    #[inline]
    pub fn xwait_skip() -> bool {
        get(97) == 1
    }

    /// `xwait_skip` 2 (see there).
    #[inline]
    pub fn xwait_skip_acb() -> bool {
        get(97) == 2
    }

    /// Consecutive direct dispatches of a program run as one dispatch
    /// (`Spec::batched`; programs seen in runs, `shaders::mark_batchable`).
    /// Needs bufferDeviceAddress (batched translations address their
    /// buffers).
    #[inline]
    pub fn batch_cs() -> bool {
        get(98) != 0 && super::vk::ctx().device_addresses
    }

    /// The title's job workers without work (`jobidle`): microseconds a
    /// worker finds nothing to run before it sleeps between looks (one such
    /// worker, the watcher, keeps looking); 0: the title's own polling.
    /// Default 50: at the gate +7.8% flips/s, -8.5% CPU per command, the
    /// process ~6 logical CPUs instead of ~17; on 8 logical CPUs ~20 flips/s
    /// instead of ~6 and no audio underruns (FINDINGS.md 2026-10-21).
    #[inline]
    pub fn job_idle() -> u64 {
        get(99)
    }

    /// Microseconds such a worker sleeps before each further look
    /// (`job_idle`).
    #[inline]
    pub fn job_sleep() -> u64 {
        get(100)
    }

    /// Microseconds one such worker at a time looks without sleeps (the
    /// watch rotates; 0: no watcher; `job_idle`).
    #[inline]
    pub fn job_watch() -> u64 {
        get(101)
    }

    /// How translations declare their SGPR data (`Spec::sgpr_ubo`): 0 a
    /// storage buffer, 1 a uniform buffer where the binding is the command's
    /// own range (else a read-only storage buffer), 2 a read-only storage
    /// buffer, 3 as 1 and batched compute translations through a uniform
    /// window of the ring (1 where uniform buffers cannot span 64 KiB).
    /// Takes effect for translations made after a change.
    #[inline]
    pub fn sgpr_ubo() -> u8 {
        let v = get(102).min(3) as u8;
        if v == 3 && super::vk::ctx().limits.max_uniform_buffer_range < 4 * rdna::translate::SGPR_WINDOW_DWORDS {
            1
        } else {
            v
        }
    }

    /// The process opts out of Windows' power throttling (execution speed
    /// and the timer-resolution rule for hidden or occluded windows) and
    /// raises the timer resolution to 1 ms (`thread::apply_power_optout`,
    /// applied at start and when `tune power_optout` changes).
    #[inline]
    pub fn power_optout() -> bool {
        get(103) != 0
    }

    /// Flips only at the emulated vblanks (59.94 Hz: a frame that misses one
    /// waits up to 16.7 ms more). 0: a ready flip is shown at the vblank
    /// thread's next 1 ms tick once the title's flip rate allows it (as many
    /// vblank periods since the last flip as the rate asks), while the
    /// title still sees its vblanks at 59.94 Hz (`hle::libs::videoout`).
    #[inline]
    pub fn flip_grid() -> bool {
        get(104) != 0
    }

    /// The presenter keeps the frame's aspect and halves a frame more than
    /// twice the window's size before its last blit (`gpu::vk::present`). 0:
    /// one blit stretched over the whole window (comparison).
    #[inline]
    pub fn present_fit() -> bool {
        get(105) != 0
    }

    /// A frame-rate cap in frames per second (0: none): flips are shown at
    /// least 1/cap apart, on the vblank grid a whole number of vblanks
    /// (`hle::libs::videoout`; 30 -> every 2nd vblank at 59.94 Hz, 20 -> 3rd).
    /// The title's game time follows real time, so its speed is kept.
    #[inline]
    pub fn frame_cap() -> u64 {
        let cap = get(106);
        #[cfg(target_os = "linux")]
        {
            // Relay safety ceiling survives self-play changing the live frame_cap.
            static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let limit = *LIMIT.get_or_init(|| std::env::var("DS_RELAY_FPS_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(0));
            if limit > 0 { return if cap == 0 { limit } else { cap.min(limit) }; }
        }
        cap
    }

    /// The texture scrubber's sleep between passes in write-watch mode
    /// (`tex_scrub` 2), ms: each pass takes the marks of the next 64
    /// uploaded textures (diagnosis of its queries' cost to the others).
    #[inline]
    pub fn scrub_ms() -> u64 {
        get(107).clamp(1, 60_000)
    }

    /// Diagnosis (timing only, wrong results; for replay A/Bs of what a
    /// step costs): bit 0 skips draws' and dispatches' texture syncs, bit 1
    /// draws' image-hazard checks (fast clears, overwrites, block copies,
    /// depth views are not resolved).
    #[inline]
    pub fn skip_steps() -> u64 {
        get(108)
    }

    /// Draws whose pixel shader exports the stencil test value (the device
    /// has no stencil export): 0 use the register's test value as the
    /// reference, 1 use 0 (diagnosis: what the marking those draws write
    /// changes on screen).
    #[inline]
    pub fn stencil_export() -> u64 {
        get(109)
    }

    /// Smallest written-buffer shadow whose working copy is shared after its
    /// merge (`shadow_share`), bytes (the copy cache's own floor is
    /// `pre_share_min`).
    #[inline]
    pub fn shadow_share_min() -> u64 {
        get(110)
    }

    /// The graphics command processor records past its parallel runs
    /// before collecting their chunks (`Cp::past_start`): its batches after
    /// a run held until a stop (at the latest its submission's end), then
    /// submitted after the chunks' or dropped and recorded again (default;
    /// +0.8% at the gate, FINDINGS.md 2026-10-23); 0 off.
    #[inline]
    pub fn past_runs() -> bool {
        get(111) != 0
    }

    /// Diagnosis: what it recorded past a run is always dropped (the path
    /// that records it again, tested).
    #[inline]
    pub fn past_drop() -> bool {
        get(112) != 0
    }

    /// Recording past a parallel run goes on into the title's next
    /// submission when it is already queued, as past a label wait, instead
    /// of stopping at the end of the run's submission (2026-11-11: 73% of
    /// the stops past runs at the walk's Castlewall Front, the CP then
    /// waiting ~179 us for the chunks with the GPU mostly idle; in-run A/B
    /// x1.0046, five of five cycles ahead, FINDINGS.md 2026-11-11); 0 off.
    #[inline]
    pub fn past_subs() -> bool {
        get(165) != 0
    }

    /// Workers each compute queue's command processor records its runs'
    /// chunks with under front end 2 (`cp::compute_workers`; live: a pool
    /// grows to it at the next run), 0 none; 1000 (the default since
    /// 2026-11-11): as many as the graphics queue's (`par::workers`: half
    /// the logical CPUs, 2 to 8; until then a fixed 4). On 24 logical CPUs
    /// 8 against 4: Castlewall Front x1.038, where queue 0x50 records the
    /// title's GPU particles in one long run a frame on the critical path,
    /// the vista x1.031; 12 level (FINDINGS.md).
    #[inline]
    pub fn par_compute() -> u64 {
        get(166)
    }

    /// Batch fusion (`exec::submit_specs`, with `submit_coalesce` above 1):
    /// consecutive plain batches the submission thread drains together are
    /// played into the first one's command buffer (each prologue before its
    /// commands) and handed to Vulkan as one submission signalling the last
    /// one's sequence: no submission boundary between them on the GPU (~22
    /// us of idle a batch start at Castlewall Front, FINDINGS.md 2026-11-11).
    /// 2: the same, but with the main queue idle a group's first batch is
    /// submitted alone at once (the GPU does not wait for the others'
    /// replay). Live; default 2 since 2026-11-11 (fusion: Castlewall Front
    /// x1.027, the swamp x1.010 in-run; mode 2 against 1 x1.004 at
    /// Castlewall Front; the 1-1 walk x1.015-1.019 against none in
    /// alternating boots; FINDINGS.md).
    #[inline]
    pub fn submit_fuse() -> bool {
        get(167) != 0
    }

    /// `submit_fuse`'s mode (0, 1, 2).
    #[inline]
    pub fn submit_fuse_mode() -> u64 {
        get(167)
    }

    /// The armer behind the graphics CP and queue 0x50 (`fe2::behind_thread`)
    /// arms pages quiet for front end 2's quiet time shifted left by this
    /// (default 2: four times as long). Live.
    pub fn behind_quiet() -> u64 {
        get(168).min(8)
    }

    /// The armer's time a round (2 ms apart) in us (default 300). Live.
    pub fn behind_us() -> u64 {
        get(169)
    }

    /// Graphics pipelines missing at a draw compiled in the background
    /// (`exec::draw` `async_gfx`): 1 (default) a draw that may be left out
    /// meanwhile is (no memory writes, no depth or stencil clear, every target
    /// it writes drawn into in 3 of the 4 frames before); 0 every draw waits
    /// for its pipeline (as before 2026-11-13). Live.
    #[inline]
    pub fn async_gfx() -> u64 {
        get(170)
    }

    /// Threads that compile `async_gfx`'s pipelines (normal priority); 0: a
    /// third of the logical CPUs, at least 2 (a sixth before 2026-11-18: a
    /// new player's first 1-1 walk at 8 against 4 left 9,326 draws out
    /// against 25,678, a pipeline made 252 ms after its queueing against
    /// 755 (at most 2.1 against 4.2 s), its frames x0.976 while it compiled).
    pub fn async_gfx_workers() -> usize {
        match get(171) {
            0 => (std::thread::available_parallelism().map_or(8, |n| n.get()) / 3).max(2),
            n => (n as usize).min(16),
        }
    }

    /// Parallel runs inside a stretch recorded past a label wait (`Cp::past_wait_start`): the
    /// graphics command processor plans them there as elsewhere (runs of `held_min` commands or
    /// more), its workers record the chunks held (submitted when the stretch commits, in program
    /// order: `Cp::held_collect`; a run still being collected when it settles goes on unheld,
    /// each chunk checked as the stretch was: `Cp::unheld_collect`), and the stretch's parts
    /// share front end 2's uploads from its first held run (`Exec::fe2_share`). Since
    /// 2026-11-13 (TowerL Top x1.006-1.009, the CP -0.5-0.65 ms a flip; Castlewall Front level;
    /// FINDINGS.md). 0: a stretch is recorded by the command processor alone (as before).
    #[inline]
    pub fn held_runs() -> bool {
        get(172) != 0
    }

    /// Fast clears of a layered depth surface whose HTILE is laid out a layer after another clear
    /// the layers its metadata writes reach (`images::meta_layer_bytes`); 0: the whole image, as
    /// before 2026-11-19 (a fill of the shadow atlas's layer 0 cleared all 80 layers).
    #[inline]
    pub fn layer_clears() -> bool {
        get(188) != 0
    }

    /// The checks of threads that arm as they take (all but the graphics CP's and queue
    /// 0x50's) arm runs of at most this many pages themselves and leave longer ones to the
    /// armer behind them (`fe2::behind_thread`); 0: every run themselves. 1024 the default since
    /// 2026-11-21 (the walk x1.065, arrivals' p99 -13%; its extra arrival flashes were the
    /// view-mask walks', gone with `maskwait sub`: FINDINGS.md 2026-11-21 afternoon). Live.
    #[inline]
    pub fn arm_inline_max() -> u64 {
        get(189)
    }

    /// A stretch past a label wait whose check at its label (`Exec::past_current_for`) finds
    /// pages written since its own checks uploads them in a batch of their own, submitted before
    /// the stretch's (its copies run after the label, from guest memory as it is then:
    /// `Exec::past_repair`), instead of being dropped and recorded again. Since 2026-11-13 (held
    /// runs' chunks check earlier: their stale drops and the backoff they caused, ~200 waits a
    /// window not gone past at TowerL Top, went). 0: dropped (as before).
    #[inline]
    pub fn past_repair() -> bool {
        get(173) != 0
    }

    /// The fewest commands a run planned inside a stretch past a label wait is held at (`tune
    /// held_runs`): shorter ones are recorded by the command processor as before (Castlewall
    /// Front's ~96-command runs cost it a wait per chunk for no time saved).
    #[inline]
    pub fn held_min() -> usize {
        get(174) as usize
    }

    /// Graphics pipelines the store holds made at their first draw, up to this many ms of a
    /// frame's first uses (`exec::draw`; the rest are left out while a worker makes them, as
    /// `async_gfx` does with every pipeline); 0: every one left out.
    pub fn async_gfx_inline() -> u64 {
        get(175)
    }

    /// Zero fills' stores into guest memory left out where guest memory holds zeros already
    /// (`gpu::zk`; with the ordered compute queue on, its batches store their fills whole and
    /// only the main queue's zero stores set bits: `Exec::fill_in_place_parts`).
    pub fn zk() -> bool {
        get(176) != 0
    }

    /// Ordered async compute: a compute batch's prologue (its uploads into the mirror) on the
    /// compute family's queue right before it instead of on the main queue (`Exec::begin_batch`;
    /// live at batch boundaries): no main-queue wait for an earlier compute batch whose writes
    /// its uploads would undo; its uploads count for other batches once it completed, and the
    /// main queue's later batches that write into its pages wait for it (`exec::XC_UPLOADS`).
    /// The default since 2026-11-16: Doran x1.034 at 240 Hz, TowerL Top x1.010 at 59.94 Hz, the
    /// swamp x1.005, every cycle (FINDINGS.md 2026-11-16).
    pub fn xq_pre_c() -> bool {
        get(177) != 0
    }

    /// The lean lane (ROUTE60.md "2026-11-16 midday"): stage preparations through compiled
    /// plans (`shaders::lean`), with the full path for what they leave; `DS_LEAN_CHECK=1`
    /// prepares both ways and compares. Live (each preparation reads it). The default since
    /// 2026-11-16 evening (exact by the compare mode; preparations 17.5% cheaper); `DS_LEAN=0`
    /// is the diagnosis fallback.
    #[inline]
    pub fn lean() -> bool {
        get(178) != 0
    }

    /// The lean lane's run-time parts (2026-11-16 night; the preparations are `lean`): front end
    /// 2's page memo (`exec::PageMemo`: a range on pages the executor's batch placed under the
    /// same epoch, wanted state and mappings is placed without front end 2's check), its lean
    /// check of small read-only ranges (`Exec::fe2_place_quick`) and the lean shader-read sync
    /// (`Exec::sync_quick`). `DS_LANE=0` the fallback; `DS_PAGE_MEMO_CHECK=1` places every memo
    /// and lean-check answer again in full and compares.
    pub fn lane() -> bool {
        get(179) != 0
    }

    /// Runs of a batched program as data (2026-11-17; ROUTE60.md's first section): a later
    /// command of an open batched run whose resources are the run's but for read-only buffers
    /// binds those alone and takes the rest of its SGPR data from the run's first command
    /// (`Exec::join_try`). 1: on; 2: the check (the full path records every command, and the
    /// block the join would have written is compared with its own); 3: the form without the
    /// full preparation checked against the one after it; 4: only the form after it; 5: the
    /// lean binding of a joined command's buffers (`Exec::join_bind_lean`) checked against
    /// `bind_stage`'s general arm, whose result is used; 6: the general arm alone; 0: off.
    pub fn join() -> u64 {
        get(180)
    }

    /// The copy kernel's tiny read-only inputs (its pattern and its control block, at most
    /// `exec::TINY_SNAP_BYTES`: CPU-written, as its forwarding takes them) are bound from a
    /// snapshot of guest memory taken when the dispatch is recorded, not from front end 2's
    /// mirror (a stale pattern there gave whole batches of draw records garbage counts:
    /// FINDINGS.md 2026-11-17 afternoon). 0: from the mirror like every other buffer.
    pub fn copy_snap() -> bool {
        get(181) != 0
    }

    /// Whether a run without a window presents a small image a flip to a swapchain that is
    /// never shown (`vk::present::headless_present`): the driver's power management then
    /// runs the GPU as it does for a run with a window. 0: nothing is presented.
    pub fn headless_present() -> bool {
        get(182) != 0
    }

    /// The game's step per frame (`gamestep`): 1 the port sets it from the frames shown (the
    /// time between the flips, smoothed over `pace_smooth` frames: the same game speed at
    /// every frame rate, the step exact while every frame is shown as long as the last); 0
    /// the title's own (1/59.94 s at least and in halves of it: fast above 60 frames a
    /// second).
    pub fn pace_step() -> bool {
        get(183) != 0
    }

    /// Over how many frames the game's step is the mean of the times between flips (1 to 32;
    /// 1: each frame steps by the time the frame before it was shown).
    pub fn pace_smooth() -> u64 {
        get(184).clamp(1, 32)
    }

    /// The physics' fixed step while the port sets the game's step: 0 the title's (1/59.94
    /// s), 1 the frame limit's time a frame (one physics step a frame while the limit is
    /// held; uncapped: the game's step, at most 1/30 s), 2 the game's step every frame (at
    /// most 1/30 s).
    pub fn pace_phys() -> u64 {
        get(185)
    }

    /// Whether the window's display is taken as a variable-refresh one (G-Sync, FreeSync):
    /// 2 as the driver or the display clock says (`vrr`), 1 always, 0 never.
    pub fn vrr() -> u64 {
        get(186)
    }

    /// On the display's grid, the title's vblank ticks this many microseconds before the
    /// display's vblank it is due at (a flip presented then can be shown at that vblank, a
    /// refresh sooner, where presents go straight to the screen; PACING.md "Open" 3); 0 (the
    /// default): 300 us after it.
    pub fn tick_lead() -> u64 {
        get(187)
    }

    /// Whether translations get `rdna::translate::Spec::hw_bounds` (scalar
    /// loads of buffers only they read without a check of each dword): where
    /// robustBufferAccess2 and null descriptors are enabled (since 2026-11-12
    /// night; FINDINGS.md).
    #[inline]
    pub fn hw_bounds_on() -> bool {
        super::vk::ctx().robust2_null
    }

    /// The title's convolution reverb computed (`hle::libs::acm_conv`,
    /// default); 0: its jobs left undone (no reverb, the stand-in before
    /// 2026-10-23).
    #[inline]
    pub fn acm_reverb() -> bool {
        get(113) != 0
    }

    /// The graphics CP's label waits of 1 ms or more still to be logged in
    /// detail (`tune trace_waits <n>`: the next n, wherever they happen;
    /// `DS_TRACE_WAITS=<queue>` logs a queue's first 200 from the start).
    pub fn take_trace_waits() -> bool {
        get(114) > 0 && count_down("trace_waits")
    }

    /// The graphics CP's planning of parallel runs without its costliest
    /// steps (2026-10-23, a vista's profile): indirect-argument ranges
    /// merged as they come instead of sorted, run costs pruned once a frame,
    /// the instance-offset slot's last answer reused; 0: as before (A/B).
    #[inline]
    pub fn plan_fast() -> bool {
        get(115) != 0
    }

    /// Programs compiled in the background at their first use, their
    /// dispatches left out until then, instead of the command processor
    /// stopping for the compile (`exec::compile_in_background`): 2 the
    /// engine programs of per-frame effects named in `DS_ASYNC_ENGINE`
    /// (the particles' lighting atlas passes), 1 also the title's particle
    /// programs (`particles::in_background`), 0 none. Particle programs
    /// left out broke their effects' state in first sessions (an effect's
    /// other programs ran on it: shaders looping until the loop watchdog
    /// stopped them, 43,024 times in a walk of 1-2, and GPU work of 100-190
    /// ms a frame for minutes; FINDINGS.md 2026-10-24).
    #[inline]
    pub fn async_particles() -> u64 {
        get(116)
    }

    /// A dispatch's new VRAM shadows get the guest data in the batch's
    /// prologue (before all its work) when nothing earlier in the batch
    /// writes their window, instead of by copies and two barriers before
    /// the dispatch: those barriers drained the GPU before every such
    /// dispatch, so the title's particle chain (~1,000 small dispatches a
    /// frame, each writing small buffers of its own) ran one dispatch at a
    /// time (`--dispatch-bench`: 256 dispatches overlap in 61 us, 6 ms with
    /// a barrier between). 0: as before.
    #[inline]
    pub fn shadow_pre() -> bool {
        get(117) != 0
    }

    /// The fewest bytes a read-only binding inside a range GPU work writes
    /// through large bindings needs to give that range a home (`home_reads`;
    /// default 1 MiB; 0: over `pre_max`). GPU-written buffers of 4-8 MiB that
    /// later batches bind read-only were written in place across PCIe and
    /// copied back for every reading batch (a vista: ~54 MB a frame).
    #[inline]
    pub fn home_read_min() -> u64 {
        match get(118) {
            0 => pre_max() + 1,
            v => v,
        }
    }

    /// The largest written buffer binding given a VRAM shadow (`Exec`: a
    /// working and a pristine copy in the slot's shadow segment, merged back
    /// at its batch's end, its working copy then shared with later readers,
    /// `tune shadow_share`); larger ones are written in place across PCIe and
    /// copied back for every batch that reads them (`exec::SHADOW_MAX`, at
    /// most 12 MiB: both copies fit a segment of `SHADOW_BYTES`).
    #[inline]
    pub fn shadow_max() -> u64 {
        get(119).min(12 << 20)
    }

    /// Logical CPUs the compute queues' command processors run on (`prio`):
    /// 0 those of the front end (`cp_cpus`), 1 (default) the title's
    /// (`guest_cpus` AUTO: its last-level-cache group has cores to spare
    /// while the front end's runs the graphics command processor and its
    /// workers; +1.7% at 1-1's vista, FINDINGS.md 2026-10-26; nothing
    /// changes where there is one such group), else a decimal mask.
    #[inline]
    pub fn acb_cpus() -> u64 {
        get(120)
    }

    /// Diagnostics, hosted runs (`hostcensus`): bit 1 arms shared copies
    /// clean for 30 flips read-only, bit 2 GPU-written windows of up to 4 MiB
    /// no-access once their batch completed; the host's faults on them are
    /// counted (observation only).
    #[inline]
    pub fn host_census() -> u64 {
        get(121)
    }

    /// Diagnostics (`placement`): sampled logical CPUs of the command
    /// processors and the title's job workers, by physical core.
    #[inline]
    pub fn placement() -> bool {
        get(122) != 0
    }

    /// Logical CPUs (a decimal mask) of queue 0x50's command processor
    /// alone; 0: as the other compute queues (`acb_cpus`).
    #[inline]
    pub fn acb50_cpus() -> u64 {
        get(123)
    }

    /// A write-watched home's check for CPU writes queries its pages with
    /// the homes' lock released (`homes::check_before_lock`; 0: under the
    /// lock, as before).
    #[inline]
    pub fn home_check_unlocked() -> bool {
        get(124) != 0
    }

    /// A write-watched home's check for CPU writes covers the 2 MiB windows
    /// of the binding that needs it (`homes::CHECK_WIN`; 0: the whole home
    /// at once, as before).
    #[inline]
    pub fn home_check_win() -> bool {
        get(125) != 0
    }

    /// GPU-resolved chains (RENDERER.md "GPU-resolved item passes"): 0 off,
    /// 1 the guest page table is kept and programs found eligible resolve
    /// their per-item chains in the shader (batched compute programs and draw
    /// pairs), 2 compute programs only (front end 2's stage F3: the draws'
    /// runs cost more than they save, FINDINGS.md 2026-10-28), 3 draw pairs
    /// only (since 2026-11-18: 1 hung the GPU twice under front end 2 with the
    /// item culls GPU-resolved, FINDINGS.md 2026-11-18).
    pub fn gpu_chains() -> u64 {
        get(126)
    }

    /// A forwarded copy's entry also accepts, as words memory holds before
    /// it runs, those of the earlier copies into the same destination it
    /// replaced (`forward::Entry::earlier`); 0: only the destination's words
    /// when it was recorded (a reader that met an earlier copy's word ended
    /// the entry as the title's rewrite).
    #[inline]
    pub fn forward_earlier() -> bool {
        get(127) != 0
    }

    /// The graphics CP forwards the per-view copies it records past a
    /// parallel run too, into a view of its own until the stretch commits
    /// (`forward::hold_begin`); 0: not forwarded there (readers after them
    /// waited for the GPU).
    #[inline]
    pub fn forward_held() -> bool {
        get(128) != 0
    }

    /// While most of a label's recent stretches past its waits were
    /// dropped (`Cp::past_wait_ok`; each outcome weighs a fourth), only every n-th
    /// wait on it is gone past (a probe); 0: every one.
    #[inline]
    pub fn past_backoff() -> u64 {
        get(129)
    }

    /// Batches the submission thread takes at once when that many are queued
    /// (`exec::submit_specs`): fused into one submission with `submit_fuse`
    /// (the default 16 since 2026-11-11), else handed to Vulkan in one call
    /// (alone that measured x0.985-0.989, TRIED.md).
    #[inline]
    pub fn submit_coalesce() -> u64 {
        get(130)
    }

    /// Front end 2: at its label a stretch past a label wait whose checks
    /// needed an older epoch than its label does (the label's writer
    /// synchronized after them) is checked again as of that epoch, and
    /// dropped and recorded again when a page it bound was written since
    /// (`Exec::past_current_for`; 0: kept unchecked, as before 2026-10-29,
    /// when 59% of the vista's stretches bound such pages).
    pub fn past_check() -> bool {
        get(131) != 0
    }

    /// Front end 2: a stretch past a label wait uploads the pages of its
    /// small bindings written within their hot window as they are when it
    /// runs, whatever its checks found (`fe2::hot_pages`; 0: as checked).
    pub fn past_hot() -> bool {
        get(132) != 0
    }

    /// Flips a read that dropped a stretch past a label wait stops later
    /// stretches before it (doubled each time it drops one again, up to
    /// 32x); 0: for the rest of the session, as before 2026-10-29.
    pub fn past_conflict_hold() -> u64 {
        get(133)
    }

    /// The window (pages, whole 64-page words) a front end 2 take of at
    /// most 16 pages widens over the GPU-written pages around it
    /// (`fe2::widen`; 64: its word, as before 2026-10-29).
    pub fn fe2_wide_pages() -> u64 {
        get(134)
    }

    /// The shortest run the parallel planner splits whatever `par_min` and
    /// `par_min_compute` say (`par::floor`; 48 as before 2026-10-29).
    pub fn par_floor() -> u64 {
        get(135)
    }

    /// Timing only (a title thread that sees a label may read data older
    /// than it): front end 2 records no write-back before the CP writes the
    /// GPU performs; what the batch's work wrote reaches guest memory at its
    /// close (the upper bound of write-backs only where the CPU reads).
    pub fn fe2_cpw_close() -> bool {
        get(136) != 0
    }

    /// Front end 2 (default since 2026-10-29: +2.5% at the vista over four
    /// rounds, queue 0x50's GPU time -0.7 to -1.0 ms a frame; tour clean):
    /// no write-back before a CP
    /// write the GPU performs to a label one of our command processors waits
    /// on (`cp::cp_waited`: GPU-to-GPU synchronization; those threads read
    /// GPU-written data on the CPU only once the writer's batch is done, its
    /// close's write-back included); labels no CP waits on keep theirs (a
    /// title thread may poll them). Left open: a title thread polling a label
    /// our CPs wait on too, and reading data before the batch completes.
    pub fn fe2_cpw_waited() -> bool {
        get(137) != 0
    }

    /// Diagnosis: ns of spin at each draw and dispatch of the executors
    /// `ballast_at` names (`exec::ballast`; live).
    #[inline]
    pub fn ballast_ns() -> u64 {
        get(148)
    }

    /// Diagnosis: the executor kinds `ballast_ns` spins in (bit 0 the
    /// graphics CP, 1 its workers, 2 queue 0x50, 3 its workers, 4 the other
    /// compute queues; the graphics CP's parts of its stream alone: 5 past
    /// a label wait, 6 past a parallel run, 7 its own chunk of a run, 8 the
    /// rest; live).
    #[inline]
    pub fn ballast_at() -> u64 {
        get(149)
    }

    /// Ordered async compute (RENDERER.md "Ordered async compute"; the
    /// default from 2026-11-01 to 2026-11-14 and from 2026-11-16 to 11-18): 2, the compute
    /// queues' batches on a queue of a compute-only family (set up when the value is 2 or more
    /// at boot: `vk::async_compute_boot`; live at batch boundaries; x1.013-1.046 in-run,
    /// FINDINGS.md 2026-11-16); 0 (the default again since 2026-11-18: with 2 a boot without
    /// prepared shaders lost the device, a compute batch writing at address 0; FINDINGS.md
    /// 2026-11-18 night), every batch on the main queue; 1, one
    /// queue with a census of the waits
    /// the main queue's batches would need, by reason, counted and marked in
    /// the timeline (`after #seq (xq ...)`, `tools/schedmodel.py`).
    #[inline]
    pub fn async_compute() -> u64 {
        get(150)
    }

    /// Image eviction (`images::maybe_evict`) counts other processes'
    /// dedicated video memory (`hostload`, sampled once a second): it starts
    /// once ours passes the heap's size less theirs and this many MiB, and
    /// goes to a GiB under that (or the budget's `vram_high` / `vram_low`
    /// shares, if lower); 0: the budget alone, as before 2026-11-02.
    #[inline]
    pub fn vram_others() -> u64 {
        get(151)
    }

    /// Front end 2's mirror words no placement covered in this many flips
    /// are unbound and their memory reused (`resident::unback_cold`, every
    /// 5 s on the completion thread); 0: never. Live.
    #[inline]
    pub fn mirror_cold() -> u64 {
        get(152)
    }

    /// Buffer-delta draws (`exec::draw`, RENDERER.md): a draw right after a
    /// full draw of the same programs, images and samplers with the same
    /// context state takes over its texture sync, targets, pipeline, image
    /// hazards, rendering instance and dynamic state, and its bound images
    /// and samplers; 0: off; 2: the full path runs and the results are
    /// compared (differences counted and logged; a diagnosis). Live.
    #[inline]
    pub fn buffer_delta() -> u64 {
        get(153)
    }

    /// LOD statistics of compute programs in rotation (as `lod_late` does
    /// for pixel programs whose depth tests stay after the shader): N >= 2,
    /// a dispatch binding a counted texture measures in one of N
    /// (`lodstats::rotate`, prepared again with them), the others are
    /// prepared without them and carry their textures' recent measurements;
    /// 0 or 1: every such dispatch measures (before 2026-11-05). Batched
    /// dispatches keep them. Default 8 (in-run A/B at the vista: +1.1%
    /// frames, FINDINGS.md 2026-11-05 night). Read when a compute stage is
    /// prepared.
    #[inline]
    pub fn lod_cs() -> u64 {
        get(154)
    }

    /// Exact ballast in the translations (diagnosis: what a form costs live,
    /// its result unchanged): bit 0, every store's write marks done twice
    /// (`rdna::translate::Spec::mark_twice`); bit 1, two dependent loads
    /// before each stage's user SGPRs (`Spec::dep_loads`: descriptors fetched
    /// by the shaders at their start); bit 2, each store's page-touch test
    /// twice (`Spec::touch_twice`); bit 3, each store's partitioned reduction
    /// and map OR twice (`Spec::part_twice`). Read when a stage is prepared;
    /// an in-run A/B needs warm pipelines for both values.
    #[inline]
    pub fn tr_ballast() -> u64 {
        get(155)
    }

    /// Stage preparations memoized by packet during replay iterations
    /// (diagnosis: the ceiling of preparation off the recording threads;
    /// exact there, where the frame's words repeat): 1 the graphics command
    /// processor's, 2 every recording thread's.
    #[inline]
    pub fn prep_memo() -> u64 {
        get(156)
    }

    /// A dry run of a take per title submission (diagnosis,
    /// docs/FRONTEND3.md phase 1): a helper thread queries every run of the
    /// mirror's valid pages not armed after each title submission
    /// (coalesced), without resetting (`fe2::fe3_scan_dry`).
    #[inline]
    pub fn fe3_scan() -> u64 {
        get(157)
    }

    /// The uncapped frame rate (the PC tab's Frame Rate Limit, Uncapped): the vblank thread
    /// ticks at `videoout::UNCAPPED_HZ` instead of 59.94 Hz, so the title, which flips at
    /// every vblank, shows each frame as soon as it is done (the display's refresh still
    /// paces presentation: the presenter's FIFO swapchain). `DS_VBLANK_HZ`, when set, stays
    /// the rate. Live (scripts switch it in-run).
    #[inline]
    pub fn uncapped() -> bool {
        get(158) != 0
    }

    /// GPU-performed CP writes (WRITE_DATA) without the full barrier after
    /// them (front end 2): a later command, fill, CP write or timestamp store
    /// of the batch whose ranges overlap one records the barrier first; and
    /// without the one before them unless the title's synchronization is
    /// pending or the batch wrote the range since its last barrier (the
    /// console's CP does not wait for earlier work at a WRITE_DATA either).
    /// 0: a full barrier on both sides (as before 2026-11-09). Live.
    #[inline]
    pub fn cpw_lazy() -> bool {
        get(159) != 0
    }

    /// Diagnosis (live, exact: only GPU time and order added): after a
    /// write-back's barriers, one workgroup of a ballast kernel running
    /// this many steps of a dependent chain into a buffer of its own,
    /// between full barriers (0 off; `fe2::record_scatter`). Prices GPU
    /// time at write-backs' places in frames (FINDINGS.md 2026-10-31
    /// evening, 2026-11-09).
    pub fn wb_probe() -> u64 {
        get(160)
    }

    /// Where `wb_probe` adds its work: 0 after every write-back, 1 only at
    /// closes of the graphics command processor's and its workers' batches,
    /// 2 only before their GPU-performed CP writes, 3 only at the workers'
    /// chunk batches' closes (`Exec::fe2_scatter_at`).
    pub fn wb_probe_site() -> u64 {
        get(161)
    }

    /// Unbounded stage bindings (base pointers whose loads the translation
    /// cannot bound, placed up to 256 MiB) are bounded by the page
    /// allocator's record holding their base (`engine::record_extent`):
    /// front end 2 checks and places that. It assumes a shader reads only
    /// inside the engine allocation its pointer lies in; 0 places the 256
    /// MiB as before 2026-11-09 night. Live.
    #[inline]
    pub fn unb_bound() -> bool {
        get(162) != 0
    }

    /// The GPU arena's coherence passes as a census (`gpu::kco`): a helper
    /// takes the arena's CPU-written scope after each title submission and
    /// records the age of the pages it finds written (nothing consumes them).
    pub fn kco() -> bool {
        get(163) != 0
    }

    /// The recording path rebuilt (R2, ROUTE60.md 2026-11-10 afternoon):
    /// per-thread program and plan records, leaner preparation, binding and
    /// synchronization, each producing what the earlier path produces. The
    /// default since 2026-11-10 evening (the 1-1 walk x1.06-1.09 in boots,
    /// the vista level; 0: the earlier path, for A/Bs).
    #[inline(always)]
    pub fn r2() -> bool {
        get(164) != 0
    }

    /// Bumped by every switch change after boot (`set`, `set_default`):
    /// per-thread records of switch-derived state hold while it is unchanged.
    static GEN: AtomicU64 = AtomicU64::new(0);

    #[inline(always)]
    pub fn generation() -> u64 {
        GEN.load(Ordering::Acquire)
    }


    /// Pixel programs that write no memory run their depth and stencil
    /// tests before the shader where the draw's state lets them (no kill,
    /// coverage or depth export, or no depth and stencil writes:
    /// `rdna::translate::Spec::early_ok`), as the hardware does; 0: only
    /// where the title asks (DEPTH_BEFORE_SHADER), as before 2026-10-30,
    /// so the LOD statistics' atomics had every rasterized fragment shaded.
    /// Read when a pixel program is prepared.
    pub fn early_auto() -> bool {
        get(138) != 0
    }

    /// Timing only at 0 (GPU writes then never reach guest memory: the
    /// upper bound, in replays, of what the stores' write marks cost the
    /// translations, with the write-backs they feed): translations leave
    /// front end 2's write marks out. Read when a stage is prepared.
    pub fn write_marks() -> bool {
        get(139) != 0
    }

    /// The mirror's backing granule in pages of 64 KiB (a power of two; 0,
    /// the default: 16 with a device-local heap of 16 GiB or more, else 1):
    /// a binding's first use backs the aligned granule around it, in one
    /// bind with whatever other threads claimed meanwhile
    /// (`resident::back_in`).
    pub fn fe2_back_granule() -> u64 {
        get(147)
    }

    /// LOD statistics of pixel programs whose depth tests stay after the
    /// shader (a kill, coverage or export with depth or stencil writes):
    /// their atomics are side effects, so the driver shades every fragment
    /// of such draws before testing depth, hidden ones too. N >= 2 (default
    /// 8 since 2026-11-02): in rotation, one such draw in N (`lodstats`),
    /// the others prepared without them (tested early and written late, as
    /// for any program without side effects) carrying their textures'
    /// recent measurements (the same banks and finest-mip mix as every draw
    /// measuring, FINDINGS.md 2026-10-30); 1: every draw binding a counted
    /// texture measures (with `lod_skip` 0 every draw): with the camera
    /// inside 1-1's ivy one batch of alpha-tested draws then took ~30 ms
    /// instead of ~3 (frames of 67-88 ms; FINDINGS.md 2026-11-02); 0
    /// (diagnosis): none measures, so the title's texture streaming lacks
    /// those textures' statistics.
    pub fn lod_late() -> u64 {
        get(140)
    }

    /// With `lod_late` 1: draws binding no texture whose T# enables
    /// counting are prepared without LOD statistics (their atomics never
    /// ran: the same reports, without the side effects; ~96% of such draws
    /// at 1-3, -2.1% of a replayed frame's GPU time, FINDINGS.md
    /// 2026-10-30 midday); 0: every draw keeps them (the reference).
    pub fn lod_skip() -> bool {
        get(141) != 0
    }

    /// Front end 2 stage F4's first step: uploads of 2D and array textures
    /// of at least N MiB detiled on the GPU (`images::Images::upload_gpu`):
    /// the tiled bytes copied from the guest-memory import windows into
    /// video memory, a kernel (`rdna::kernels::detile`) writing them as
    /// packed rows, then copied into the image, submitted without waiting
    /// (`images::Held`): the transfer on the copy queue when there is one
    /// (the main queue's part waits on its timeline), so it runs beside the
    /// frames' work. No CPU detiling, staging or transfer waits; parts are
    /// hashed as on the CPU path. Smaller textures stay on the CPU: the
    /// title maps each texture apart, and importing them all (4,098 live
    /// imports against 605 at Castlewall) slowed every span lookup. Default
    /// 8 since 2026-10-30 (Castlewall's arrival and the 1-1 walks, FINDINGS.md
    /// 2026-10-30 evening); 0: all detiled on the CPU.
    pub fn gpu_detile() -> u64 {
        get(142)
    }

    /// Timing only at 0 (GPU writes then never reach guest memory; replays
    /// only, as `write_marks 0`): translations mark their stores' dwords
    /// but leave out the touched-page list (`Spec::marks_untouched`), so
    /// write-backs find nothing to write: with `write_marks 0` it split
    /// what the marks cost (diagnosis, 2026-10-31). Read when a stage is
    /// prepared.
    pub fn mark_touch() -> bool {
        get(143) != 0
    }

    /// Texture uploads detiled on the CPU that waited for their transfer on
    /// the main queue are recorded into one command buffer they share,
    /// submitted ahead of the next batch (`images::flush_uploads`) and not
    /// waited for (`images`; their staging freed once it completes). Read
    /// at each upload. Default 1 since 2026-10-31 (the 1-1 walk's arrivals
    /// 24% less hitch time, level loads' hitches lower, content drawn
    /// sooner; FINDINGS.md); 0 waits for each transfer as before.
    pub fn upload_batch() -> bool {
        get(144) != 0
    }

    /// Batches recorded into lists that the submission thread replays into
    /// their command buffers just before submitting them (`vk::defer`): 1
    /// the graphics command processor's, 2 the compute ones', 4 the graphics
    /// queue's parallel workers', 8 the compute queues' workers'. Read at
    /// each batch's begin. Default 5 since 2026-10-31 (1: the graphics CP's
    /// CPU -5.6% at the vista, its critical path -3.8%; 4: the workers' CPU
    /// -6.6%, +1% at the vista, Castlewall +2%; 2 lost 1.2% in replays:
    /// FINDINGS.md).
    pub fn record_defer() -> u64 {
        get(145)
    }

    /// A large zero fill under front end 2 stores into guest memory only
    /// where no GPU-written uncompressed image lies (`images::
    /// gpu_written_spans`; the mirror is filled over all of it): the bytes
    /// `fill_skip` already leaves out when such images cover a fill whole,
    /// left out of the render-target pools' clears too (an 18 MiB one a
    /// vista frame, in front of a label). Live; default 1 since 2026-10-31
    /// (the vista +3%, GPU busy -1.8 ms a frame; tour clean: FINDINGS.md).
    pub fn fill_partial() -> bool {
        get(146) != 0
    }


    /// A switch's value by name; None if unknown.
    pub fn value(name: &str) -> Option<u64> {
        ALL.iter().position(|t| t.name == name).map(get)
    }

    /// Sets a switch by name; false if unknown.
    pub fn set(name: &str, v: u64) -> bool {
        match ALL.iter().find(|t| t.name == name) {
            Some(t) => {
                t.value.store(v, Ordering::Relaxed);
                GEN.fetch_add(1, Ordering::AcqRel);
                true
            }
            None => false,
        }
    }

    /// Sets switch `name` to `v` unless its environment variable is set (a
    /// mode's own defaults, which A/Bs can still override). False: no such
    /// switch.
    pub fn set_default(name: &str, v: u64) -> bool {
        match ALL.iter().find(|t| t.name == name) {
            Some(t) => {
                if std::env::var_os(t.env).is_none() {
                    t.value.store(v, Ordering::Relaxed);
                    GEN.fetch_add(1, Ordering::AcqRel);
                }
                true
            }
            None => false,
        }
    }
}

pub mod ahead;
pub mod compbench;
pub mod dispbench;
pub mod compile;
pub mod pacing;
pub mod pagetable;
pub mod gpures;
pub mod drawcensus;
pub mod itemcensus;
pub mod particles;
pub mod forward;
pub mod hscensus;
pub mod par;
pub mod prio;
pub mod precache;
pub mod fe2;
pub mod zk;
pub mod kco;
pub mod fe2census;
pub mod precompile;
pub mod pipeuse;
pub mod lockstat;
pub mod exec;

/// Whether a GPU import may cover guest [base, base+len) (`exec`).
pub(crate) fn imports_may_cover(base: u64, len: u64) -> bool {
    exec::imports_may_cover(base, len)
}
pub mod pcache;
pub mod reconv;
pub mod trcache;
pub mod pool;
pub mod lodstats;
pub mod pstats;
pub mod ranges;
pub mod homes;
pub mod resident;
pub mod slab;
pub mod timeline;
pub mod replay;
pub mod threadcpu;

/// Fast non-cryptographic hashing for maps on the command processors' hot
/// paths (the image cache). Lookups compare keys, so a collision costs time
/// only; keys that stand for content (translation and pipeline hashes) keep
/// SipHash.
pub mod fx {
    use std::hash::{BuildHasherDefault, Hasher};

    #[derive(Default, Clone, Copy)]
    pub struct FxHasher(u64);

    const K: u64 = 0x517c_c1b7_2722_0a95;

    impl FxHasher {
        #[inline]
        fn add(&mut self, w: u64) {
            self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(K);
        }
    }

    impl Hasher for FxHasher {
        fn write(&mut self, bytes: &[u8]) {
            for c in bytes.chunks(8) {
                let mut b = [0u8; 8];
                b[..c.len()].copy_from_slice(c);
                self.add(u64::from_le_bytes(b));
            }
        }
        fn write_u8(&mut self, i: u8) {
            self.add(i as u64)
        }
        fn write_u16(&mut self, i: u16) {
            self.add(i as u64)
        }
        fn write_u32(&mut self, i: u32) {
            self.add(i as u64)
        }
        fn write_u64(&mut self, i: u64) {
            self.add(i)
        }
        fn write_usize(&mut self, i: usize) {
            self.add(i as u64)
        }
        /// Finished by a rotation (as rustc-hash 2 finishes, since
        /// 2026-11-01): the multiply keeps a key's trailing zero bits in the
        /// product's low bits, which hashbrown takes for a key's first
        /// bucket, so aligned addresses (guest pages, image bases, program
        /// addresses) all began probing at a few buckets.
        fn finish(&self) -> u64 {
            self.0.rotate_left(26)
        }
    }

    pub type Map<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
    pub type Set<T> = std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;

    #[cfg(test)]
    mod tests {
        use super::FxHasher;
        use std::hash::Hasher;

        /// Page-aligned keys reach most buckets of a table once finished
        /// (unfinished, 4,096 of them began at two of 8,192).
        #[test]
        fn aligned_keys_spread() {
            let starts = |rotate: bool| {
                let mut s = std::collections::HashSet::new();
                for i in 0..4096u64 {
                    let mut h = FxHasher::default();
                    h.write_u64(0x10_0000_0000 + i * 4096);
                    let v = if rotate { h.0.rotate_left(26) } else { h.0 };
                    s.insert(v & 8191);
                }
                s.len()
            };
            assert!(starts(false) <= 2, "{}", starts(false));
            assert!(starts(true) > 3000, "{}", starts(true));
        }
    }
}
pub mod images;
pub mod png;
pub mod track;
pub mod shaders;
pub mod swizzle_tables;
pub mod tiling;
#[cfg_attr(target_os = "linux", path = "../platform/linux/upscale.rs")]
pub mod upscale;
pub mod upmerge;
pub mod uiscale;
pub mod verify;
#[cfg(target_os = "linux")]
pub mod scatter_check;
pub mod vk;
pub mod xwait;
pub mod census;

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Flips presented so far (VideoOut).
pub static FRAMES: AtomicU64 = AtomicU64::new(0);
/// Command buffer submissions received (DCB/ACB).
pub static SUBMITS: AtomicU64 = AtomicU64::new(0);
/// Debug: dump GPU-visible images at the next flip (control `dump`).
pub static DUMP_REQUEST: AtomicBool = AtomicBool::new(false);
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

pub fn shutting_down() -> bool {
    SHUTDOWN.load(Ordering::Relaxed)
}

/// Stops command-processor waits (process shutdown).
pub fn shutdown() {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Queue id of the graphics (DCB) queue; compute queues use their Agc ids.
pub const GFX: u32 = u32::MAX;

/// Submitted command buffers; with `DS_CHECK_SUBMIT=1` also their content
/// hashes and the submission time (see `worker`).
struct Submission {
    bufs: Vec<(u64, u32)>,
    hashes: Vec<u64>,
    /// Copies of the buffers at submission (DS_CHECK_SUBMIT).
    copies: Vec<Vec<u32>>,
    at: std::time::Instant,
    /// Work other than command buffers (frame replay).
    ctl: Option<Ctl>,
    /// The queue's submission count when it was queued (its number).
    index: u64,
}

/// Work for a queue's command processor besides command buffers (frame
/// replay, `gpu::replay`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Ctl {
    /// Completes the GPU work recorded so far and its deferred effects.
    Finish,
    /// Keeps the command processor's register state in slot 0 or 1.
    Save(usize),
    /// Restores the register state kept in that slot.
    Restore(usize),
}

/// Queues work for a queue's command processor outside the title's
/// submission path (no throttling, no replay hold).
pub(crate) fn enqueue(queue_id: u32, bufs: Vec<(u64, u32)>, ctl: Option<Ctl>) {
    let q = queue(queue_id);
    let mut g = q.pending.lock();
    let index = q.submitted.fetch_add(1, Ordering::SeqCst) + 1;
    g.push_back(Submission { bufs, hashes: Vec::new(), copies: Vec::new(), at: std::time::Instant::now(), ctl, index });
    drop(g);
    q.cv.notify_one();
}

/// Waits until the graphics command processor has run every title submission queued so far
/// (`cullread`'s diagnosis): None when it had, else whether it got there within `max` (false
/// also when it polls a label meanwhile: the label may come from the waiting thread's work).
/// The graphics queue's title submissions not yet run by its command processor (diagnosis).
pub(crate) fn gfx_backlog() -> u64 {
    let qs = QUEUES.lock();
    qs.iter().find(|q| q.id == GFX).map_or(0, |q| q.submitted.load(Ordering::SeqCst).saturating_sub(q.done.load(Ordering::SeqCst)))
}

pub(crate) fn wait_gfx_caught_up(max: std::time::Duration) -> Option<bool> {
    let q = {
        let qs = QUEUES.lock();
        *qs.iter().find(|q| q.id == GFX)?
    };
    let target = q.submitted.load(Ordering::SeqCst);
    if q.done.load(Ordering::SeqCst) >= target {
        return None;
    }
    let t0 = std::time::Instant::now();
    let mut g = q.pending.lock();
    while q.done.load(Ordering::SeqCst) < target {
        if q.cp_waiting.load(Ordering::SeqCst) || shutting_down() {
            return Some(false);
        }
        let left = max.saturating_sub(t0.elapsed());
        if left.is_zero() {
            return Some(false);
        }
        q.done_cv.wait_for(&mut g, left.min(std::time::Duration::from_millis(5)));
    }
    Some(true)
}

/// The queues created so far.
pub(crate) fn queue_ids() -> Vec<u32> {
    QUEUES.lock().iter().map(|q| q.id).collect()
}

/// Whether every queue has completed all the work queued to it.
pub(crate) fn queues_idle() -> bool {
    QUEUES.lock().iter().all(|q| q.done.load(Ordering::SeqCst) >= q.submitted.load(Ordering::SeqCst))
}

/// The compute queues' command processors: (queue id, host thread id).
pub(crate) fn compute_threads() -> Vec<(u32, u32)> {
    QUEUES.lock().iter().filter(|q| q.id != GFX).map(|q| (q.id, q.tid.load(Ordering::Relaxed))).filter(|x| x.1 != 0).collect()
}

/// The GPU front end's threads by kind ("gfx", "acb", "par", "prep",
/// "submit", "complete", "scrub") and host thread id.
pub(crate) fn cpu_threads() -> Vec<(&'static str, u32)> {
    let mut v: Vec<(&'static str, u32)> = QUEUES.lock().iter().map(|q| (if q.id == GFX { "gfx" } else { "acb" }, q.tid.load(Ordering::Relaxed))).filter(|x| x.1 != 0).collect();
    v.extend(PROFILE_THREADS.lock().iter().copied());
    v
}

/// Whether one of the command buffers (address, dwords) holds a flip (the
/// private FLIP packet) in its top level.
pub(crate) fn has_flip(bufs: &[(u64, u32)]) -> bool {
    use crate::hle::libs::agc_cmd::{pop, DS_MAGIC};
    bufs.iter().any(|&(addr, size)| {
        let mut i = 0u64;
        while i < size as u64 {
            let a = addr + 4 * i;
            if !cp::guest_readable(a, 12) {
                return false;
            }
            let h = unsafe { *(a as *const u32) };
            if h >> 30 != 3 {
                i += 1;
                continue;
            }
            let n = ((h >> 16) & 0x3FFF) as u64 + 1;
            if (h >> 8) & 0xFF == 0x10 && n >= 2 && unsafe { *((a + 4) as *const u32) } == DS_MAGIC && unsafe { *((a + 8) as *const u32) } == pop::FLIP {
                return true;
            }
            i += n + 1;
        }
        false
    })
}

fn buffer_copy(addr: u64, size: u32) -> Vec<u32> {
    if !cp::guest_readable(addr, 4 * size as u64) {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) }.to_vec()
}

/// Diagnosis (`DS_SNAP_CHECK=<hex address>:<hex bytes>`): a sampled hash (every 16th dword) of a
/// CPU-written range at each graphics submission, compared when the graphics command processor
/// begins and ends recording it (`snap check: ...`): did the title rewrite what the submission
/// reads between submitting it and our recording?
fn snap_range() -> Option<(u64, u64)> {
    static R: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let v = std::env::var("DS_SNAP_CHECK").ok()?;
        let (a, n) = v.split_once(':')?;
        let p = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
        Some((p(a)?, p(n)?))
    })
}

fn snap_hash() -> u64 {
    let Some((a, n)) = snap_range() else { return 0 };
    if !cp::guest_readable(a, n) {
        return 0;
    }
    let mut h = fx::FxHasher::default();
    for k in (0..n / 4).step_by(16) {
        std::hash::Hasher::write_u32(&mut h, unsafe { std::ptr::read_volatile((a + 4 * k) as *const u32) });
    }
    std::hash::Hasher::finish(&h)
}

/// The graphics submissions' hashes by number; [checked, changed before recording, changed
/// during recording].
static SNAP: Mutex<Option<std::collections::HashMap<u64, u64>>> = Mutex::new(None);
static SNAP_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn snap_line() -> String {
    if snap_range().is_none() {
        return String::new();
    }
    let v: Vec<u64> = SNAP_STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    format!("    snap check (`DS_SNAP_CHECK`): {} graphics submissions, the range changed before their recording in {}, during it in {}\n", v[0], v[1], v[2])
}

/// `DS_TRACE_SUBMIT=1`: logs submissions and their completion.
fn trace_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_TRACE_SUBMIT").ok().as_deref() == Some("1"))
}

fn check_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_CHECK_SUBMIT").ok().as_deref() == Some("1"))
}

fn buffer_hash(addr: u64, size: u32) -> u64 {
    if !cp::guest_readable(addr, 4 * size as u64) {
        return 0;
    }
    let w = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
    w.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// Submission backpressure (`DS_SUBMIT_THROTTLE=1`, diagnostic): a submit
/// first waits (up to `THROTTLE_MAX`) until the queue completed all earlier
/// submissions, GPU work included, to test whether a problem depends on how
/// far the command processor lags behind the submitting threads (it can lag
/// by much more than hardware, e.g. during pipeline compilation). Not while
/// the CP polls a label (it may be waiting for the submitting thread).
fn throttle() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_SUBMIT_THROTTLE").ok().as_deref() == Some("1"))
}
const THROTTLE_MAX: std::time::Duration = std::time::Duration::from_millis(250);
/// Submissions up to this many dwords wait for their completion (see `submit`).
const SMALL_SUBMIT: u64 = 512;

/// The packets of a command buffer copy, one token each (diagnostics).
fn pm4_summary(w: &[u32]) -> String {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < w.len() {
        let h = w[i];
        if h >> 30 != 3 {
            out.push(format!("{i}:t{}", h >> 30));
            i += 1;
            continue;
        }
        let n = ((h >> 16) & 0x3FFF) as usize + 1;
        let p = |k: usize| w.get(i + 1 + k).copied().unwrap_or(0);
        let a64 = |k: usize| p(k) as u64 | (p(k + 1) as u64) << 32;
        let s = match (h >> 8) & 0xFF {
            0x10 if p(0) == crate::hle::libs::agc_cmd::DS_MAGIC => format!("mk{}", p(1)),
            0x93 => format!("wait64[{:#x}]>={:#x}", a64(1), a64(3)),
            0x3C => format!("wait[{:#x}]={:#x}", a64(1), p(3)),
            0x37 => format!("wd[{:#x}]={:#x}", a64(1), p(3)),
            0x49 => format!("rel[{:#x}]={:#x}", a64(2), a64(4)),
            0x15 => format!("disp{}x{}x{}", p(0), p(1), p(2)),
            0x76 => format!("sh{:#x}+{}", p(0), n - 1),
            0x46 => format!("ev{:#x}", p(0) & 0x3F),
            0x58 => "acq".to_string(),
            o => format!("op{o:#x}/{n}"),
        };
        out.push(format!("{i}:{s}"));
        i += n + 1;
    }
    out.join(" ")
}

/// (dispatches, draws, other packets) of a command buffer's top level.
fn packet_mix(addr: u64, size: u32) -> (u32, u32, u32) {
    let (mut d, mut w, mut o) = (0, 0, 0);
    let mut i = 0u64;
    while i < size as u64 {
        let a = addr + 4 * i;
        if !cp::guest_readable(a, 4) {
            break;
        }
        let h = unsafe { *(a as *const u32) };
        if h >> 30 != 3 {
            i += 1;
            continue;
        }
        let n = ((h >> 16) & 0x3FFF) as u64 + 1;
        match (h >> 8) & 0xFF {
            0x15 | 0x16 => d += 1,
            0x24 | 0x25 | 0x27 | 0x2C | 0x2D | 0x30 | 0x35 | 0x38 => w += 1,
            0x76 | 0x69 | 0x79 | 0x10 => {}
            _ => o += 1,
        }
        i += n + 1;
    }
    (d, w, o)
}

/// `DS_PM4_CENSUS=1` (diagnosis): how often the title submits command
/// buffers whose packets repeat byte for byte (a recorded submission could
/// then be replayed instead of recorded again). Per title submission, each
/// buffer's dwords are hashed with the buffers its INDIRECT_BUFFER packets
/// call (two levels) and compared with earlier submissions of a buffer at
/// the same address on the same queue (the last four kept); its draws and
/// dispatches are counted by what matched. The status report has the line;
/// `profile reset` starts the counts again.
fn pm4_census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DS_PM4_CENSUS").ok().as_deref() == Some("1"))
}

/// (queue, buffer address) -> its last submissions: (flip, hash, dwords).
static PM4_SEEN: Mutex<Option<std::collections::HashMap<(u32, u64), std::collections::VecDeque<(u64, u64, u32)>>>> = Mutex::new(None);

/// [buffers, of them first seen at their address, the same as the last
/// submission there, the same as an earlier one of the last four (not the
/// last), different; then the same five classes' draws and dispatches;
/// then their dwords; the age in flips of the matched submission summed,
/// matches 0, 1, 2, 3+ flips old].
static PM4_STATS: [AtomicU64; 20] = [const { AtomicU64::new(0) }; 20];

/// A buffer's dwords hashed with the buffers its INDIRECT_BUFFER packets call
/// (`depth` levels), and its (dispatches, draws).
fn pm4_hash(addr: u64, size: u32, depth: u32, h: &mut u64, mix: &mut (u32, u32)) {
    if !cp::guest_readable(addr, 4 * size as u64) {
        *h ^= 0x9e37_79b9_7f4a_7c15;
        return;
    }
    let w = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
    for &x in w {
        *h = (*h ^ x as u64).wrapping_mul(0x100_0000_01b3);
    }
    let mut i = 0usize;
    while i < w.len() {
        let hd = w[i];
        if hd >> 30 != 3 {
            i += 1;
            continue;
        }
        let n = ((hd >> 16) & 0x3FFF) as usize + 1;
        match (hd >> 8) & 0xFF {
            0x15 | 0x16 => mix.0 += 1,
            0x24 | 0x25 | 0x27 | 0x2C | 0x2D | 0x30 | 0x35 | 0x38 => mix.1 += 1,
            0x3F if depth > 0 && i + 3 < w.len() => {
                let target = (w[i + 1] & !3) as u64 | ((w[i + 2] as u64 & 0xFFFF) << 32);
                let dw = w[i + 3] & 0xF_FFFF;
                if target != 0 && dw != 0 {
                    pm4_hash(target, dw, depth - 1, h, mix);
                }
            }
            _ => {}
        }
        i += n + 1;
    }
}

/// (queue, address) -> the last submission's dwords there (`pm4_census`).
static PM4_LAST: Mutex<Option<std::collections::HashMap<(u32, u64), Vec<u32>>>> = Mutex::new(None);
/// [buffers compared (same size), dwords compared, dwords differing, buffers
/// of another size], then differing dwords by the opcode of their packet.
static PM4_DIFF: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static PM4_DIFF_OPS: Mutex<Option<std::collections::HashMap<u32, u64>>> = Mutex::new(None);

fn pm4_diff(queue_id: u32, addr: u64, size: u32) {
    if !cp::guest_readable(addr, 4 * size as u64) {
        return;
    }
    let w = unsafe { std::slice::from_raw_parts(addr as *const u32, size as usize) };
    let mut g = PM4_LAST.lock();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    if let Some(prev) = m.get(&(queue_id, addr)) {
        if prev.len() == w.len() {
            PM4_DIFF[0].fetch_add(1, Ordering::Relaxed);
            PM4_DIFF[1].fetch_add(w.len() as u64, Ordering::Relaxed);
            let mut ops = PM4_DIFF_OPS.lock();
            let o = ops.get_or_insert_with(std::collections::HashMap::new);
            let mut i = 0usize;
            let mut nd = 0u64;
            while i < w.len() {
                let hd = w[i];
                let n = if hd >> 30 == 3 { ((hd >> 16) & 0x3FFF) as usize + 2 } else { 1 };
                let op = if hd >> 30 == 3 { (hd >> 8) & 0xFF } else { 0x100 };
                let end = (i + n).min(w.len());
                let d = (i..end).filter(|&k| w[k] != prev[k]).count() as u64;
                if d != 0 {
                    *o.entry(op).or_default() += d;
                    nd += d;
                }
                i = end;
            }
            PM4_DIFF[2].fetch_add(nd, Ordering::Relaxed);
        } else {
            PM4_DIFF[3].fetch_add(1, Ordering::Relaxed);
        }
    }
    m.insert((queue_id, addr), w.to_vec());
    if m.len() > 4096 {
        m.clear();
    }
}

fn pm4_census(queue_id: u32, buffers: &[(u64, u32)]) {
    for &(addr, size) in buffers {
        pm4_diff(queue_id, addr, size);
    }
    let flip = FRAMES.load(Ordering::Relaxed);
    let mut g = PM4_SEEN.lock();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    for &(addr, size) in buffers {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ size as u64;
        let mut mix = (0u32, 0u32);
        pm4_hash(addr, size, 2, &mut h, &mut mix);
        let cmds = (mix.0 + mix.1) as u64;
        let e = m.entry((queue_id, addr)).or_default();
        let class = if e.is_empty() {
            1
        } else if e.back().is_some_and(|l| l.1 == h && l.2 == size) {
            2
        } else if e.iter().any(|l| l.1 == h && l.2 == size) {
            3
        } else {
            4
        };
        if class == 2 || class == 3 {
            let age = e.iter().rev().find(|l| l.1 == h && l.2 == size).map_or(0, |l| flip.saturating_sub(l.0));
            PM4_STATS[15].fetch_add(age, Ordering::Relaxed);
            PM4_STATS[16 + (age.min(3) as usize)].fetch_add(1, Ordering::Relaxed);
        }
        PM4_STATS[0].fetch_add(1, Ordering::Relaxed);
        PM4_STATS[class].fetch_add(1, Ordering::Relaxed);
        PM4_STATS[5 + class - 1].fetch_add(cmds, Ordering::Relaxed);
        PM4_STATS[10 + class - 1].fetch_add(size as u64, Ordering::Relaxed);
        if e.len() >= 4 {
            e.pop_front();
        }
        e.push_back((flip, h, size));
    }
    if m.len() > 1 << 16 {
        m.clear();
    }
}

/// The census's status line ("" when off).
pub fn pm4_census_line() -> String {
    if !pm4_census_on() {
        return String::new();
    }
    let s: Vec<u64> = PM4_STATS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    let dd: Vec<u64> = PM4_DIFF.iter().map(|x| x.load(Ordering::Relaxed)).collect();
    let mut ops: Vec<(u32, u64)> = PM4_DIFF_OPS.lock().as_ref().map(|m| m.iter().map(|(&k, &v)| (k, v)).collect()).unwrap_or_default();
    ops.sort_by(|a, b| b.1.cmp(&a.1));
    let diff_line = format!(
        "    PM4 census, against the last submission at the same address: {} buffers of the same size ({} of another), {} of {} dwords differ ({:.2}%); by the packet holding them: {}\n",
        dd[0],
        dd[3],
        dd[2],
        dd[1],
        100.0 * dd[2] as f64 / dd[1].max(1) as f64,
        ops.iter().take(12).map(|&(op, n)| if op == 0x100 { format!("not a type-3 header {n}") } else { format!("{op:#x} {n}") }).collect::<Vec<_>>().join(", ")
    );
    let cmds: u64 = s[5..9].iter().sum::<u64>().max(1);
    let pct = |x: u64| 100.0 * x as f64 / cmds as f64;
    let matched = (s[2] + s[3]).max(1);
    format!(
        "    PM4 census: {} title command buffers (first at their address {}, as the last there {}, as an earlier one {}, new {}); their draws and dispatches {}: in repeated buffers {:.1}% (as the last {:.1}%, as an earlier one {:.1}%), first seen {:.1}%, new {:.1}%; dwords {} / {} / {} / {}; matched submissions {:.1} flips old on average (0: {}, 1: {}, 2: {}, 3+: {})\n",
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        cmds,
        pct(s[6] + s[7]),
        pct(s[6]),
        pct(s[7]),
        pct(s[5]),
        pct(s[8]),
        s[10],
        s[11],
        s[12],
        s[13],
        s[15] as f64 / matched as f64,
        s[16],
        s[17],
        s[18],
        s[19]
    ) + &diff_line
}

fn pm4_census_reset() {
    for x in PM4_STATS.iter().chain(PM4_DIFF.iter()) {
        x.store(0, Ordering::Relaxed);
    }
    if let Some(m) = PM4_DIFF_OPS.lock().as_mut() {
        m.clear();
    }
}

/// Whether a submission returns only once its work completed (see `submit`).
fn waits_for_completion(bufs: &[(u64, u32)]) -> bool {
    static SYNC: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *SYNC.get_or_init(|| match std::env::var("DS_SUBMIT_SYNC").ok().as_deref() {
        Some("1") => 2,
        Some("small") => 1,
        _ => 0,
    });
    mode == 2 || (mode == 1 && bufs.iter().map(|x| x.1 as u64).sum::<u64>() <= SMALL_SUBMIT)
}
static THROTTLE_WAITS: AtomicU64 = AtomicU64::new(0);
static THROTTLE_NS: AtomicU64 = AtomicU64::new(0);
static THROTTLE_TIMEOUTS: AtomicU64 = AtomicU64::new(0);

/// Submission backpressure statistics.
pub fn throttle_status() -> String {
    format!(
        "submit throttle: {} waits, {} ms total, {} timeouts",
        THROTTLE_WAITS.load(Ordering::Relaxed),
        THROTTLE_NS.load(Ordering::Relaxed) / 1_000_000,
        THROTTLE_TIMEOUTS.load(Ordering::Relaxed)
    )
}

struct Queue {
    id: u32,
    pending: Mutex<VecDeque<Submission>>,
    cv: Condvar,
    /// Submissions completed (diagnostics, backpressure).
    done: AtomicU64,
    /// Submissions queued so far.
    submitted: AtomicU64,
    /// Signalled when a submission completes.
    done_cv: Condvar,
    /// The command processor is polling a label.
    cp_waiting: std::sync::atomic::AtomicBool,
    /// Host thread id of the queue's command processor (stack samples).
    tid: std::sync::atomic::AtomicU32,
}

static QUEUES: Mutex<Vec<&'static Queue>> = Mutex::new(Vec::new());

fn queue(id: u32) -> &'static Queue {
    let mut qs = QUEUES.lock();
    if let Some(q) = qs.iter().find(|q| q.id == id) {
        return q;
    }
    let q: &'static Queue = Box::leak(Box::new(Queue {
        id,
        pending: Mutex::new(VecDeque::new()),
        cv: Condvar::new(),
        done: AtomicU64::new(0),
        submitted: AtomicU64::new(0),
        done_cv: Condvar::new(),
        cp_waiting: std::sync::atomic::AtomicBool::new(false),
        tid: std::sync::atomic::AtomicU32::new(0),
    }));
    qs.push(q);
    let name = if id == GFX { "gpu-gfx".to_string() } else { format!("gpu-acb{id:#x}") };
    crate::info!("gpu: {name} command processor started");
    crate::thread::spawn_host(&name.clone(), 8 << 20, Box::new(move || worker(q, &name))).expect("gpu queue thread");
    q
}

fn worker(q: &'static Queue, name: &str) {
    crate::log::set_thread_name(name);
    q.tid.store(unsafe { crate::platform::os::Win32::System::Threading::GetCurrentThreadId() }, Ordering::Relaxed);
    crate::exception::arm_watch_current_thread(false);
    let mut cp = cp::Cp::new(name, q.id != GFX);
    if q.id == GFX {
        forward::enable_here();
        prio::set_main();
    }
    shaders::set_compute_queue(q.id != GFX);
    if q.id != GFX {
        prio::set_compute(q.id);
    }
    cp.wait_flag = Some(&q.cp_waiting);
    // (A title submission: never a replay's control entry, which a stretch
    // held past a run or label wait must not run into.)
    cp.more_follows = Some(Box::new(move || q.pending.lock().front().is_some_and(|b| b.ctl.is_none())));
    // Register states kept for a frame replay (`Ctl::Save`).
    let mut saved: [Option<cp::ChunkState>; 2] = [None, None];
    // Submissions a stretch past a label wait runs through.
    let mut kept: Vec<Kept> = Vec::new();
    loop {
        prio::apply();
        let batch = {
            let mut g = q.pending.lock();
            let mut idle = None;
            let mut since: Option<std::time::Instant> = None;
            let b = loop {
                if let Some(b) = g.pop_front() {
                    break b;
                }
                idle = idle.or_else(timeline::mark);
                if cp.holds_across() {
                    // (Recording past a run or a label wait with no title
                    // submission to go on with: what it holds waits too.)
                    let t = *since.get_or_insert_with(std::time::Instant::now);
                    if q.cv.wait_for(&mut g, std::time::Duration::from_secs(1)).timed_out() && t.elapsed() > std::time::Duration::from_secs(2) && since.is_some() {
                        crate::warn!("gpu[{name}]: no title submission for 2 s while recording past a run or label wait (its held work waits too)");
                        since = None;
                    }
                } else {
                    q.cv.wait(&mut g);
                }
            };
            timeline::end(name, idle, || "idle: no title submission".into());
            b
        };
        if let Some(c) = batch.ctl {
            match c {
                Ctl::Finish => cp.finish(),
                Ctl::Save(k) => saved[k] = Some(cp.chunk_state()),
                Ctl::Restore(k) => {
                    if let Some(s) = &saved[k] {
                        cp.set_chunk_state(s);
                    }
                }
            }
            let _g = q.pending.lock();
            q.done.fetch_add(1, Ordering::SeqCst);
            q.done_cv.notify_all();
            continue;
        }
        if q.id == GFX && frameprobe::on() {
            frameprobe::submission(&batch.bufs);
        }
        // Recording past a label wait runs on through submissions: they are
        // kept until it settles, to run again from the wait if what was
        // recorded past it is dropped (`Cp::take_rewind_sub`).
        if cp.holds_across() {
            kept.push(Kept { index: batch.index, bufs: batch.bufs.clone(), at: batch.at });
        }
        // (Timelines: how long the submission waited in the queue, how many
        // more wait behind it, and its run: what a planner ahead of the
        // command processor would have to work on, `tools/sublag.py`.)
        let tl_sub = timeline::mark();
        if tl_sub.is_some() {
            let behind = q.pending.lock().len();
            let dwords: u64 = batch.bufs.iter().map(|b| b.1 as u64).sum();
            timeline::span_back(&format!("{name} queued"), batch.at.elapsed(), || format!("sub {} queued ({} buffers, {dwords} dwords, {behind} behind)", batch.index, batch.bufs.len()));
        }
        let snap = (q.id == GFX && snap_range().is_some()).then(|| (SNAP.lock().as_mut().and_then(|m| m.remove(&batch.index)), snap_hash()));
        run_submission(&mut cp, &batch, name);
        if let Some((Some(h_sub), h_begin)) = snap {
            let h_end = snap_hash();
            SNAP_STATS[0].fetch_add(1, Ordering::Relaxed);
            let before = h_begin != h_sub;
            let during = h_end != h_begin;
            if before {
                SNAP_STATS[1].fetch_add(1, Ordering::Relaxed);
            }
            if during {
                SNAP_STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            static LOGGED: AtomicU64 = AtomicU64::new(0);
            if (before || during) && LOGGED.fetch_add(1, Ordering::Relaxed) < 400 {
                crate::info!(
                    "snap check: submission {} ({} buffers) at flip {}: the range changed {} ({} ms after its submission)",
                    batch.index,
                    batch.bufs.len(),
                    crate::hle::libs::videoout::QUEUED_FLIP.load(Ordering::Relaxed),
                    match (before, during) {
                        (true, true) => "before and during its recording",
                        (true, false) => "before its recording",
                        _ => "during its recording",
                    },
                    batch.at.elapsed().as_millis()
                );
            }
        }
        timeline::end(&format!("{name} subs"), tl_sub, || format!("sub {} run", batch.index));
        while let Some((s, buf, addr)) = cp.take_rewind_sub() {
            let Some(k) = kept.iter().position(|x| x.index == s) else {
                crate::error!("gpu[{name}]: a stretch past a label wait or run was dropped but submission {s} is not kept; going on");
                break;
            };
            // (They stay kept while they run again: a stretch recorded there
            // may be dropped again, back into one of them.)
            static REWINDS: AtomicU64 = AtomicU64::new(0);
            if REWINDS.fetch_add(1, Ordering::Relaxed) < 40 {
                crate::info!("gpu[{name}]: going back into submission {s} at {addr:#x} (buffer {buf}; {} submissions kept, from {})", kept.len(), kept.first().map_or(0, |x| x.index));
            }
            let again: Vec<Kept> = kept[k..].to_vec();
            for (j, x) in again.iter().enumerate() {
                let from = if j == 0 { Some((buf, addr)) } else { None };
                run_kept(&mut cp, x, from, name);
                if cp.rewind_sub_pending() {
                    break;
                }
            }
        }
        if cp.holds_across() {
            if !kept.iter().any(|x| x.index == batch.index) {
                kept.push(Kept { index: batch.index, bufs: batch.bufs.clone(), at: batch.at });
            }
        } else {
            kept.clear();
        }
        if throttle() || waits_for_completion(&batch.bufs) {
            cp.finish();
        }
        let _g = q.pending.lock();
        let d = q.done.fetch_add(1, Ordering::SeqCst) + 1;
        q.done_cv.notify_all();
        if trace_submit() {
            crate::info!("submit trace: queue {:#x} #{d} completed ({} ms after submission)", q.id, batch.at.elapsed().as_millis());
        }
    }
}

/// A submission kept while the graphics command processor records past a
/// label wait or a run (its number, command buffers and arrival).
#[derive(Clone)]
struct Kept {
    index: u64,
    bufs: Vec<(u64, u32)>,
    at: std::time::Instant,
}

/// Runs a kept submission again, from buffer `from.0` at address `from.1`
/// (a stretch past a label wait was dropped), or whole.
fn run_kept(cp: &mut cp::Cp, k: &Kept, from: Option<(usize, u64)>, name: &str) {
    cp.submitted_at = Some(k.at);
    cp.sub_index = k.index;
    cp.new_submission();
    let nbufs = k.bufs.len();
    let (mut i, mut at) = match from {
        Some((b, a)) => (b, Some(a)),
        None => (0, None),
    };
    while i < nbufs {
        let (addr, size) = k.bufs[i];
        cp.next_in_submission = i + 1 < nbufs;
        cp.cur_buf = i;
        match at.take() {
            // (Its end too: a run gone past that ended there, nothing left
            // of the buffer to run; taken for outside, the buffer ran again
            // from its start.)
            Some(a) if a >= addr && a <= addr + 4 * size as u64 => cp.run(a, size - ((a - addr) / 4) as u32),
            Some(a) => {
                crate::error!("gpu[{name}]: rewind to {a:#x} outside its command buffer {addr:#x}");
                cp.run(addr, size)
            }
            None => cp.run(addr, size),
        }
        if cp.rewind_sub_pending() {
            return;
        }
        if let Some((j, a)) = cp.take_rewind() {
            i = j;
            at = Some(a);
            continue;
        }
        i += 1;
    }
}

/// Runs a title submission's command buffers.
fn run_submission(cp: &mut cp::Cp, batch: &Submission, name: &str) {
    cp.submitted_at = Some(batch.at);
    cp.sub_index = batch.index;
    cp.new_submission();
    let nbufs = batch.bufs.len();
    // (Recording past a parallel run may go back to an earlier buffer of
    // the submission: `Cp::take_rewind`.)
    let mut i = 0usize;
    let mut from: Option<u64> = None;
    {
        while i < nbufs {
            let (addr, size) = batch.bufs[i];
            cp.next_in_submission = i + 1 < nbufs;
            cp.cur_buf = i;
            if trace_submit() && size == 277 {
                // Bink decode command buffers: their counter (dword 272) and
                // dword 159 at submission and now, and the label the first
                // packet waits for.
                let rd = |k: u64| if cp::guest_readable(addr + 4 * k, 4) { unsafe { *((addr + 4 * k) as *const u32) } } else { 0 };
                let sub = batch.copies.get(i).map(|w| (w.get(271).copied().unwrap_or(0), w.get(158).copied().unwrap_or(0))).unwrap_or((0, 0));
                let label = (rd(8) as u64) | ((rd(9) as u64) << 32);
                let lv = if cp::guest_readable(label, 8) { unsafe { *(label as *const u64) } } else { 0 };
                crate::info!("bink trace: {addr:#x} start: [272] {:#x} (submitted {:#x}) [159] {:#x} (submitted {:#x}), waits [{label:#x}] >= {:#x} (now {lv:#x}), {} ms after submission", rd(272), sub.0, rd(159), sub.1, rd(10), batch.at.elapsed().as_millis());
            }
            if let Some(&h) = batch.hashes.get(i) {
                if buffer_hash(addr, size) != h {
                    crate::warn!("gpu[{name}]: command buffer {addr:#x} ({size} dwords) changed between its submission and its execution ({} ms later)", batch.at.elapsed().as_millis());
                }
            }
            match from.take() {
                Some(a) => cp.run(a, size - ((a - addr) / 4) as u32),
                None => cp.run(addr, size),
            }
            if cp.rewind_sub_pending() {
                return;
            }
            if let Some((j, a)) = cp.take_rewind() {
                i = j;
                from = Some(a);
                continue;
            }
            if let Some(&h) = batch.hashes.get(i) {
                if buffer_hash(addr, size) != h {
                    let now = buffer_copy(addr, size);
                    let diffs: Vec<String> = batch.copies.get(i).map(|old| old.iter().zip(&now).enumerate().filter(|(_, (a, b))| a != b).take(6).map(|(k, (a, b))| format!("[{k}] {a:#x}->{b:#x}")).collect()).unwrap_or_default();
                    crate::warn!("gpu[{name}]: command buffer {addr:#x} ({size} dwords) changed during its execution ({} ms after submission): {}", batch.at.elapsed().as_millis(), diffs.join(" "));
                }
            }
            i += 1;
        }
    }
}

/// Queues command buffers (address, size in dwords) for execution on `queue_id`.
/// The graphics queue at each flip it reaches (`profile report`, since
/// `profile reset`): title submissions queued behind the flip and how long
/// the oldest has waited, ms (how far the title runs ahead of recording).
static RUN_AHEAD: parking_lot::Mutex<Vec<(u32, f32)>> = parking_lot::Mutex::new(Vec::new());

pub(crate) fn note_run_ahead() {
    let q = queue(GFX);
    let (n, age) = {
        let g = q.pending.lock();
        (g.len() as u32, g.front().map_or(0.0, |s| s.at.elapsed().as_secs_f32() * 1000.0))
    };
    let mut r = RUN_AHEAD.lock();
    if r.len() < 1 << 16 {
        r.push((n, age));
    }
}

fn run_ahead_line() -> String {
    let mut r = RUN_AHEAD.lock().clone();
    if r.is_empty() {
        return String::new();
    }
    let pct = |v: &mut Vec<f32>, q: f32| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f32 * q) as usize]
    };
    let mut n: Vec<f32> = r.iter().map(|x| x.0 as f32).collect();
    let mut a: Vec<f32> = r.iter().map(|x| x.1).collect();
    let line = format!(
        "    graphics queue at its flips ({}): title submissions queued behind p10 {:.0}, p50 {:.0}, p90 {:.0}; the oldest waited p10 {:.1}, p50 {:.1}, p90 {:.1} ms\n",
        r.len(),
        pct(&mut n, 0.1),
        pct(&mut n, 0.5),
        pct(&mut n, 0.9),
        pct(&mut a, 0.1),
        pct(&mut a, 0.5),
        pct(&mut a, 0.9)
    );
    r.clear();
    line
}

/// The take-per-submission dry run (`tune fe3_scan`): a helper thread scans
/// after title submissions, those that come meanwhile coalesced; [scans,
/// submissions, runs, pages queried, pages written, failed queries, ns].
mod fe3_probe {
    use parking_lot::{Condvar, Mutex};
    use std::sync::atomic::{AtomicU64, Ordering};
    static PENDING: Mutex<u64> = Mutex::new(0);
    static CV: Condvar = Condvar::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
    /// `tune fe3_scan 2`: scans, then `fe2::fe3_scan_dry2`'s sums.
    static STATS2: [AtomicU64; 23] = [const { AtomicU64::new(0) }; 23];

    pub fn note_submission() {
        STARTED.call_once(|| {
            let _ = crate::thread::spawn_host("gpu-fe3-probe", 1 << 20, Box::new(run));
        });
        *PENDING.lock() += 1;
        CV.notify_one();
    }

    fn run() {
        crate::log::set_thread_name("gpu-fe3-probe");
        loop {
            let n = {
                let mut g = PENDING.lock();
                while *g == 0 {
                    CV.wait(&mut g);
                }
                std::mem::take(&mut *g)
            };
            if super::tune::fe3_scan() == 2 {
                let v = super::fe2::fe3_scan_dry2();
                STATS2[0].fetch_add(1, Ordering::Relaxed);
                for (k, x) in v.into_iter().enumerate() {
                    STATS2[k + 1].fetch_add(x, Ordering::Relaxed);
                }
                STATS[1].fetch_add(n, Ordering::Relaxed);
                continue;
            }
            let (runs, queried, written, failed, ns) = super::fe2::fe3_scan_dry();
            for (k, v) in [1, n, runs, queried, written, failed, ns].into_iter().enumerate() {
                STATS[k].fetch_add(v, Ordering::Relaxed);
            }
        }
    }

    pub fn reset() {
        for c in STATS.iter().chain(STATS2.iter()) {
            c.store(0, Ordering::Relaxed);
        }
    }

    pub fn line(flips: u64) -> String {
        let v: Vec<u64> = STATS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let w: Vec<u64> = STATS2.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        if w[0] != 0 {
            let s = w[0] as f64;
            let mib = |pages: u64| pages as f64 * 4096.0 / 1048576.0 / s;
            return format!(
                "    take-per-submission dry run, cheapest calls (`tune fe3_scan 2`): {} scans for {} title submissions; per scan {:.0} calls over {:.1} MiB, {:.0} pages written since their last take, {:.1} failed, {:.1} us; {:.2} ms of scans a flip; valid pages not armed per scan {:.1} MiB: last CPU write found < 2 flips ago {:.1}, < 16 {:.1}, < 128 {:.1}, longer {:.1}, never {:.1}; GPU-written {:.1}; in runs the title's CPU cannot write {:.1}, outside settled imports {:.1}; written within 16 flips in the title's 'batch' mappings {:.1}, elsewhere {:.1}; the rings' spans of pages written within 16 flips {:.1} MiB, a query-only scan of them (one call a ring) {:.1} us; pages written within 2 flips in rings {:.2} MiB over spans of {:.1} MiB in {:.1} rings (by 64 MiB window: spans of {:.1} MiB)\n",
                w[0],
                v[1],
                w[1] as f64 / s,
                mib(w[2]),
                w[3] as f64 / s,
                w[4] as f64 / s,
                w[5] as f64 / s / 1000.0,
                w[5] as f64 / 1e6 / flips.max(1) as f64,
                mib(w[6]),
                mib(w[7]),
                mib(w[8]),
                mib(w[9]),
                mib(w[10]),
                mib(w[11]),
                mib(w[12]),
                mib(w[13]),
                mib(w[14]),
                mib(w[15]),
                mib(w[16]),
                mib(w[17]),
                w[18] as f64 / s / 1000.0,
                mib(w[20]),
                mib(w[19]),
                w[21] as f64 / s,
                mib(w[22])
            );
        }
        if v[0] == 0 {
            return String::new();
        }
        let s = v[0] as f64;
        format!(
            "    take-per-submission dry run (`tune fe3_scan`): {} scans for {} title submissions; per scan {:.0} runs of valid pages not armed, {:.1} MiB queried, {:.0} pages written since their last take, {:.1} failed queries, {:.1} us; {:.2} ms of scans a flip
",
            v[0],
            v[1],
            v[2] as f64 / s,
            v[3] as f64 * 4096.0 / 1048576.0 / s,
            v[4] as f64 / s,
            v[5] as f64 / s,
            v[6] as f64 / s / 1000.0,
            v[6] as f64 / 1e6 / flips.max(1) as f64
        )
    }
}

pub fn submit(queue_id: u32, buffers: Vec<(u64, u32)>) {
    crate::engine::on_submit();
    if buffers.is_empty() {
        return;
    }
    if tune::fe3_scan() != 0 && fe2::on() {
        fe3_probe::note_submission();
    }
    if pm4_census_on() {
        pm4_census(queue_id, &buffers);
    }
    kco::trigger();
    // A frame replay holds the title at the start and end of the frame it
    // records.
    replay::hold();
    SUBMITS.fetch_add(1, Ordering::Relaxed);
    let q = queue(queue_id);
    let hashes = if check_submit() { buffers.iter().map(|&(a, n)| buffer_hash(a, n)).collect() } else { Vec::new() };
    let copies = if check_submit() || trace_submit() { buffers.iter().map(|&(a, n)| buffer_copy(a, n)).collect() } else { Vec::new() };
    let mut g = q.pending.lock();
    if throttle() {
        let t0 = std::time::Instant::now();
        let mut waited = false;
        while q.done.load(Ordering::SeqCst) < q.submitted.load(Ordering::SeqCst) && !q.cp_waiting.load(Ordering::SeqCst) {
            waited = true;
            let left = THROTTLE_MAX.saturating_sub(t0.elapsed());
            if left.is_zero() {
                THROTTLE_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                break;
            }
            q.done_cv.wait_for(&mut g, left.min(std::time::Duration::from_millis(5)));
        }
        if waited {
            THROTTLE_WAITS.fetch_add(1, Ordering::Relaxed);
            THROTTLE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
    let index = q.submitted.fetch_add(1, Ordering::SeqCst) + 1;
    if trace_submit() {
        let (d, w, o) = buffers.iter().map(|&(a, n)| packet_mix(a, n)).fold((0, 0, 0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2));
        crate::info!("submit trace: queue {queue_id:#x} #{index} {:#x}+{} (done {}) dispatches {d} draws {w} other {o}", buffers[0].0, buffers[0].1, q.done.load(Ordering::SeqCst));
        // Bink decodes (they start with a "BinkGPU" marker): list the
        // packets of the first one and of ones with other sizes.
        static LISTED: AtomicU64 = AtomicU64::new(0);
        if let Some(w) = copies.first() {
            let bink = w.len() > 3 && w[1] == crate::hle::libs::agc_cmd::DS_MAGIC && w[2] == crate::hle::libs::agc_cmd::pop::PUSH_MARKER && w.get(4) == Some(&u32::from_le_bytes(*b"Bink"));
            if bink && (w.len() != 277 || LISTED.load(Ordering::Relaxed) == 0) && LISTED.fetch_add(1, Ordering::Relaxed) < 10 {
                crate::info!("submit trace: #{index} packets: {}", pm4_summary(w));
            }
        }
    }
    if queue_id == GFX && snap_range().is_some() {
        let h = snap_hash();
        SNAP.lock().get_or_insert_with(std::collections::HashMap::new).insert(index, h);
    }
    // In queue order (under the queue's lock): the replay keeps each queue's.
    replay::note_submitted(queue_id, &buffers);
    g.push_back(Submission { bufs: buffers, hashes, copies, at: std::time::Instant::now(), ctl: None, index });
    q.cv.notify_one();
    // DS_SUBMIT_SYNC=small (up to SMALL_SUBMIT dwords) or =1 (all),
    // diagnostic: the submit returns once its work completed (no CPU/GPU
    // overlap for those submissions; about halves the frame rate). Bounded,
    // and not while the CP polls a label that may come from the submitter.
    if waits_for_completion(&g.back().map(|b| b.bufs.clone()).unwrap_or_default()) {
        let t0 = std::time::Instant::now();
        while q.done.load(Ordering::SeqCst) < index && !q.cp_waiting.load(Ordering::SeqCst) && t0.elapsed() < THROTTLE_MAX {
            q.done_cv.wait_for(&mut g, std::time::Duration::from_millis(5));
        }
    }
    drop(g);
    q.cv.notify_one();
}

/// `DS_PROFILE_GPU=1`: samples the GPU queue threads every 10 ms; the most
/// frequent host stacks (top 3 frames) are listed in hang reports.
static PROFILE: Mutex<Option<std::collections::HashMap<String, u64>>> = Mutex::new(None);

/// Inclusive samples per (queue kind, host function) and samples per queue
/// kind ("gfx" / "acb"): the share of time a function was on the stack.
static PROFILE_FN: Mutex<Option<(std::collections::HashMap<(&'static str, String), u64>, std::collections::HashMap<&'static str, u64>)>> = Mutex::new(None);

/// Samples of a queue kind not waiting or sleeping, by the innermost function
/// of ours on the stack: where command processing spends its CPU time.
static PROFILE_BUSY: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);

/// Samples of a queue kind blocked on a mutex (parking_lot `lock_slow` or an
/// SRW lock), by the innermost function of ours on the stack: who waits for
/// which lock.
static PROFILE_LOCKS: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);

/// Busy samples by the function our innermost function was calling (another
/// crate's, the runtime library's, the driver's: "module!name") and by the
/// innermost frame of all: where per-command time goes below our code.
static PROFILE_CALLS: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);
/// Busy samples by the source line our innermost function was at (its own
/// work or the call it was making): where per-command time goes within a
/// function.
static PROFILE_LINES: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);
static PROFILE_LEAF: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);
/// Busy samples by the instruction at the sampled address: [busy, at a locked read-modify-write
/// or an exchange with memory, right after one (the sample taken once it retired), at a
/// spin-wait pause]. What the cross-core traffic of shared counters, reference counts and lock
/// words costs the recording threads (2026-11-16).
static PROFILE_ATOMIC: Mutex<Option<std::collections::HashMap<&'static str, [u64; 4]>>> = Mutex::new(None);
/// Busy samples at or right after a locked instruction, by its function and source line.
static PROFILE_ATOMIC_SITES: Mutex<Option<std::collections::HashMap<(&'static str, String), u64>>> = Mutex::new(None);

/// The instruction at a sampled address: 1 a locked read-modify-write or an exchange with
/// memory, 2 right after one (with its address), 3 a spin-wait pause, 0 anything else.
fn sampled_instruction(rip: u64) -> (u8, u64) {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind};
    use crate::platform::os::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use crate::platform::os::Win32::System::Threading::GetCurrentProcess;
    let mut buf = [0u8; 32];
    let mut n = 0usize;
    let ok = rip > 16 && unsafe { ReadProcessMemory(GetCurrentProcess(), (rip - 16) as *const core::ffi::c_void, buf.as_mut_ptr() as *mut core::ffi::c_void, 32, &mut n) } != 0 && n == 32;
    if !ok {
        return (0, 0);
    }
    let locked = |i: &iced_x86::Instruction| i.has_lock_prefix() || (i.mnemonic() == Mnemonic::Xchg && (i.op0_kind() == OpKind::Memory || i.op1_kind() == OpKind::Memory));
    let at = Decoder::with_ip(64, &buf[16..], rip, DecoderOptions::NONE).decode();
    if !at.is_invalid() {
        if locked(&at) {
            return (1, rip);
        }
        if at.mnemonic() == Mnemonic::Pause {
            return (3, rip);
        }
    }
    for k in 1..=15usize {
        let start = rip - k as u64;
        let i = Decoder::with_ip(64, &buf[16 - k..16], start, DecoderOptions::NONE).decode();
        if !i.is_invalid() && i.len() == k && locked(&i) {
            return (2, start);
        }
    }
    (0, 0)
}

/// Threads sampled besides the queues' command processors: (kind, host tid)
/// ("par": parallel recording workers, "prep": stage preparation workers).
static PROFILE_THREADS: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());

/// Samples the calling thread as `kind` while profiling (`DS_PROFILE_GPU`).
pub fn profile_this_thread(kind: &'static str) {
    let tid = unsafe { crate::platform::os::Win32::System::Threading::GetCurrentThreadId() };
    PROFILE_THREADS.lock().push((kind, tid));
}

/// The thread sampler's period in ms (`DS_PROFILE_MS`).
static PROFILE_MS: AtomicU64 = AtomicU64::new(10);

/// Kinds of sampled threads, in report order.
const PROFILE_KINDS: [&str; 6] = ["gfx", "acb", "par", "cpar", "prep", "submit"];

/// Starts the 10 ms thread sampler: `DS_PROFILE_GPU=1` (with the per-command
/// timers and statistics) or `DS_PROFILE_GPU=2` (the sampler alone: function
/// profiles of an otherwise unprofiled run, whose timers and statistics
/// would show up in them).
pub fn start_profiler() {
    if !matches!(std::env::var("DS_PROFILE_GPU").ok().as_deref(), Some("1" | "2")) {
        return;
    }
    // `DS_PROFILE_MS=<ms>`: the sampling period (default 10).
    PROFILE_MS.store(std::env::var("DS_PROFILE_MS").ok().and_then(|v| v.parse::<u64>().ok()).filter(|&ms| ms > 0).unwrap_or(10), Ordering::Relaxed);
    *PROFILE.lock() = Some(std::collections::HashMap::new());
    *PROFILE_FN.lock() = Some(Default::default());
    *PROFILE_BUSY.lock() = Some(Default::default());
    *PROFILE_LOCKS.lock() = Some(Default::default());
    *PROFILE_CALLS.lock() = Some(Default::default());
    *PROFILE_LINES.lock() = Some(Default::default());
    *PROFILE_LEAF.lock() = Some(Default::default());
    *PROFILE_ATOMIC.lock() = Some(Default::default());
    *PROFILE_ATOMIC_SITES.lock() = Some(Default::default());
    let _ = crate::thread::spawn_host(
        "gpu-profiler",
        1 << 20,
        Box::new(|| {
            crate::log::set_thread_name("gpu-profiler");
            crate::thread::sampler_priority();
            let period = PROFILE_MS.load(Ordering::Relaxed);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(period));
                // `DS_PROFILE_ACB=<text>`: only the compute queues whose name
                // (`acb0x50`) holds the text are sampled.
                static ACB_ONLY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
                let only = ACB_ONLY.get_or_init(|| std::env::var("DS_PROFILE_ACB").ok());
                let mut targets: Vec<(&'static str, String, u32)> = QUEUES
                    .lock()
                    .iter()
                    .map(|q| if q.id == GFX { ("gfx", "gfx".to_string(), q.tid.load(Ordering::Relaxed)) } else { ("acb", format!("acb{:#x}", q.id), q.tid.load(Ordering::Relaxed)) })
                    .filter(|t| t.0 != "acb" || only.as_ref().is_none_or(|o| t.1.contains(o.as_str())))
                    .collect();
                targets.extend(PROFILE_THREADS.lock().iter().map(|&(kind, tid)| (kind, kind.to_string(), tid)));
                for (kind, name, tid) in targets {
                    let Some((frames, _)) = crate::exception::sample_frames(tid, 0) else { continue };
                    let key: Vec<String> = frames.iter().take(4).map(|&a| crate::exception::host_symbol(a)).collect();
                    let k = format!("{name}: {}", key.join(" <- "));
                    if let Some(m) = PROFILE.lock().as_mut() {
                        *m.entry(k).or_default() += 1;
                    }
                    let labels: Vec<String> = frames.iter().map(|&a| crate::exception::frame_label(a)).collect();
                    let mut fns: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                    let waiting = fns.iter().any(|f| f.contains("Wait") || f.contains("Sleep") || f.contains("DelayExecution") || f.contains("SetTimer") || f.contains("CreateWaitableTimer"));
                    let is_ours = |f: &str| (f.starts_with("dsport::") || f.starts_with("rdna::")) && !f.contains("parking_lot");
                    let at = fns.iter().position(|f| is_ours(f));
                    let ours = at.map_or("(none)", |i| fns[i]);
                    // (A thread whose innermost function of ours is its entry
                    // runs none of our work: a worker idle in an unnamed
                    // system wait, ~92% of the workers' samples counted as
                    // busy before.)
                    let waiting = waiting || ours == "dsport::thread::spawn_host::proc_";
                    if !waiting {
                        if let Some(m) = PROFILE_BUSY.lock().as_mut() {
                            *m.entry((kind, ours.to_string())).or_default() += 1;
                        }
                        // What our innermost function was calling (itself when
                        // it was running its own code).
                        let callee = match at {
                            Some(i) if i > 0 => fns[i - 1],
                            Some(i) => fns[i],
                            None => fns.last().copied().unwrap_or("(none)"),
                        };
                        if let Some(m) = PROFILE_CALLS.lock().as_mut() {
                            *m.entry((kind, format!("{callee} <- {ours}"))).or_default() += 1;
                        }
                        // The line our innermost function was at (a return
                        // address less one: the call's line).
                        if let Some(i) = at {
                            if let Some(line) = crate::exception::host_line(frames[i].saturating_sub((i > 0) as u64)) {
                                if let Some(m) = PROFILE_LINES.lock().as_mut() {
                                    *m.entry((kind, format!("{line} {}", ours.rsplit("::").next().unwrap_or(ours)))).or_default() += 1;
                                }
                            }
                        }
                        if let Some(m) = PROFILE_LEAF.lock().as_mut() {
                            *m.entry((kind, fns.first().copied().unwrap_or("(none)").to_string())).or_default() += 1;
                        }
                        if let Some(&rip) = frames.first() {
                            let (class, site) = sampled_instruction(rip);
                            if let Some(m) = PROFILE_ATOMIC.lock().as_mut() {
                                let e = m.entry(kind).or_default();
                                e[0] += 1;
                                if class > 0 {
                                    e[class as usize] += 1;
                                }
                            }
                            if matches!(class, 1 | 2) {
                                let line = crate::exception::host_line(site).unwrap_or_default();
                                let key = format!("{} @ {line} (in {ours})", crate::exception::frame_label(site));
                                if let Some(m) = PROFILE_ATOMIC_SITES.lock().as_mut() {
                                    *m.entry((kind, key)).or_default() += 1;
                                }
                            }
                        }
                    } else if fns.iter().any(|f| f.contains("lock_slow") || f.contains("AcquireSRWLock")) {
                        if let Some(m) = PROFILE_LOCKS.lock().as_mut() {
                            *m.entry((kind, ours.to_string())).or_default() += 1;
                        }
                    }
                    fns.sort_unstable();
                    fns.dedup();
                    if let Some((m, n)) = PROFILE_FN.lock().as_mut() {
                        *n.entry(kind).or_default() += 1;
                        for f in fns {
                            *m.entry((kind, f.to_string())).or_default() += 1;
                        }
                    }
                }
            }
        }),
    );
}

/// Clears the samples and dispatch timings (`profile reset` control command).
pub fn reset_profile() {
    crate::displayclock::reset_cadence();
    vk::present::reset_stats();
    slotcensus::reset();
    cp::reset_classes();
    exec::wb_census_reset();
    exec::replay_census_reset();
    exec::cpw_census_reset();
    pm4_census_reset();
    fe3_probe::reset();
    kco::reset();
    if let Some(m) = PROFILE.lock().as_mut() {
        m.clear();
    }
    if let Some((m, n)) = PROFILE_FN.lock().as_mut() {
        m.clear();
        n.clear();
    }
    if let Some(m) = PROFILE_BUSY.lock().as_mut() {
        m.clear();
    }
    if let Some(m) = PROFILE_LOCKS.lock().as_mut() {
        m.clear();
    }
    for p in [&PROFILE_CALLS, &PROFILE_LEAF, &PROFILE_LINES, &PROFILE_ATOMIC_SITES] {
        if let Some(m) = p.lock().as_mut() {
            m.clear();
        }
    }
    if let Some(m) = PROFILE_ATOMIC.lock().as_mut() {
        m.clear();
    }
    exec::reset_shader_profile();
    exec::reset_batch_misses();
    cp::reset_wait_stats();
    par::reset_stats();
    cp::past_wait_reset();
    lockstat::reset_stats();
    images::reset_memo_stats();
    exec::reset_pre_window();
    exec::reset_import_window();
    xwait::reset();
    census::reset();
    smalltrack::reset();
    RUN_AHEAD.lock().clear();
    frameprobe::reset();
    runprobe::reset();
    crate::mem::reset_host_calls();
    threadcpu::reset_window();
    crate::audio::reset_window();
    crate::guestprof::reset();
    fe2::reset_callers();
    resident::reset_backing();
    fe2census::reset();
    exec::reset_f3_regions();
}

/// The profile so far (`profile report` control command).
pub fn profile_report() -> String {
    let mut s = profile_summary();
    s.push_str(&exec::shader_profile());
    s.push_str(&drawcensus::lines(exec::profile_flips()));
    s.push_str(&exec::laps_report());
    s.push_str(&images::tsharp_diff_line());
    s.push_str(&shaders::prep_memo_line());
    s.push_str(&fe3_probe::line(exec::profile_flips()));
    s.push_str(&cp::wait_stats());
    s.push_str(&cp::classes_line());
    s.push_str(&cp::op_census_line());
    s.push_str(&cp::cb_census_line());
    s.push_str(&exec::write_logs_line());
    s.push_str(&exec::f3_regions_line());
    s.push_str(&exec::race_line());
    s.push_str(&exec::unordered_mirror_line());
    s.push_str(&forward::status_line());
    s.push_str(&fe2::callers_line());
    s.push_str(&resident::backing_line());
    s.push_str(&fe2census::report());
    s.push_str(&crate::hle::kernel::timeouts_line());
    s.push_str(&cp::long_waits_line());
    s.push_str(&cp::past_wait_status());
    s.push_str(&par::status());
    s.push_str(&images::memo_stats_line());
    s.push_str(&exec::pre_window_line());
    if !exec::profiling_on() {
        // (With DS_PROFILE_GPU=1 the shader profile has them.)
        s.push_str(&crate::mem::ww_lines());
    }
    s.push_str(&exec::push_line());
    s.push_str(&slotcensus::line());
    if shaders::r2_check() {
        let c: Vec<u64> = shaders::R2_CHECKS.iter().map(|x| x.load(Ordering::Relaxed)).collect();
        s.push_str(&format!("    R2 checks (`DS_R2_CHECK`): {} template hits checked, {} mismatches
", c[0], c[1]));
    }
    if shaders::armed_census_on() {
        let pct = |a: u64, b: u64| a as f64 * 100.0 / b.max(1) as f64;
        let parts: Vec<String> = ["vertex", "pixel", "compute"]
            .iter()
            .zip(shaders::ARMED_CENSUS.iter())
            .map(|(name, st)| {
                let c: Vec<Vec<u64>> = st.iter().map(|k| k.iter().map(|x| x.load(Ordering::Relaxed)).collect()).collect();
                format!(
                    "{name}: descriptors {} ranges, {:.1}% armed ({:.1}% of {} KiB), scalar loads {} ranges, {:.1}% armed ({:.1}% of {} KiB)",
                    c[0][0],
                    pct(c[0][1], c[0][0]),
                    pct(c[0][3], c[0][2]),
                    c[0][2] >> 10,
                    c[1][0],
                    pct(c[1][1], c[1][0]),
                    pct(c[1][3], c[1][2]),
                    c[1][2] >> 10
                )
            })
            .collect();
        s.push_str(&format!("    preparations' reads in armed pages (`DS_R2_ARMED_CENSUS`): {}
", parts.join("; ")));
    }
    s.push_str(&exec::async_line());
    s.push_str(&shaders::fmt_line());
    s.push_str(&vk::framewatch::line());
    s.push_str(&exec::worker_progs_line());
    s.push_str(&crate::pcsettings::pointer::direct_line());
    s.push_str(&crate::displayclock::status_line());
    s.push_str(&vk::present::stats_line());
    s.push_str(&exec::import_window_line());
    s.push_str(&crate::mem::host_calls_line());
    s.push_str(&xwait::status());
    if census::on() {
        s.push_str(&census::status());
    }
    if smalltrack::on() {
        s.push_str(&smalltrack::report());
    }
    s.push_str(&crate::audio::window_line());
    s.push_str(&crate::hle::libs::acm_conv::status());
    s.push_str(&crate::guestprof::report());
    s.push_str(&threadcpu::window_line());
    s.push_str(&run_ahead_line());
    if frameprobe::on() {
        s.push_str(&frameprobe::report());
    }
    if runprobe::on() {
        s.push_str(&runprobe::report());
    }
    s
}

fn profile_summary() -> String {
    let g = PROFILE.lock();
    let Some(m) = g.as_ref() else { return String::new() };
    let mut v: Vec<(&String, &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let mut s = format!("    gpu profile ({} ms samples):\n", PROFILE_MS.load(Ordering::Relaxed));
    for (k, n) in v.iter().take(20) {
        s.push_str(&format!("      {n:6}  {k}\n"));
    }
    drop(g);
    // Inclusive time per function of our crate (and the waits), per queue kind.
    if let Some((m, totals)) = PROFILE_FN.lock().as_ref() {
        let top = std::env::var("DS_PROFILE_FN_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(40);
        for kind in PROFILE_KINDS {
            let total = totals.get(kind).copied().unwrap_or(0).max(1);
            let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, f), _)| *k == kind && (f.starts_with("dsport::") || f.contains("Wait") || f.contains("Sleep") || f.contains("Submit"))).map(|((_, f), n)| (f, n)).collect();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            s.push_str(&format!("    gpu profile {kind}: {total} samples, inclusive share per function:\n"));
            for (f, n) in v.iter().take(top) {
                s.push_str(&format!("      {:5.1}%  {f}\n", **n as f64 * 100.0 / total as f64));
            }
        }
    }
    if let Some(m) = PROFILE_BUSY.lock().as_ref() {
        for kind in PROFILE_KINDS {
            let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, _), _)| *k == kind).map(|((_, f), n)| (f, n)).collect();
            let busy: u64 = v.iter().map(|x| *x.1).sum();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            s.push_str(&format!("    gpu profile {kind} busy (not waiting): {busy} samples, by innermost function of ours:\n"));
            // (`DS_PROFILE_BUSY_TOP`: how many functions, default 30.)
            let busy_top = std::env::var("DS_PROFILE_BUSY_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(30);
            for (f, n) in v.iter().take(busy_top) {
                s.push_str(&format!("      {:6.2}%  {f}\n", **n as f64 * 100.0 / busy.max(1) as f64));
            }
        }
    }
    // (`DS_PROFILE_LINES_TOP`: how many source lines, default 50.)
    let lines_top = std::env::var("DS_PROFILE_LINES_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(50);
    for (table, what, n) in [(&PROFILE_CALLS, "by callee of our innermost function", 40), (&PROFILE_LEAF, "by innermost frame", 25), (&PROFILE_LINES, "by source line of our innermost function", lines_top)] {
        if let Some(m) = table.lock().as_ref() {
            for kind in PROFILE_KINDS {
                let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, _), _)| *k == kind).map(|((_, f), n)| (f, n)).collect();
                let busy: u64 = v.iter().map(|x| *x.1).sum();
                if busy == 0 {
                    continue;
                }
                v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                s.push_str(&format!("    gpu profile {kind} busy (not waiting): {busy} samples, {what}:\n"));
                for (f, c) in v.iter().take(n) {
                    s.push_str(&format!("      {:5.1}%  {f}\n", **c as f64 * 100.0 / busy as f64));
                }
            }
        }
    }
    if let Some(m) = PROFILE_ATOMIC.lock().as_ref() {
        let sites = PROFILE_ATOMIC_SITES.lock();
        for kind in PROFILE_KINDS {
            let Some(e) = m.get(kind) else { continue };
            let pct = |x: u64| x as f64 * 100.0 / e[0].max(1) as f64;
            s.push_str(&format!(
                "    gpu profile {kind} busy samples by instruction: {} busy, at a locked instruction {} ({:.1}%), right after one {} ({:.1}%), at a spin-wait pause {} ({:.1}%)
",
                e[0], e[1], pct(e[1]), e[2], pct(e[2]), e[3], pct(e[3])
            ));
            if let Some(sm) = sites.as_ref() {
                let mut v: Vec<(&String, &u64)> = sm.iter().filter(|((k, _), _)| *k == kind).map(|((_, f), n)| (f, n)).collect();
                v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                for (f, n) in v.iter().take(30) {
                    s.push_str(&format!("      {:5.2}%  {f}
", pct(**n)));
                }
            }
        }
    }
    if let Some(m) = PROFILE_LOCKS.lock().as_ref() {
        for kind in PROFILE_KINDS {
            let mut v: Vec<(&String, &u64)> = m.iter().filter(|((k, _), _)| *k == kind).map(|((_, f), n)| (f, n)).collect();
            let total: u64 = v.iter().map(|x| *x.1).sum();
            v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            s.push_str(&format!("    gpu profile {kind} lock waits: {total} samples, by innermost function of ours:\n"));
            for (f, n) in v.iter().take(15) {
                s.push_str(&format!("      {n:6}  {f}\n"));
            }
        }
    }
    s
}

/// One line per queue for hang reports.
pub fn status() -> String {
    status_ex(false)
}

/// Queue lines, with host stack samples of the command processors.
pub fn status_ex(with_stacks: bool) -> String {
    let mut s = String::new();
    let qs: Vec<&'static Queue> = match QUEUES.try_lock() {
        Some(g) => g.clone(),
        None => return s,
    };
    for q in qs {
        let pending = q.pending.try_lock().map(|g| g.len().to_string()).unwrap_or_else(|| "?".into());
        let name = if q.id == GFX { "gpu-gfx".to_string() } else { format!("gpu-acb{:#x}", q.id) };
        let state = cp::STATE.try_lock().and_then(|g| g.iter().find(|(n, _)| *n == name).map(|(_, s)| s.clone())).unwrap_or_else(|| "idle".into());
        s.push_str(&format!("    {name}: {} submission(s) done, {pending} pending; {state}\n", q.done.load(Ordering::Relaxed)));
        if with_stacks {
            if let Some(st) = crate::exception::sample_thread(q.tid.load(Ordering::Relaxed), 16) {
                s.push_str(&format!("        {st}\n"));
            }
        }
    }
    let hidden = vk::present::hidden_line();
    if !hidden.is_empty() {
        s.push_str(&format!("    {hidden}
"));
    }
    s.push_str(&format!(
        "    gpu totals: {} submits, {} packets, {} draws ({} from {} multi-draw packets, {} of them with a GPU-written count; {} indexed indirect draws dropped, their arguments past the index buffer, {} drawn as they read later), {} dispatches, {} EOP interrupts, {} flips, {} label waits latched, {} passed on writes made after their submission, {} on counters past their value, {} passed on promises, {} CP reads waited for another queue{}\n",
        SUBMITS.load(Ordering::Relaxed),
        cp::PACKETS.load(Ordering::Relaxed),
        cp::DRAWS.load(Ordering::Relaxed),
        cp::MDI_DRAWS.load(Ordering::Relaxed),
        cp::MDI_PACKETS.load(Ordering::Relaxed),
        cp::MDI_COUNT_FROM_MEMORY.load(Ordering::Relaxed),
        cp::ARGS_BEYOND[0].load(Ordering::Relaxed),
        cp::ARGS_BEYOND[1].load(Ordering::Relaxed),
        cp::DISPATCHES.load(Ordering::Relaxed),
        cp::EOP_INTERRUPTS.load(Ordering::Relaxed),
        FRAMES.load(Ordering::Relaxed),
        cp::LATCHED.load(Ordering::Relaxed),
        cp::MISSED.load(Ordering::Relaxed),
        cp::COUNTER_PASSED.load(Ordering::Relaxed),
        cp::PROMISED.load(Ordering::Relaxed),
        exec::FOREIGN_WAITS.load(Ordering::Relaxed),
        format!(
            "; {} read-only buffers bound as snapshots of guest memory taken anew ({} of them tiny ones: the copy kernel's inputs, and others nothing else placed){}",
            exec::TINY_SNAPS.load(Ordering::Relaxed),
            exec::TINY_BOUND.load(Ordering::Relaxed),
            if exec::UPLOAD_HAZARDS[2].load(Ordering::Relaxed) > 0 || exec::SUBMIT_HAZARDS[0].load(Ordering::Relaxed) > 0 {
                format!(
                    "; {} uploads into the mirror over pages an unfinished batch writes ({} of them the other device queue's); at closed batches' submission {} ({} over pages front end 4 holds)",
                    exec::UPLOAD_HAZARDS[2].load(Ordering::Relaxed),
                    exec::UPLOAD_HAZARDS[0].load(Ordering::Relaxed),
                    exec::SUBMIT_HAZARDS[0].load(Ordering::Relaxed),
                    exec::SUBMIT_HAZARDS[1].load(Ordering::Relaxed)
                )
            } else {
                String::new()
            }
        )
    ));
    s.push_str(&format!("    {}\n", throttle_status()));
    s.push_str(&snap_line());
    s.push_str(&pm4_census_line());
    s.push_str(&exec::status());
    s.push_str(&pagetable::status());
    s.push_str(&gpures::status());
    s.push_str(&profile_summary());
    s.push_str(&exec::shader_profile());
    s.push_str(&zk::status_line());
    #[cfg(target_os = "linux")]
    s.push_str(&format!("{}\n", crate::platform::os::dma_status()));
    {
        let k = images::Images::keyed_views();
        if k[0] + k[1] > 0 {
            s.push_str(&format!("    compute images bound in their instructions' dimension (`shaders::d2_resources`): 2D textures as 1-layer arrays {}, arrays' base slices as 2D {}
", k[0], k[1]));
        }
    }
    s
}

#[cfg(target_os = "linux")]
#[path = "../platform/linux/framegen.rs"]
pub mod framegen;
