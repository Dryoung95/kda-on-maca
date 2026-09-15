# ncu → mcTracer Metric Mapping

How each upstream Nsight Compute metric maps onto the MACA toolchain. Use this
when porting analysis text between the NVIDIA and MACA backends, and to check
whether a number you want to cite exists at all.

**Legend:**
- **✅ direct** — mcTracer provides a semantically equivalent value.
- **🔶 derived** — computed by `mctracer_utils` from trace fields + device
  constants.
- **🔶 runtime-reported** — the MACA runtime reports it in the trace.
- **❌ gap** — no equivalent; see `11-maca-proxies.md`.

---

## Launch geometry

| ncu metric | mcTracer source | Status |
|---|---|---|
| `launch__grid_size` | `grid.x * grid.y * grid.z` | 🔶 derived |
| `launch__block_size` | `block.x * block.y * block.z` | 🔶 derived |
| `launch__grid_dim_x/y/z` | `args.grid.x/y/z` | ✅ direct |
| `launch__block_dim_x/y/z` | `args.block.x/y/z` | ✅ direct |
| `launch__thread_count` | grid_size * block_size | 🔶 derived |
| `launch__registers_per_thread` | `args.mem.registers_per_thread` | ✅ direct |
| `launch__shared_mem_per_block_static` | `args.mem.static_shared` | ✅ direct |
| `launch__shared_mem_per_block_dynamic` | `args.mem.dynamic_shared` | ✅ direct |
| `launch__shared_mem_per_block` | static + dynamic | 🔶 derived |
| `launch__occupancy_limit_blocks` | device constant (32) | 🔶 derived |
| `launch__occupancy_limit_registers` | `regs_per_sm // (regs * threads)` | 🔶 derived |
| `launch__occupancy_limit_shared_mem` | `smem_per_sm // smem_per_block` | 🔶 derived |
| `launch__occupancy_limit_warps` | `max_threads_per_sm // threads` | 🔶 derived |
| `launch__waves_per_multiprocessor` | `grid_size / wave_size` | 🔶 derived |
| `device__attribute_multiprocessor_count` | 104 (from `torch` device props) | 🔶 derived |

Device constants used by the derivations (verified on this board via
`torch.cuda.get_device_properties(0)`):

```
multiprocessor_count            = 104
max_threads_per_multiprocessor  = 2048
regs_per_multiprocessor         = 131072
shared_memory_per_block         = 65536
shared_memory_per_multiprocessor= 65536
warp_size                       = 64
l2_cache_size                   = 8388608
total_memory                    = 16341008384
```

---

## Timing

| ncu metric | mcTracer source | Status |
|---|---|---|
| `gpu__time_duration.sum` | event `dur` (ns, device-side) | ✅ direct |
| (kernel start) | event `ts` | ✅ direct |
| — (no ncu equivalent) | `queue_ts`, `submit_ts` (host enqueue) | ✅ direct |
| `smsp__cycles_active.avg` | — | ❌ gap |

**Critical:** `complete_ts - queue_ts` is the queue lifecycle, not execution
time. Observed discrepancy on a PyTorch workload: 432 µs (queue lifecycle) vs
86 µs (`dur`) for the same kernel. Report `dur`.

---

## Occupancy

| ncu metric | mcTracer source | Status |
|---|---|---|
| `sm__maximum_warps_per_active_cycle_pct` | derived from occupancy limits | 🔶 derived |
| `sm__warps_active.avg.pct_of_peak_sustained_active` (achieved) | — | ❌ gap |
| — | `args["mtreg_occupancy(%)"]` | 🔶 runtime-reported |
| — | `args["shared_memeory_occupancy(%)"]` (sic) | 🔶 runtime-reported |

The two runtime-reported fields are MACA-native: they have no ncu name and
must be cited by their literal key (typos included — the field names in the
trace are spelled that way).

---

## Throughput / SOL

| ncu metric | mcTracer source | Status |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | — | ❌ gap |
| `gpu__compute_memory_throughput.avg.*` | — | ❌ gap |
| `dram__bytes_read.sum` / `dram__bytes_write.sum` | — | ❌ gap |
| `dram__bytes_read.sum.per_second` | — | ❌ gap |
| `l1tex__t_sector_hit_rate.pct` | — | ❌ gap |
| `lts__t_sector_hit_rate.pct` | — | ❌ gap |
| `bytes` (device copies only) | `args.bytes` on `mcMemcpy` events | ✅ direct |

The only byte counts available are for explicit `mcMemcpy` calls — a kernel's
own load/store traffic is invisible. Roofline (with a supplied op count) is
the sanctioned substitute.

---

## Compute pipes

| ncu metric | mcTracer source | Status |
|---|---|---|
| `sm__inst_executed_pipe_fma.avg.*` | — | ❌ gap |
| `sm__pipe_tensor_cycles_active.avg.*` | — | ❌ gap |
| `sm__ops_path_tensor_op_hmma_*` | — | ❌ gap |

Tensor-core usage must be inferred from kernel source or from the kernel
name (`mcblas__Mck_tf32gemm_...` in the trace indicates the library GEMM
path).

---

## Stall reasons (all gapped)

| ncu metric | Status |
|---|---|
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | ❌ gap |
| `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` | ❌ gap |
| `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` | ❌ gap |
| `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` | ❌ gap |
| `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` | ❌ gap |
| `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` | ❌ gap |
| `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio` | ❌ gap |
| `smsp__pcsamp_sample_count` | ❌ gap |
| `smsp__pcsamp_warps_issue_stalled_*` (per-PC) | ❌ gap |

No per-PC sampling exists, so there is also no `pc_to_source_line` analogue —
source attribution comes from reading the harness source, not from the trace.

---

## Instruction counts / spill

| ncu metric | mcTracer source | Status |
|---|---|---|
| `smsp__sass_inst_executed_op_global_ld.sum` | — | ❌ gap |
| `smsp__sass_inst_executed_op_global_st.sum` | — | ❌ gap |
| `smsp__sass_inst_executed_op_local_ld.sum` (spill) | — | ❌ gap |
| `smsp__sass_inst_executed_op_shared.sum` | — | ❌ gap |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio` | — | ❌ gap |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | — | ❌ gap |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | — | ❌ gap |

Register pressure is visible only as `launch__registers_per_thread` plus the
resulting occupancy limit — not as a spill count.

---

## NCU rule engine

| ncu feature | mcTracer source | Status |
|---|---|---|
| `--page details` rule suggestions | — | ❌ gap |
| `Est. Speedup: X%` | — | ❌ gap |
| `rule_results_as_dicts()` | — | ❌ gap |

Rank findings with wave math and controlled experiments instead, and show the
arithmetic in the report.

---

## Summary

Of the upstream curated `B200_KEY_METRICS` list (~90 names):

- **✅ direct or 🔶 derived:** 20 (all of launch geometry, timing, and the
  runtime-reported occupancies)
- **❌ gap:** ~70 (stall reasons, cache hit rates, pipe utilization, sector
  counts, spill detection, rule engine)

The four fully-supported analysis dimensions (launch geometry, timeline,
inventory, roofline) cover upstream Patterns A, B, F, J, K, L, M, N with
varying observability. Patterns C, D, E, G, H, I require source inspection
plus a confirmation experiment.
