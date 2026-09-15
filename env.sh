#!/usr/bin/env bash
# KDA-MACA environment: MACA SDK + cu-bridge + mcTracer profiling toolchain.
# Source this before any KDA-MACA command:  source env.sh

export MACA_PATH="${MACA_PATH:-/opt/maca-3.3.0}"
# /opt/maca is the conventional symlink to the installed SDK version.
if [ -d /opt/maca ]; then export MACA_PATH=/opt/maca; fi
export CUBRIDGE_PATH="${CUBRIDGE_PATH:-$MACA_PATH/tools/cu-bridge}"
export CUDA_HOME="$CUBRIDGE_PATH"
export MCTRACER="$MACA_PATH/bin/mcTracer"
export CUT_STAT="$MACA_PATH/tools/cut_stat"

export PATH="$CUBRIDGE_PATH/bin:$MACA_PATH/bin:$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$MACA_PATH/tools/cu-bridge/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/tools/cu-bridge/lib:${LIBRARY_PATH:-}"
export CPATH="$MACA_PATH/tools/cu-bridge/include:$MACA_PATH/include:${CPATH:-}"

# mcTracer JSON output root (per-run dirs are created under here by the skill)
export KDA_PROFILE_ROOT="${KDA_PROFILE_ROOT:-/tmp/kda_maca_profile}"
mkdir -p "$KDA_PROFILE_ROOT"

# KernelBench harness for validation/evaluation runs
export KERNELBENCH_ROOT="${KERNELBENCH_ROOT:-/data/cuda-harness-migration/KernelBench}"

# cucc needs the cu-bridge macro headers; keep the flag explicit for harness builds.
export MACA_CUCC_FLAGS="-DUSE_MACA -I$MACA_PATH/tools/cu-bridge/include"

# mcTracer --odname must be a RELATIVE path (relative to the cwd). An absolute
# path makes it fail with "Output file open error!". cd into the run dir, then
# pass a bare directory name to --odname.
