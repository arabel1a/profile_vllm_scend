#!/bin/bash
# Clean rebuild of vllm-ascend INCLUDING the AscendC custom operators.
#
# WHY THIS EXISTS, and when to use it instead of rebuild_va.sh
# -----------------------------------------------------------
# rebuild_va.sh removes build/ dist/ *.egg-info. That is enough for the Python
# extension but NOT for the custom ops: the AscendC op packages are built under
# csrc/build and csrc/build_out, which rebuild_va.sh leaves in place, and the
# kernel compile step is stamped done after its first successful run.
#
# A kernel that lives entirely in a header -- e.g.
# csrc/moe/scatter_nd_update_v2/op_kernel/*.h -- has no build target depending
# on it, so it is NEVER recompiled by an incremental build. The build prints
# success in seconds while reinstalling the previously compiled .o.
#
# Measured on 2026-10-06: six consecutive builds reported EXIT=0 while all six
# shipped a kernel binary compiled four days earlier, which made six different
# source variants produce byte-identical results. Hence the guard at the end.
#
# Use rebuild_va.sh for Python or host-side C++ changes.
# Use this one whenever an AscendC kernel changed.
set -euo pipefail

VLLM_ASCEND_PATH=/vllm-workspace/vllm-ascend/vllm_ascend
cd "$VLLM_ASCEND_PATH/.."

NEWEST_SRC=$(find csrc -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.hpp' \) \
             -printf '%T@\n' | sort -n | tail -1)

rm -rf build/ dist/ ./*.egg-info
rm -rf csrc/build csrc/build_out          # <-- the part rebuild_va.sh misses

# third_party ascend_protobuf is a cmake ExternalProject whose inner
# `cmake --build .` takes no -j; without these it crawls on 1 core of 192 and
# looks hung.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-64}"
export MAKEFLAGS="${MAKEFLAGS:--j64}"

COMPILE_CUSTOM_KERNELS=1 SOC_VERSION="${SOC_VERSION:-ascend910b1}" MAX_JOBS="${MAX_JOBS:-128}" \
    python setup.py build_ext --inplace

python -c "
import torch, torch_npu, vllm_ascend.vllm_ascend_C
for op in ('add_lora', 'add_lora_shrink', 'add_lora_expand', 'npu_scatter_nd_update_v2'):
    print(f'{op:26s} present:', hasattr(torch.ops._C_ascend, op))
"

# ---- KERNEL FRESHNESS GUARD -------------------------------------------------
# The whole point of the clean rebuild. Silence here is not success: compare the
# installed kernel binaries against the newest kernel source.
OLDEST_OBJ=$(find vllm_ascend/_cann_ops_custom -name '*.o' -printf '%T@\n' 2>/dev/null \
             | sort -n | head -1)
if [ -z "$OLDEST_OBJ" ]; then
    echo "KERNEL FRESHNESS: FAIL -- no custom-op .o installed at all"
    exit 1
fi
if ! awk "BEGIN{exit !($OLDEST_OBJ > $NEWEST_SRC)}"; then
    echo "KERNEL FRESHNESS: FAIL -- installed kernel .o are OLDER than csrc sources."
    echo "  You would be running a stale kernel. Do not trust any measurement."
    find vllm_ascend/_cann_ops_custom -name '*.o' -printf '%TY-%Tm-%Td %TH:%TM %p\n' \
        | sort | head -3
    exit 1
fi
echo "KERNEL FRESHNESS: OK -- every installed kernel .o is newer than every csrc source"
