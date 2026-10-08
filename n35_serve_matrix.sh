#!/bin/bash
# Parametrised serve for the profiler matrix (n36_prof_matrix.sh).
#
# This is n04_serve_target.sh with exactly two things made switchable:
#   LORA=1|0      whether --enable-lora + the three adapters are passed at all
#   OVERLAP=0|1   VLLM_ASCEND_LORA_MOE_OVERLAP (only read on the LoRA path, so
#                 it is meaningless when LORA=0)
# Every other flag is byte-identical to n04_serve_target.sh on purpose: the
# earlier TPOT scoreboards (logs/night/fast_*) were taken with that config, so
# keeping it identical is what lets the profile numbers be read next to them.
#
#   LORA=1 OVERLAP=0 PROF=/tmp/prof_matrix/<tag> bash n35_serve_matrix.sh
set -u
export VLLM_DISABLE_COMPILE_CACHE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_LORA_MOE_OVERLAP=${OVERLAP:-0}
export VLLM_SERVER_DEV_MODE=1
sysctl -w vm.swappiness=0 >/dev/null
sysctl -w kernel.numa_balancing=0 >/dev/null
sysctl kernel.sched_migration_cost_ns=50000 >/dev/null

LORA=${LORA:-1}
PORT=${PORT:-1995}
PROF=${PROF:-/tmp/ds_prof_matrix}
# overridable so the matrix runs on a machine with other paths
MODEL=${MODEL:-/data/models/DeepSeek-V4-Flash-0731-w8a8}
L=${LORA_ZERO:-/data/models/DeepSeek-V4-Flash-0731-lora-zero-rank16}
R=${LORA_RAND:-/data/models/DeepSeek-V4-Flash-0731-lora-rand16}

ARGS=(
  "${MODEL}"
  --safetensors-load-strategy 'prefetch'
  --max-model-len 90000
  --max-num-batched-tokens 16384
  --served-model-name ds
  --gpu-memory-utilization 0.9
  --max-num-seqs 8
  --data-parallel-size 1
  --tensor-parallel-size 8
  --enable-expert-parallel
  --quantization ascend
  --port "${PORT}"
  --block-size 32
  --enable-chunked-prefill
  --enable-prefix-caching
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --async-scheduling
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding": "true","multistream_overlap_shared_expert":false}'
)

if [ "${LORA}" = "1" ]; then
    ARGS+=(
      --enable-lora
      --max-loras 3
      --max-lora-rank 16
      --lora-modules "lora-adapter1=$L" "lora-adapter2=$L" "lora-rand=$R"
    )
fi

ARGS+=(
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${PROF}\", \"torch_profiler_with_stack\": false}"
)

echo "n35_serve_matrix: LORA=${LORA} OVERLAP=${VLLM_ASCEND_LORA_MOE_OVERLAP} PROF=${PROF} PORT=${PORT}"
echo "n35_serve_matrix: MODEL=${MODEL}"
if [ "${DRY:-0}" = "1" ]; then
    # print one argument per line so quoting bugs are visible without paying
    # the ~8 min model load to find them
    printf 'vllm serve\n'; printf '  [%s]\n' "${ARGS[@]}"; exit 0
fi
exec vllm serve "${ARGS[@]}"
