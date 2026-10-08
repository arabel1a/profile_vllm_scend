#!/bin/bash
# Profiler-trace matrix over the LoRA branches -- RANK0 ONLY.
#
# ============================================================================
# WHAT IT NEEDS ON THE MACHINE IT RUNS ON
# ============================================================================
# Shipped together (put all three in the same directory, that is all):
#     n36_prof_matrix.sh      this driver
#     n35_serve_matrix.sh     the parametrised serve
#     n37_prof_compare.py     the cross-cell comparison (self-contained)
# Assumed present (your own, unchanged):
#     clean_rebuild_va.sh     next to this script, or set REBUILD=/path
# Needed from the environment:
#     the vllm-ascend git checkout at $VA (default /vllm-workspace/vllm-ascend)
#     with the cell branches reachable (local, or on a remote this script will
#     resolve: <name>, mmo/<name>, mmo/misha_<name>, origin/<name>)
#     vllm, python3 with torch_npu, git, curl, awk, find, setsid, pgrep/pkill
#     the base model at $MODEL and the two adapters $LORA_ZERO / $LORA_RAND
#     the load dataset at $DATASET (default data/dataset.jsonl next to this
#     script -- the kit ships 8 LongAlpaca prompts). REQUIRED: there is no
#     filler fallback, on purpose. Filler text (one sentence repeated to length)
#     is trivially predictable, so the dspark MTP draft accepts nearly every
#     token on it; acceptance is what TPOT gets divided by, so a filler load
#     silently reports an acceptance and a per-step cost no real traffic gives.
# NOT needed:
#     no load.py -- embedded below, byte-identical to the one
#     lora_prof_capture_2.sh writes (md5 4edff3613850853fdd965a4c28677989), so
#     the load matches every earlier round. LOADER=... overrides it.
#     no n22_delta_probe.sh -- inlined.
#     no lora_prof_analyze_2.py -- used for the richer per-cell report if it
#     happens to sit next to this script, skipped otherwise. n37 never needs it.
#
#   bash n36_prof_matrix.sh --smoke     <-- RUN THIS FIRST ON A NEW MACHINE
# does everything except starting a server and waiting on it: every presence and
# permission check, resolves every cell's branch against the actual checkout,
# prints the build plan, and drives the whole post-capture pipeline (prune ->
# claim -> meta -> report -> compare) on synthetic captures. Minutes, no NPU.
#
# ============================================================================
# WHY RANK0 ONLY
# ============================================================================
# One capture window writes ~3.2 GB per rank; TP=8 makes that ~26 GB, and
# /tmp/ds_night_prof had reached 174 GB. torch_npu's analyse() re-parses every
# rank dir it finds, which is the bulk of the ~20 min attribution cost that made
# the earlier cycles drop profiling altogether. Nothing downstream needs the
# other seven ranks: lora_prof_analyze_2.py picks ONE primary rank for every
# number it reports and uses the rest only to print straggler lines. So every
# rank != 0 capture dir is deleted after each window, BEFORE analyse(). That
# ordering is the reason this script captures itself instead of calling
# lora_prof_capture_2.sh, which analyses at the end with all 8 ranks on disk.
# Peak disk is still ~26 GB per window (all ranks exist until the prune).
#
# ============================================================================
# TRACES ARE KEPT FOR MANUAL REVIEW
# ============================================================================
# Every rank0 capture survives the run. After analyse(), each window's capture
# directory is MOVED next to its own csvs, so one cell is one self-contained
# tree you can copy off the box:
#     <OUT_ROOT>/<tag>/<phase>/<arm>/ascend_pt_rank0/
#         ASCEND_PROFILER_OUTPUT/trace_view.json      <- the timeline
#         ASCEND_PROFILER_OUTPUT/kernel_details.csv
#         ...
#     <OUT_ROOT>/<tag>/<phase>/<arm>/kernel_details_rank0.csv   (flat copy)
#     <OUT_ROOT>/traces.tsv      index: tag, phase, arm, trace_view.json path
# KEEP=traces (default) keeps the whole analysed output including the timeline;
# KEEP=all also keeps the unparsed PROF_*/FRAMEWORK dirs, so the capture can be
# re-analysed with other options; KEEP=csv keeps only the csvs (~150 MB/window)
# when disk is tight. Ranks 1-7 are never kept -- that is the point of the
# script -- so a trace shows one rank's timeline, not the collective.
#
# ============================================================================
# WARMUP
# ============================================================================
# Nothing is captured cold. Per cell: an optional cross-arm burst
# (GLOBAL_WARMUP=1, 16 tokens on every arm, so the first arm measured is not the
# one that paid for first-touch weight/adapter loading), then per arm a
# 128-token warmup load that is allowed to finish, then the real load starts and
# runs PRE_WAIT seconds (30 in decode, 3 in prefill) BEFORE start_profile, so
# the window opens on a steady batch. reset_prefix_cache runs after each arm.
#
# WHAT IS NOT MEASURED HERE: client TPOT medians -- that is
# n23_measure_repeats.sh. A profile window perturbs the thing it measures, so
# the load.log TOTAL lines written here are provenance, not a scoreboard.
#
# ============================================================================
# USAGE
# ============================================================================
#   bash n36_prof_matrix.sh --smoke             presence + pipeline check
#   bash n36_prof_matrix.sh                     the full 10-cell matrix
#   CELLS="z1z2_nolora scatter_nolora" bash n36_prof_matrix.sh
#   PHASES="decode prefill" bash n36_prof_matrix.sh
#   DO_BUILD=0 CELLS=scatter_lora_ov1 bash n36_prof_matrix.sh
#
# Env: CELLS PHASES(decode) ARMS_LORA DO_BUILD(1) RUN_UT(0) SKIP_DONE(1)
#      KEEP(traces|all|csv) GLOBAL_WARMUP(1) PREFILL_VARIANTS(4)
#      PARALLEL(8) DATASET FORCE_DATASET(0) PORT(1995) VA REBUILD
#      MODEL LORA_ZERO LORA_RAND OUT_ROOT PROF_ROOT LOADER
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VA=${VA:-/vllm-workspace/vllm-ascend}
REBUILD=${REBUILD:-${SCRIPT_DIR}/clean_rebuild_va.sh}
PORT=${PORT:-1995}
PHASES=${PHASES:-decode}
PARALLEL=${PARALLEL:-8}
DATASET=${DATASET:-${SCRIPT_DIR}/data/dataset.jsonl}
ARMS_LORA=${ARMS_LORA:-"ds lora-adapter1"}
DO_BUILD=${DO_BUILD:-1}
RUN_UT=${RUN_UT:-0}
SKIP_DONE=${SKIP_DONE:-1}
# What to keep per capture window, rank0 only (see "TRACES" in the header):
#   all     everything, incl. the unparsed PROF_*/FRAMEWORK dirs   ~3.2 GB
#   traces  the full analysed output: trace_view.json (timeline),
#           *.db, every csv -- drops only the unparsed raw          ~2.3 GB
#   csv     kernel_details/op_statistic/step_trace/api_statistic     ~150 MB
KEEP=${KEEP:-traces}
[ "${KEEP_RAW:-0}" = "1" ] && KEEP=all
MODEL=${MODEL:-/data/models/DeepSeek-V4-Flash-0731-w8a8}
LORA_ZERO=${LORA_ZERO:-/data/models/DeepSeek-V4-Flash-0731-lora-zero-rank16}
LORA_RAND=${LORA_RAND:-/data/models/DeepSeek-V4-Flash-0731-lora-rand16}
OUT_ROOT=${OUT_ROOT:-${SCRIPT_DIR}/logs/prof_matrix}
PROF_ROOT=${PROF_ROOT:-/tmp/prof_matrix}
LOADER=${LOADER:-}
LOADER_REF_MD5=4edff3613850853fdd965a4c28677989
SERVE_READY_TRIES=${SERVE_READY_TRIES:-150}      # x10s
DRAIN_TRIES=${DRAIN_TRIES:-90}                   # x10s
QUIET_TRIES=${QUIET_TRIES:-40}                   # x15s, dump-settle wait
GLOBAL_WARMUP=${GLOBAL_WARMUP:-1}
PREFILL_VARIANTS=${PREFILL_VARIANTS:-4}
SMOKE=${SMOKE:-0}
[ "${1:-}" = "--smoke" ] && SMOKE=1
RUN_LOG=""

# ---- the matrix ------------------------------------------------------------
# cell = tag:branch:LORA:OVERLAP       (OVERLAP "-" where not applicable)
# Grouped by branch so each clean rebuild serves every cell that needs it.
#   test            stock baseline: bgmv/sgmv LoRA only, no add_lora_fused and
#                   no combined_lora_idx, and no VLLM_ASCEND_LORA_MOE_OVERLAP
#                   env -> overlap is not applicable there.
#   night_scatter   = night_z1z2 + the two ScatterNdUpdateV2 commits and
#                   nothing else, so the two LORA=0 cells are a clean
#                   with/without-scatter A/B on the base model path.
#   addlora_path    LORA=0 deliberately absent: with LoRA off it is the same
#                   code as night_z1z2 LORA=0 (the LoRA kernels never run).
# OVERLAP is only read on the LoRA path, hence "-" on every LORA=0 cell.
MATRIX_ALL=(
  "test_nolora:test:0:-"
  "test_lora:test:1:-"
  "addlora_lora_ov0:addlora_path:1:0"
  "addlora_lora_ov1:addlora_path:1:1"
  "z1z2_lora_ov0:night_z1z2:1:0"
  "z1z2_lora_ov1:night_z1z2:1:1"
  "z1z2_nolora:night_z1z2:0:-"
  "scatter_lora_ov0:night_scatter:1:0"
  "scatter_lora_ov1:night_scatter:1:1"
  "scatter_nolora:night_scatter:0:-"
)

# ---- embedded load generator ----------------------------------------------
# Byte-identical to the one lora_prof_capture_2.sh writes. Keep it that way:
# the load is what makes rounds comparable.
write_loader() {
    cat > "$1" <<'PYEOF'
import json, sys, time, urllib.request, concurrent.futures
port, model, par, max_tok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
mode, dataset, ignore_eos = sys.argv[5], sys.argv[6], sys.argv[7] == "1"
URL = f"http://localhost:{port}/v1/chat/completions"
FILLER = ("This is a filler sentence used to expand the token count for "
          "performance profiling of long-input decode. ") * 420
REQS = None
if mode == "dataset":
    REQS = [json.loads(l)["messages"] for l in open(dataset) if l.strip()]

def hit(i):
    msgs = REQS[i % len(REQS)] if REQS else [{"role": "user", "content": FILLER}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tok,
            "temperature": 0.0, "ignore_eos": ignore_eos, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                continue
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = ev.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if ttft is None and (d.get("content") or d.get("reasoning_content")):
                    ttft = time.time() - t0
            u = ev.get("usage")
            if u and u.get("completion_tokens"):
                n = u["completion_tokens"]
    dt = time.time() - t0
    if ttft is None:
        ttft = dt
    return n, dt, ttft

t_start = time.time()
tots, dts, ttfts = [], [], []
with concurrent.futures.ThreadPoolExecutor(par) as ex:
    for n, dt, ttft in ex.map(hit, range(len(REQS) if REQS else par)):
        tots.append(n); dts.append(dt); ttfts.append(ttft)
        print(f"req{len(tots)-1}: {n} tok ttft={ttft:.2f}s "
              f"decode={dt-ttft:.1f}s tpot={(dt-ttft)/max(n-1,1)*1000:.2f} ms/tok",
              flush=True)

def pct(v, p):
    s = sorted(v)
    return s[min(len(s)-1, int(len(s)*p/100))]

delivered, wall = sum(tots), time.time() - t_start
mean_tpot = sum(d/t for d, t in zip(dts, tots)) / len(tots) * 1000.0
dec_tpot = sum((d-f)/max(t-1, 1) for d, f, t in zip(dts, ttfts, tots)) / len(tots) * 1000.0
print(f"TOTAL delivered={delivered} tok requests={len(tots)} "
      f"wall={wall:.1f}s mean_tpot={mean_tpot:.2f} ms/tok "
      f"ttft_mean={sum(ttfts)/len(ttfts):.2f}s ttft_p50={pct(ttfts,50):.2f}s "
      f"ttft_p90={pct(ttfts,90):.2f}s ttft_max={max(ttfts):.2f}s "
      f"ttft_min={min(ttfts):.2f}s decode_tpot={dec_tpot:.2f} ms/tok", flush=True)
PYEOF
}

# ---- embedded meta writer --------------------------------------------------
# meta.json is what lora_prof_analyze_2.py --meta eats and what n37 reads to
# learn which cell a directory belongs to.
write_meta_json() {   # cell phase "arms" max_tokens capture_secs out
    python3 - "$@" <<'PYEOF'
import json, os, sys
cell, phase, arms_s, maxtok, csecs, out = sys.argv[1:7]
tag, branch, lora, overlap, parallel, dataset, git, serve_log = sys.argv[7:15]
arms = []
for arm in arms_s.split():
    adir = os.path.join(cell, phase, arm)
    csv = os.path.join(adir, "kernel_details_rank0.csv")
    acc, nlines = None, 0
    accf = os.path.join(adir, "acc.txt")
    if os.path.exists(accf):
        parts = open(accf).read().split()
        if len(parts) == 2:
            acc, nlines = float(parts[0]), int(parts[1])
    capf = os.path.join(adir, "capture_dir.txt")
    if not os.path.exists(csv):
        print(f"  !! {tag}/{phase}/{arm}: no rank0 kernel_details csv")
    arms.append({
        "arm": arm,
        "csvs": ([{"path": csv, "rank": 0,
                   "orig_path": (open(capf).read().strip()
                                 if os.path.exists(capf) else None)}]
                 if os.path.exists(csv) else []),
        "load_log": os.path.join(adir, "load.log"),
        "acc_telemetry": acc, "acc_lines": nlines})
meta = {"parallel": int(parallel), "max_tokens": int(maxtok),
        "capture_secs": int(csecs),
        "load_mode": "dataset" if dataset else "filler",
        "dataset": dataset or None, "node": {"git": git},
        "serve_log": serve_log or None, "arms": arms,
        "cell": {"tag": tag, "branch": branch, "lora": lora,
                 "overlap": overlap, "phase": phase, "ranks_kept": [0]}}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(meta, open(out, "w"), indent=1)
print(f"  wrote {out}")
PYEOF
}

# ---- falsifiability probe (inlined from n22_delta_probe.sh) ----------------
# The perf adapters are ZERO-weight, so a DROPPED LoRA delta is invisible and
# reads as a speedup -- exactly the failure mode that would fake a win. lora-rand
# carries real weights, so its answer must differ from the base arm's.
#   expect: ds == lora-adapter1 (zero delta)   and   lora-rand != ds
delta_probe() {
    local p='Explain in one short paragraph why the sky appears blue.' m
    for m in ds lora-adapter1 lora-rand; do
        printf "%-16s " "$m"
        curl -s "http://localhost:${PORT}/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":48,\"temperature\":0.0}" \
          | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['choices'][0]['message']; print(repr((c.get('content') or c.get('reasoning_content') or '')[:150]))" 2>&1 | head -2
    done
}

# ---- helpers ---------------------------------------------------------------
# a cell's branch may exist only on a remote on a fresh checkout
resolve_branch() {
    local b=$1 c
    for c in "refs/heads/$b" "refs/remotes/mmo/$b" "refs/remotes/mmo/misha_$b" \
             "refs/remotes/origin/$b" "refs/remotes/tianchen/$b"; do
        if git -C "${VA}" rev-parse --verify -q "$c" >/dev/null; then
            echo "${c#refs/}"; return 0
        fi
    done
    return 1
}

checkout_branch() {   # local branch if it exists, else create it off the remote
    local b=$1 r
    r=$(resolve_branch "$b") || { echo "   !! branch '$b' not found locally or on mmo/origin/tianchen"; return 1; }
    git -C "${VA}" checkout -- . 2>/dev/null
    if [ "$r" = "heads/$b" ]; then
        git -C "${VA}" checkout -q "$b" || return 1
    else
        echo "   resolved '$b' -> $r (creating local branch)"
        git -C "${VA}" checkout -q -B "$b" "$r" || return 1
    fi
}

stop_serve() {
    pkill -f "vllm serve" 2>/dev/null
    for _ in $(seq 60); do pgrep -f "vllm serve" >/dev/null || break; sleep 2; done
    pgrep -f "vllm serve" >/dev/null && { pkill -9 -f "vllm serve"; sleep 10; }
    for _ in $(seq 30); do
        pgrep -f "VLLM::|EngineCore" >/dev/null || break
        pkill -9 -f "VLLM::|EngineCore" 2>/dev/null; sleep 2
    done
    echo "   serve stopped"
}

phase_params() {   # -> PRE_WAIT CAPTURE_SECS MAX_TOKENS
    case "$1" in
      prefill) echo "3 15 64" ;;
      decode)  echo "30 8 1024" ;;
      *) echo "!! unknown phase $1" >&2; return 1 ;;
    esac
}

# wait until nothing INSIDE $1 has been written for 20s. -mindepth 1 matters:
# the profiler dir's own mtime is fresh the moment a capture dir appears, so
# without it this never goes quiet and the prune would race the dump.
wait_dump_quiet() {
    local d=$1 i
    for i in $(seq "${QUIET_TRIES}"); do
        if [ -z "$(find "$d" -mindepth 1 -newermt '-20 seconds' -print -quit 2>/dev/null)" ]; then
            echo "   dump quiet after ~$((i*15))s"
            return 0
        fi
        sleep 15
    done
    echo "   WARNING: dump still writing after $((QUIET_TRIES*15))s; pruning anyway"
}

# delete every capture dir whose rank is not 0; refuse to act when no rank0 dir
# is present, so a changed dir-naming scheme fails loudly instead of silently
# deleting the capture
prune_to_rank0() {
    local d=$1 n0 other
    n0=$(find "$d" -maxdepth 1 -type d -name '*_rank0_*_ascend_pt' | wc -l)
    if [ "${n0}" -eq 0 ]; then
        echo "   !! no *_rank0_*_ascend_pt dir under $d -- NOT pruning"
        find "$d" -maxdepth 1 -type d -name '*_ascend_pt' | head -3
        return 1
    fi
    other=$(find "$d" -maxdepth 1 -type d -name '*_ascend_pt' ! -name '*_rank0_*' -print)
    if [ -n "${other}" ]; then
        echo "   pruning $(echo "${other}" | wc -l) non-rank0 capture dir(s)"
        echo "${other}" | xargs -r rm -rf
    fi
}

# the one rank0 dir not yet listed in the claim file $2
claim_rank0() {
    local d=$1 claims=$2 new n
    new=$(find "$d" -maxdepth 1 -type d -name '*_rank0_*_ascend_pt' -printf '%f\n' \
          | sort | awk -v cf="${claims}" \
                      'BEGIN{while((getline l < cf) > 0) seen[l]} !($0 in seen)')
    n=$(echo "${new}" | grep -c .)
    if [ "${n}" != "1" ]; then
        echo "   !! expected exactly 1 unclaimed rank0 capture, got ${n}" >&2
        return 1
    fi
    echo "${new}" >> "${claims}"
    echo "${new}"
}

acc_between() {    # serve log, first line, last line -> "mean nsamples"
    local sl=$1 n0=$2 n1=$3
    [ -f "${sl}" ] || { echo ""; return; }
    sed -n "${n0},${n1}p" "${sl}" 2>/dev/null \
      | grep -oE "acceptance length: [0-9.]+" \
      | awk '{s+=$3; n++} END{if(n) printf "%.3f %d", s/n, n}'
}

# A load padded with a repeated sentence is the failure this script most wants
# to avoid: the dspark MTP draft accepts nearly every token of such text, and
# acceptance is what TPOT is divided by, so the whole scoreboard shifts. Measured
# on the two datasets at hand: 8 LongAlpaca prompts -> top 40-char chunk repeats
# x2, covering 0.3% of a prompt; the older padded dataset -> x33, covering 5.5%.
dataset_sanity() {   # 0 = natural prose, 1 = padded
    python3 - "$1" <<'PYEOF'
import collections, json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except Exception as e:
    print(f"unreadable: {e}"); sys.exit(1)
if not rows:
    print("empty"); sys.exit(1)
worst_rep, worst_cov, worst_chunk = 0, 0.0, ""
for r in rows:
    try:
        t = r["messages"][0]["content"]
    except Exception:
        print("rows are not {'messages': [{'role','content'}]}"); sys.exit(1)
    if len(t) < 80:
        continue
    g = collections.Counter(t[j:j + 40] for j in range(0, len(t) - 40, 40))
    chunk, rep = g.most_common(1)[0]
    if rep > worst_rep:
        worst_rep, worst_cov, worst_chunk = rep, rep * 40 / len(t), chunk
print(f"worst 40-char chunk repeats x{worst_rep}, "
      f"covering {worst_cov * 100:.1f}% of a prompt")
if worst_rep >= 8 or worst_cov >= 0.02:
    print(f"  the repeated chunk: {worst_chunk[:60]!r}")
    sys.exit(1)
PYEOF
}

# ---- presence checks (the core of --smoke) ---------------------------------
CHK_FAIL=0; CHK_WARN=0
ck()  { echo "  ok    $*"; }
ckw() { echo "  WARN  $*"; CHK_WARN=$((CHK_WARN+1)); }
ckf() { echo "  FAIL  $*"; CHK_FAIL=$((CHK_FAIL+1)); }

check_env() {
    local f c
    echo "== commands =="
    for c in vllm python3 git curl awk sed find xargs setsid nohup pgrep pkill du df; do
        command -v "$c" >/dev/null && ck "$c" || ckf "$c MISSING"
    done
    command -v npu-smi >/dev/null && ck "npu-smi" || ckw "npu-smi missing (provenance only)"

    echo "== python =="
    python3 -c "import torch, torch_npu" 2>/dev/null \
        && ck "torch + torch_npu import" || ckf "torch/torch_npu do not import"
    python3 -c "from torch_npu.profiler.profiler import analyse" 2>/dev/null \
        && ck "torch_npu.profiler.profiler.analyse" \
        || ckf "analyse() not importable -- finalization would fail"

    echo "== scripts =="
    for f in "${SCRIPT_DIR}/n35_serve_matrix.sh" "${SCRIPT_DIR}/n37_prof_compare.py"; do
        [ -f "$f" ] && ck "$(basename "$f")" || ckf "$(basename "$f") MISSING (ships with this script)"
    done
    [ -f "${REBUILD}" ] && ck "clean_rebuild_va.sh at ${REBUILD}" \
        || { if [ "${DO_BUILD}" = "1" ]; then ckf "clean_rebuild_va.sh NOT at ${REBUILD} (set REBUILD=...)"
             else ckw "no clean_rebuild_va.sh, but DO_BUILD=0"; fi; }
    if [ -f "${SCRIPT_DIR}/lora_prof_analyze_2.py" ]; then
        ck "lora_prof_analyze_2.py present (per-cell reports will be written)"
    else
        ckw "lora_prof_analyze_2.py absent -- per-cell reports skipped, n37 still works"
    fi

    echo "== checkout =="
    if git -C "${VA}" rev-parse --git-dir >/dev/null 2>&1; then
        ck "git checkout at ${VA} ($(git -C "${VA}" rev-parse --short=10 HEAD))"
        local dirty; dirty=$(git -C "${VA}" status --porcelain | wc -l)
        [ "${dirty}" -eq 0 ] && ck "tree clean" \
            || ckw "${dirty} dirty entries -- 'git checkout -- .' will discard tracked edits"
    else
        ckf "no git checkout at ${VA} (set VA=...)"
    fi

    echo "== model and adapters =="
    [ -f "${MODEL}/config.json" ] && ck "base model ${MODEL}" \
        || ckf "no config.json under MODEL=${MODEL}"
    local need_lora=0 spec
    for spec in "${MATRIX_ALL[@]}"; do
        [ "$(echo "${spec}" | cut -d: -f3)" = "1" ] && need_lora=1
    done
    if [ "${need_lora}" = "1" ]; then
        [ -f "${LORA_ZERO}/adapter_config.json" ] && ck "zero adapter ${LORA_ZERO}" \
            || ckf "no adapter_config.json under LORA_ZERO=${LORA_ZERO}
          create it with: python3 gen_zero_lora_ds.py --model-path ${MODEL} --save-path ${LORA_ZERO} --rank 16"
        [ -f "${LORA_RAND}/adapter_config.json" ] && ck "rand adapter ${LORA_RAND}" \
            || ckf "no adapter_config.json under LORA_RAND=${LORA_RAND}
          create it with: python3 gen_rand_lora_ds.py --model-path ${MODEL} --save-path ${LORA_RAND} --rank 16
          (it is the falsifiability probe: with only zero-weight adapters a
           DROPPED delta reads as a speedup)"
    else
        ck "no LoRA cell selected, adapters not needed"
    fi

    echo "== load =="
    if [ -n "${LOADER}" ]; then
        [ -f "${LOADER}" ] && ck "LOADER override ${LOADER}" || ckf "LOADER=${LOADER} not found"
    else
        local tmpl; tmpl=$(mktemp); write_loader "${tmpl}"
        local m; m=$(md5sum "${tmpl}" | cut -d' ' -f1)
        python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "${tmpl}" 2>/dev/null \
            && ck "embedded loader parses" || ckf "embedded loader is not valid python"
        [ "${m}" = "${LOADER_REF_MD5}" ] && ck "embedded loader md5 matches the reference" \
            || ckw "embedded loader md5 ${m} != reference ${LOADER_REF_MD5}"
        rm -f "${tmpl}"
    fi
    if [ -f "${DATASET}" ]; then
        local nd; nd=$(grep -c . "${DATASET}")
        ck "dataset ${DATASET}: ${nd} prompt(s)$(python3 - "${DATASET}" <<'PYEOF'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    n = [len(r["messages"][0]["content"]) for r in rows]
    print(f", {min(n)}-{max(n)} chars (~{min(n)//4}-{max(n)//4} tokens)", end="")
except Exception as e:
    print(f" -- UNREADABLE: {e}", end="")
PYEOF
)"
        [ "${nd}" -ge "${PARALLEL}" ] \
            || ckw "only ${nd} prompt(s) for PARALLEL=${PARALLEL}: slots would share prompts"
        local dsmsg; dsmsg=$(dataset_sanity "${DATASET}")
        if [ $? = 0 ]; then
            ck "prompts are natural prose (${dsmsg})"
        elif [ "${FORCE_DATASET:-0}" = "1" ]; then
            ckw "PADDED/unreadable dataset, allowed by FORCE_DATASET=1: ${dsmsg}"
        else
            ckf "this dataset looks padded or malformed: ${dsmsg}
          Repeated filler makes the MTP draft accept nearly every token, so the
          acceptance TPOT is divided by stops being meaningful. Use the kit's
          data/dataset.jsonl, or set FORCE_DATASET=1 if you know what you want."
        fi
    else
        ckf "no dataset at ${DATASET}
          REQUIRED, and there is no filler fallback by design: filler text is
          trivially predictable, so the MTP draft accepts nearly every token and
          the acceptance that TPOT is divided by stops being meaningful.
          The kit ships data/dataset.jsonl (8 LongAlpaca prompts)."
    fi

    echo "== phases =="
    local p
    for p in ${PHASES}; do
        if phase_params "$p" >/dev/null 2>&1; then
            ck "phase $p -> pre_wait/window/max_tokens $(phase_params "$p")"
        else ckf "bad phase '$p' (decode|prefill)"; fi
    done

    echo "== keep level and disk =="
    local per_win
    case "${KEEP}" in
      all)    per_win=3.2; ck "KEEP=all: raw + analysed output kept per window (~3.2 GB)" ;;
      traces) per_win=2.3; ck "KEEP=traces: full analysed output incl. trace_view.json (~2.3 GB/window)" ;;
      csv)    per_win=0.2; ckw "KEEP=csv: csvs only (~150 MB/window) -- NO timeline to review manually" ;;
      *)      per_win=3.2; ckf "KEEP='${KEEP}' is not one of all|traces|csv" ;;
    esac
    local nwin; nwin=$(python3 -c "
import sys
cells=sys.argv[1].split(';')
nph=len(sys.argv[2].split())
n=0
for c in cells:
    if not c: continue
    n += nph * (len(sys.argv[3].split()) if c.split(':')[2]=='1' else 1)
print(n)" "$(selected_cells | tr '\n' ';')" "${PHASES}" "${ARMS_LORA}")
    echo "  -> ${nwin} capture window(s) total => ~$(python3 -c "print(f'{${nwin}*${per_win}:.0f}')") GB of traces kept under ${OUT_ROOT}"

    echo "== output dirs and disk =="
    mkdir -p "${OUT_ROOT}" "${PROF_ROOT}" 2>/dev/null
    [ -w "${OUT_ROOT}" ] && ck "OUT_ROOT writable ${OUT_ROOT}" || ckf "OUT_ROOT not writable ${OUT_ROOT}"
    [ -w "${PROF_ROOT}" ] && ck "PROF_ROOT writable ${PROF_ROOT}" || ckf "PROF_ROOT not writable ${PROF_ROOT}"
    local avail; avail=$(df -BG --output=avail "${PROF_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "${avail}" ]; then
        [ "${avail}" -ge 60 ] && ck "${avail} GB free under ${PROF_ROOT} (peak need ~30 GB: all 8 ranks exist until the prune)" \
            || ckw "only ${avail} GB free under ${PROF_ROOT}; one window peaks at ~26 GB before the prune"
    fi

    echo "== port and devices =="
    if curl -sf -m 2 -o /dev/null "http://localhost:${PORT}/health" 2>/dev/null; then
        ckw "something is already serving on ${PORT} -- the run will KILL it"
    else
        ck "port ${PORT} free"
    fi
    if command -v npu-smi >/dev/null; then
        # TP=8 needs all eight devices. The devices are shared with other
        # containers, so a neighbour holding memory will make the serve OOM or
        # contend, and this script cannot see their processes.
        local busy ndev
        ndev=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo 0)
        # POSIX awk only (no 3-arg match / gensub): on the HBM lines the last
        # fields are "<used> / <total> |", so $(NF-2)=="/" selects exactly them
        busy=$(npu-smi info 2>/dev/null | awk '
            { n=NF; if (n>4 && $(n-2)=="/" && $(n-1)+0 > 1000 && $(n-3)+0 > 8192) c++ }
            END{print c+0}')
        [ "${ndev}" -ge 8 ] && ck "${ndev} NPUs visible" \
            || ckw "only ${ndev} NPU(s) visible; the serve asks for tensor-parallel-size 8"
        [ "${busy}" = "0" ] && ck "no device holds >8 GB (nobody else is resident)" \
            || ckw "${busy} device(s) hold >8 GB -- another container is using them; expect contention or OOM"
    fi

    echo "== cell branches =="
    local tag branch lora overlap r nb=0
    local -a seen=()
    for spec in "${MATRIX_ALL[@]}"; do
        IFS=: read -r tag branch lora overlap <<<"${spec}"
        if [ -n "${CELLS}" ] && ! echo " ${CELLS} " | grep -q " ${tag} "; then continue; fi
        if r=$(resolve_branch "${branch}" 2>/dev/null); then
            ck "$(printf '%-18s' "${tag}") branch ${branch} -> ${r} ($(git -C "${VA}" rev-parse --short=10 "${r#*/}" 2>/dev/null || git -C "${VA}" rev-parse --short=10 "refs/${r}"))  LORA=${lora} OVERLAP=${overlap}"
        else
            ckf "$(printf '%-18s' "${tag}") branch ${branch} NOT FOUND (tried ${branch}, mmo/${branch}, mmo/misha_${branch}, origin/${branch})"
        fi
        if ! printf '%s\n' "${seen[@]:-}" | grep -qxF "${branch}"; then
            seen+=("${branch}"); nb=$((nb+1))
        fi
    done
    echo "  -> $(selected_cells | wc -l) cell(s), ${nb} clean rebuild(s)"
    local ncell; ncell=$(selected_cells | wc -l)
    local nph; nph=$(echo ${PHASES} | wc -w)
    echo "  -> rough estimate: ${nb} x ~20 min build + ${ncell} x (~8 min serve"
    echo "     + ${nph} phase(s) x arms x ~2 min window + ~4 min analyse)"
}

selected_cells() {
    local spec tag
    for spec in "${MATRIX_ALL[@]}"; do
        tag=$(echo "${spec}" | cut -d: -f1)
        if [ -n "${CELLS}" ] && ! echo " ${CELLS} " | grep -q " ${tag} "; then continue; fi
        echo "${spec}"
    done
}

# ---- smoke: the whole post-capture pipeline on synthetic captures ----------
# A synthetic kernel_details.csv with the marker kernels the analysis normalises
# by, so meta -> report -> compare is really exercised, not just imported.
fake_kernel_csv() {   # out, marker(idx|hc), scale
    python3 - "$@" <<'PYEOF'
import sys
out, marker, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = [("combined_lora_idx_finalize", "AI_VECTOR_CORE", 13.0, 40),
        ("add_lora_z1_bfloat16_t_", "AI_VECTOR_CORE", 26.0, 40),
        ("add_lora_z2_bfloat16_t_", "AI_VECTOR_CORE", 14.0, 40)] if marker == "idx" else []
rows += [("HcPre", "COMMUNICATION", 3.0, 86),
         ("aclnnScatterNdUpdateV2", "AI_VECTOR_CORE", 18.0, 61),
         ("aclnnGroupedMatmulWeightNz", "AI_CORE", 44.0, 58)]
t = 0.0
with open(out, "w") as fh:
    fh.write("Step Id,Model ID,Task ID,Stream ID,Name,Type,Accelerator Core,"
             "Start Time(us),Duration(us),Wait Time(us),Block Dim\n")
    for step in range(20):
        for name, core, dur, per in rows:
            for _ in range(per):
                d = dur * scale
                fh.write(f"0,0,0,{3 if core=='COMMUNICATION' else 2},{name},"
                         f"t,{core},{t:.3f},{d:.3f},0,20\n")
                t += d + 0.5
print(f"  wrote {out}")
PYEOF
}

smoke_pipeline() {
    local T=${PROF_ROOT}/.smoke
    rm -rf "${T}"; mkdir -p "${T}"
    echo
    echo "== prune / claim, on synthetic capture dirs =="
    local P=${T}/prof CL=${T}/claims r g
    mkdir -p "${P}"; : > "${CL}"
    for g in 20261008100000000 20261008100500000; do
        for r in 0 1 2 3 4 5 6 7; do
            mkdir -p "${P}/dp0_pp0_tp${r}_dcp0_ep${r}_rank${r}_100${r}_${g}_ascend_pt/ASCEND_PROFILER_OUTPUT"
        done
        prune_to_rank0 "${P}" >/dev/null || ckf "prune_to_rank0 failed"
        local c; c=$(claim_rank0 "${P}" "${CL}") \
            && ck "window ${g}: pruned to rank0 and claimed ${c}" \
            || ckf "claim_rank0 failed for ${g}"
    done
    [ "$(find "${P}" -maxdepth 1 -type d -name '*_ascend_pt' | wc -l)" = 2 ] \
        && ck "2 windows kept, 14 non-rank0 dirs deleted" \
        || ckf "unexpected dir count after pruning"
    if claim_rank0 "${P}" /dev/null >/dev/null 2>&1; then
        ckf "claim_rank0 accepted an ambiguous claim"
    else ck "claim_rank0 rejects ambiguity"; fi
    mkdir -p "${T}/norank0/dp0_pp0_tp3_dcp0_ep3_rank3_1_2_ascend_pt"
    if prune_to_rank0 "${T}/norank0" >/dev/null 2>&1; then
        ckf "prune_to_rank0 acted with no rank0 dir present"
    else
        [ -d "${T}/norank0/dp0_pp0_tp3_dcp0_ep3_rank3_1_2_ascend_pt" ] \
            && ck "prune refuses, and deletes nothing, when no rank0 dir exists" \
            || ckf "prune refused but deleted anyway"
    fi
    # in its own dir: other smoke artifacts under $T are legitimately fresh
    local Q=${T}/quiet; mkdir -p "${Q}"; touch "${Q}/f"
    [ -n "$(find "${Q}" -mindepth 1 -newermt '-20 seconds' -print -quit 2>/dev/null)" ] \
        && ck "find -newermt sees a fresh file (dump-settle wait works)" \
        || ckf "find -newermt '-20 seconds' found nothing; the settle wait would prune mid-dump"
    touch -d '2026-01-01 00:00' "${Q}/f"
    [ -z "$(find "${Q}" -mindepth 1 -newermt '-20 seconds' -print -quit 2>/dev/null)" ] \
        && ck "find -newermt ignores an old tree (so the wait can finish)" \
        || ckf "stale mtimes still look fresh; the settle wait would never finish"
    rm -rf "${T}/prof" "${T}/norank0"

    echo
    echo "== relocating a capture dir next to the csvs (manual-review layout) =="
    local M=${T}/mv; mkdir -p "${M}/src/ASCEND_PROFILER_OUTPUT" "${M}/arm"
    echo '{}' > "${M}/src/ASCEND_PROFILER_OUTPUT/trace_view.json"
    if mv "${M}/src" "${M}/arm/ascend_pt_rank0" 2>/dev/null \
       && [ -f "${M}/arm/ascend_pt_rank0/ASCEND_PROFILER_OUTPUT/trace_view.json" ]; then
        ck "capture dir moves into the arm dir, trace_view.json intact"
    else
        ckf "could not relocate a capture dir into the arm dir"
    fi

    echo
    echo "== meta -> report -> compare, on synthetic captures =="
    local R=${T}/prof_matrix spec tag branch lora overlap arms arm d
    while read -r spec; do
        IFS=: read -r tag branch lora overlap <<<"${spec}"
        [ "${lora}" = "1" ] && arms="${ARMS_LORA}" || arms="ds"
        for arm in ${arms}; do
            d=${R}/${tag}/decode/${arm}; mkdir -p "${d}"
            if [ "${arm}" = "ds" ] || [ "${lora}" != "1" ]; then
                fake_kernel_csv "${d}/kernel_details_rank0.csv" hc 1.0 >/dev/null
            else
                fake_kernel_csv "${d}/kernel_details_rank0.csv" idx \
                    "$([ "${overlap}" = "1" ] && echo 0.9 || echo 1.0)" >/dev/null
            fi
            echo "4.200 30" > "${d}/acc.txt"
            echo "smoke_capture_dir" > "${d}/capture_dir.txt"
            printf 'TOTAL delivered=1000 tok requests=8 wall=10.0s mean_tpot=16.00 ms/tok ttft_mean=1.00s ttft_p50=1.00s ttft_p90=1.00s ttft_max=1.00s ttft_min=1.00s decode_tpot=16.00 ms/tok\n' \
                > "${d}/load.log"
        done
        write_meta_json "${R}/${tag}" decode "${arms}" 1024 8 \
            "${R}/${tag}/decode/meta.json" "${tag}" "${branch}" "${lora}" \
            "${overlap}" "${PARALLEL}" "" "smoke" "" >/dev/null \
            && ck "meta.json for ${tag}" || ckf "meta.json for ${tag}"
        if [ -f "${SCRIPT_DIR}/lora_prof_analyze_2.py" ]; then
            python3 "${SCRIPT_DIR}/lora_prof_analyze_2.py" \
                --meta "${R}/${tag}/decode/meta.json" \
                --out "${R}/${tag}/decode/report.json" > "${R}/${tag}/decode/report.txt" 2>&1 \
                && ck "lora_prof_analyze_2 ran for ${tag}" \
                || { ckf "lora_prof_analyze_2 failed for ${tag}"; tail -4 "${R}/${tag}/decode/report.txt"; }
        fi
    done < <(selected_cells)
    if python3 "${SCRIPT_DIR}/n37_prof_compare.py" --root "${R}" \
         --out "${T}/compare.json" > "${T}/compare.txt" 2>&1; then
        ck "n37_prof_compare ran ($(wc -l < "${T}/compare.txt") lines)"
        grep -q "LoRA tax per cell" "${T}/compare.txt" && ck "tax table rendered" || ckf "tax table missing"
        grep -q "us/step per kernel family" "${T}/compare.txt" && ck "family matrix rendered" || ckf "family matrix missing"
        grep -q "named A/Bs" "${T}/compare.txt" && ck "named A/B section rendered" || ckf "named A/B section missing"
        echo
        echo "---- synthetic compare output (shape only, numbers are fake) ----"
        sed -n '/per cell/,/^$/p' "${T}/compare.txt"
        sed -n '/named A\/Bs/,$p' "${T}/compare.txt" | head -20
    else
        ckf "n37_prof_compare failed"; tail -20 "${T}/compare.txt"
    fi
    echo "  artifacts: ${T}"
}

# ---- per-cell --------------------------------------------------------------
run_cell() {
    local tag=$1 branch=$2 lora=$3 overlap=$4
    local cell=${OUT_ROOT}/${tag}
    local prof=${PROF_ROOT}/${tag}
    local claims=${cell}/.claimed
    local serve_log=${cell}/serve.log

    if [ "${SKIP_DONE}" = "1" ] && [ -f "${cell}/.done" ]; then
        echo; echo "===== cell ${tag}: already done, skipping ====="; return 0
    fi
    echo; echo "===== cell ${tag}  branch=${branch} LORA=${lora} OVERLAP=${overlap} $(date -Is) ====="
    rm -rf "${cell}"; mkdir -p "${cell}"; : > "${claims}"
    local loader=${LOADER}
    if [ -z "${loader}" ]; then loader=${cell}/load.py; write_loader "${loader}"; fi
    stop_serve

    # -- build -------------------------------------------------------------
    if [ "${DO_BUILD}" = "1" ]; then
        checkout_branch "${branch}" || { echo "!! checkout ${branch} failed"; return 1; }
        echo "--- clean rebuild (${branch}) ---"
        bash "${REBUILD}" > "${cell}/build.log" 2>&1
        local rc=$?
        tail -4 "${cell}/build.log"
        if [ ${rc} != 0 ] || ! grep -q "KERNEL FRESHNESS: OK" "${cell}/build.log"; then
            echo "!! BUILD FAILED or kernels stale (see ${cell}/build.log)"; return 1
        fi
    else
        echo "--- DO_BUILD=0: using the tree as it stands ---"
    fi
    { git -C "${VA}" log --oneline -1 | cat
      echo "HEAD $(git -C "${VA}" rev-parse --short=10 HEAD) wanted ${branch} dirty=$(git -C "${VA}" status --porcelain | wc -l)"
    } > "${cell}/git_info.txt"
    cat "${cell}/git_info.txt"

    # -- optional op-level numerics while the NPUs are free ----------------
    if [ "${RUN_UT}" = "1" ] && [ -f "${VA}/tests/ut/lora/test_add_lora_fused_moe.py" ]; then
        echo "--- UT: tests/ut/lora/test_add_lora_fused_moe.py ---"
        ( cd "${VA}" && ASCEND_RT_VISIBLE_DEVICES=0 timeout 1800 python -m pytest -x -q -s \
            tests/ut/lora/test_add_lora_fused_moe.py ) > "${cell}/ut.log" 2>&1
        tail -6 "${cell}/ut.log"
    fi

    # -- serve -------------------------------------------------------------
    rm -rf "${prof}"; mkdir -p "${prof}"
    LORA=${lora} OVERLAP=$([ "${overlap}" = "-" ] && echo 0 || echo "${overlap}") \
      PROF="${prof}" PORT="${PORT}" MODEL="${MODEL}" \
      LORA_ZERO="${LORA_ZERO}" LORA_RAND="${LORA_RAND}" \
      nohup bash "${SCRIPT_DIR}/n35_serve_matrix.sh" > "${serve_log}" 2>&1 &
    echo "   serve launched (log ${serve_log})"
    local i ready=0
    for i in $(seq "${SERVE_READY_TRIES}"); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/v1/models" 2>/dev/null)" = "200" ] \
            && { ready=1; break; }
        if grep -qE "Traceback|Engine core initialization failed" "${serve_log}" 2>/dev/null; then
            echo "!! serve FAILED to start"; grep -nE "Error|assert" "${serve_log}" | tail -8
            return 1
        fi
        sleep 10
    done
    [ "${ready}" = "1" ] || { echo "!! serve not ready after $((SERVE_READY_TRIES*10))s"; return 1; }
    echo "   serve ready after ~$((i*10))s"

    local spid; spid=$(pgrep -f "vllm serve" | head -1)
    if [ -n "${spid}" ]; then
        tr '\0' '\n' < /proc/"${spid}"/cmdline > "${cell}/serve_cmdline.txt"
        tr '\0' '\n' < /proc/"${spid}"/environ \
          | grep -E "^(VLLM_|HCCL_|OMP_|TASK_QUEUE|PYTORCH_NPU|ASCEND_RT|SOC_VERSION)" \
          > "${cell}/serve_env.txt"
        grep -E "VLLM_ASCEND_LORA_MOE_OVERLAP" "${cell}/serve_env.txt" || true
    fi

    if [ "${lora}" = "1" ]; then
        echo "--- delta probe (lora-rand must differ from ds, lora-adapter1 must not) ---"
        delta_probe 2>&1 | tee "${cell}/delta_probe.txt"
    fi

    # -- capture windows ---------------------------------------------------
    local arms; [ "${lora}" = "1" ] && arms="${ARMS_LORA}" || arms="ds"
    [ -f "${DATASET}" ] || { echo "!! DATASET missing: ${DATASET}"; return 1; }
    local nreq ds
    nreq=$(grep -c . "${DATASET}")
    if [ "${GLOBAL_WARMUP}" = "1" ] && [ "$(echo ${arms} | wc -w)" -gt 1 ]; then
        echo "--- global warmup: 16-tok burst on every arm, nothing captured ---"
        local w
        for w in ${arms}; do
            echo "   warming ${w}"
            python3 "${loader}" "${PORT}" "${w}" "${PARALLEL}" 16 dataset \
                "${DATASET}" 1 >/dev/null 2>&1
        done
    fi
    local phase pre cap maxtok arm adir
    for phase in ${PHASES}; do
        read -r pre cap maxtok <<<"$(phase_params "${phase}")"
        ds=${DATASET}
        nreq=$(grep -c . "${DATASET}")
        if [ "${phase}" = "prefill" ] && [ "${PREFILL_VARIANTS}" -gt 1 ]; then
            # The dataset holds one prompt per concurrent slot, so a prefill
            # window would catch a single wave and then only decode. Repeat it
            # with a distinct leading sentence per copy: each copy is then a
            # genuine prefill (prefix caching keys on the PREFIX, so a differing
            # first token invalidates the whole prompt) at a negligible token
            # cost, and nothing else about the load changes.
            ds=${cell}/prefill_dataset.jsonl
            python3 - "${DATASET}" "${ds}" "${PREFILL_VARIANTS}" <<'PYEOF'
import json, sys
src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(src) if l.strip()]
with open(dst, "w") as fh:
    for v in range(k):
        for r in rows:
            r = json.loads(json.dumps(r))
            m = r["messages"][0]
            m["content"] = f"Reading pass {v + 1}. " + m["content"]
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"   prefill dataset: {len(rows)} x {k} = {len(rows) * k} distinct prompts")
PYEOF
            nreq=$(grep -c . "${ds}")
        fi
        for arm in ${arms}; do
            adir=${cell}/${phase}/${arm}; mkdir -p "${adir}"
            echo; echo "--- ${tag} ${phase} ${arm}: warmup ---"
            python3 "${loader}" "${PORT}" "${arm}" "${PARALLEL}" 128 dataset \
                "${ds}" 1 > "${adir}/warmup.log" 2>&1
            grep TOTAL "${adir}/warmup.log" || tail -2 "${adir}/warmup.log"
            # the serve runs with --enable-prefix-caching and the warmup just
            # cached these very prompts; without this the measured window would
            # replay cache hits instead of doing the prefill
            curl -sf -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null \
                || echo "   WARNING: reset_prefix_cache failed"

            local n0; n0=$(wc -l < "${serve_log}" 2>/dev/null || echo 1)
            echo "--- ${tag} ${phase} ${arm}: load ${nreq} reqs x<=${maxtok} tok (background) ---"
            setsid nohup python3 "${loader}" "${PORT}" "${arm}" "${PARALLEL}" "${maxtok}" \
                dataset "${ds}" 1 > "${adir}/load.log" 2>&1 < /dev/null &
            sleep "${pre}"
            echo "--- profile window ${cap}s ---"
            curl -sf -X POST "http://localhost:${PORT}/start_profile" \
                || { echo "!! start_profile failed (serve needs profiler-config at launch)"; return 1; }
            sleep "${cap}"
            curl -sf -X POST "http://localhost:${PORT}/stop_profile" \
                || echo "   WARNING: stop_profile failed"

            echo "--- draining load ---"
            for i in $(seq "${DRAIN_TRIES}"); do
                local n; n=$(grep -c " tok " "${adir}/load.log" 2>/dev/null); [ -z "$n" ] && n=0
                [ "$n" -ge "${nreq}" ] && break
                sleep 10
            done
            grep TOTAL "${adir}/load.log" || tail -2 "${adir}/load.log"
            local n1; n1=$(wc -l < "${serve_log}" 2>/dev/null || echo 1)
            acc_between "${serve_log}" "${n0}" "${n1}" > "${adir}/acc.txt"
            echo "   acceptance: $(cat "${adir}/acc.txt")"

            wait_dump_quiet "${prof}"
            prune_to_rank0 "${prof}" || return 1
            local claimed; claimed=$(claim_rank0 "${prof}" "${claims}") || return 1
            echo "${prof}/${claimed}" > "${adir}/capture_dir.txt"
            echo "   rank0 capture: ${claimed}"
            curl -sf -X POST "http://localhost:${PORT}/reset_prefix_cache" >/dev/null \
                || echo "   WARNING: reset_prefix_cache failed"
        done
    done

    stop_serve

    # -- finalize: analyse() now sees ONLY rank0 dirs ----------------------
    echo "--- torch_npu analyse() on the rank0-only tree ---"
    du -sh "${prof}"
    python3 -c "from torch_npu.profiler.profiler import analyse; analyse('${prof}')" \
        > "${cell}/analyse.log" 2>&1 \
        || { echo "!! analyse() failed, see ${cell}/analyse.log"; return 1; }
    tail -2 "${cell}/analyse.log"

    local f capdir src
    for phase in ${PHASES}; do
        for arm in ${arms}; do
            adir=${cell}/${phase}/${arm}
            capdir=$(cat "${adir}/capture_dir.txt")
            src=${capdir}/ASCEND_PROFILER_OUTPUT
            # flat copies of the small csvs: that is what the analysis reads, and
            # it survives whatever KEEP does to the capture dir
            for f in kernel_details.csv op_statistic.csv step_trace_time.csv api_statistic.csv; do
                [ -f "${src}/${f}" ] && cp -p "${src}/${f}" "${adir}/${f%.csv}_rank0.csv"
            done
            [ -f "${adir}/kernel_details_rank0.csv" ] \
                || echo "   !! ${phase}/${arm}: no kernel_details.csv produced"

            # shrink to the requested level BEFORE moving (less to move)
            case "${KEEP}" in
              all) ;;
              traces)
                find "${capdir}" -maxdepth 1 -type d \
                     \( -name 'PROF_*' -o -name 'FRAMEWORK' \) -exec rm -rf {} + 2>/dev/null ;;
              csv)
                find "${capdir}" -maxdepth 1 -type d \
                     \( -name 'PROF_*' -o -name 'FRAMEWORK' \) -exec rm -rf {} + 2>/dev/null
                find "${capdir}" -type f \( -name 'trace_view.json' -o -name '*.db' \
                     -o -name 'communication.json' -o -name 'operator_details.csv' \) \
                     -delete 2>/dev/null ;;
              *) echo "   !! unknown KEEP='${KEEP}' (all|traces|csv); keeping everything" ;;
            esac

            # one cell = one self-contained tree, so a trace can be reviewed or
            # copied off the box without hunting through PROF_ROOT
            if mv "${capdir}" "${adir}/ascend_pt_rank0" 2>/dev/null; then
                echo "   trace -> ${adir}/ascend_pt_rank0"
            else
                # cross-device (e.g. PROF_ROOT on tmpfs): copy, then drop the original
                cp -a "${capdir}" "${adir}/ascend_pt_rank0" \
                    && rm -rf "${capdir}" \
                    && echo "   trace copied -> ${adir}/ascend_pt_rank0" \
                    || echo "   !! could not relocate ${capdir}; it stays in place"
            fi
            local tv=${adir}/ascend_pt_rank0/ASCEND_PROFILER_OUTPUT/trace_view.json
            printf '%s\t%s\t%s\t%s\n' "${tag}" "${phase}" "${arm}" \
                "$([ -f "${tv}" ] && echo "${tv}" || echo "(no trace_view.json, KEEP=${KEEP})")" \
                >> "${OUT_ROOT}/traces.tsv"
        done
    done
    echo "--- traces kept for manual review (rank0 only) ---"
    du -sh "${cell}" 2>/dev/null
    grep -P "^${tag}\t" "${OUT_ROOT}/traces.tsv" 2>/dev/null | sed 's/^/   /'

    for phase in ${PHASES}; do
        read -r pre cap maxtok <<<"$(phase_params "${phase}")"
        write_meta_json "${cell}" "${phase}" "${arms}" "${maxtok}" "${cap}" \
            "${cell}/${phase}/meta.json" "${tag}" "${branch}" "${lora}" \
            "${overlap}" "${PARALLEL}" "${DATASET}" \
            "$(cat "${cell}/git_info.txt" | tr '\n' ' ')" "${serve_log}" || return 1
        if [ -f "${SCRIPT_DIR}/lora_prof_analyze_2.py" ]; then
            echo "--- per-cell report ${tag}/${phase} ---"
            python3 "${SCRIPT_DIR}/lora_prof_analyze_2.py" --meta "${cell}/${phase}/meta.json" \
                --out "${cell}/${phase}/report.json" | tee "${cell}/${phase}/report.txt" | tail -30
        fi
    done

    rmdir "${prof}" 2>/dev/null || true   # empty now that the traces moved

    date -Is > "${cell}/.done"
    echo "===== cell ${tag} done $(date -Is) ====="
}

# N36_LIB_ONLY=1 stops here with every helper defined (for external tests)
if [ "${N36_LIB_ONLY:-0}" = "1" ]; then return 0 2>/dev/null || exit 0; fi

# ---- drive -----------------------------------------------------------------
CELLS=${CELLS:-}
mkdir -p "${OUT_ROOT}"
RUN_LOG=${OUT_ROOT}/$([ "${SMOKE}" = "1" ] && echo smoke || echo matrix)_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "${RUN_LOG}") 2>&1

if [ "${SMOKE}" = "1" ]; then
    echo "=== SMOKE: presence + pipeline, no serve, no NPU, no build  $(date -Is) ==="
    echo "VA=${VA} MODEL=${MODEL} OUT_ROOT=${OUT_ROOT} PROF_ROOT=${PROF_ROOT}"
    check_env
    smoke_pipeline
    echo
    echo "=== SMOKE RESULT: ${CHK_FAIL} failure(s), ${CHK_WARN} warning(s) ==="
    [ "${CHK_FAIL}" = 0 ] \
        && echo "everything the run needs is present. Start it with: bash $(basename "$0")" \
        || echo "fix the FAIL lines above before starting the run"
    echo "log: ${RUN_LOG}"
    exit "$([ "${CHK_FAIL}" = 0 ] && echo 0 || echo 1)"
fi

if ! DSMSG=$(dataset_sanity "${DATASET}") && [ "${FORCE_DATASET:-0}" != "1" ]; then
    echo "!! refusing to start: ${DATASET} looks padded or malformed"
    echo "   ${DSMSG}"
    echo "   repeated filler inflates MTP acceptance, which TPOT is divided by."
    echo "   Use the kit's data/dataset.jsonl, or set FORCE_DATASET=1."
    exit 1
fi
echo "=== prof matrix $(date -Is) ==="
echo "load: ${DATASET} ($(grep -c . "${DATASET}") prompts; ${DSMSG})"
echo "phases=${PHASES} parallel=${PARALLEL} out=${OUT_ROOT} prof=${PROF_ROOT} keep=${KEEP}"
FAILED=""
while read -r spec; do
    IFS=: read -r tag branch lora overlap <<<"${spec}"
    run_cell "${tag}" "${branch}" "${lora}" "${overlap}" || FAILED="${FAILED} ${tag}"
done < <(selected_cells)

echo; echo "=== matrix finished $(date -Is) ==="
[ -n "${FAILED}" ] && echo "FAILED cells:${FAILED}"
echo "--- cross-cell comparison ---"
python3 "${SCRIPT_DIR}/n37_prof_compare.py" --root "${OUT_ROOT}" \
    --out "${OUT_ROOT}/compare.json" | tee "${OUT_ROOT}/compare.txt" | tail -40
echo "run log: ${RUN_LOG}"
echo "full comparison: ${OUT_ROOT}/compare.txt"
