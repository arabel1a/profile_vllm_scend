# prof_matrix_kit -- profiler traces for the LoRA branches, rank0 only

Copy this folder into the container, run one script, come back in a few hours to
per-branch profiler traces plus a cross-branch comparison table.

    docker cp prof_matrix_kit <container>:/vllm-workspace/prof_kit
    docker exec -it <container> bash -lc \
        'cd /vllm-workspace/prof_kit && bash n36_prof_matrix.sh --smoke'
    docker exec -d  <container> bash -lc \
        'cd /vllm-workspace/prof_kit && bash n36_prof_matrix.sh'

`--smoke` first: minutes, no NPU, no server. It checks that everything the run
needs is present on THIS machine and drives the whole post-capture pipeline on
synthetic captures. Fix whatever it prints as FAIL, then start the real run.

344 KB. Nothing is downloaded at run time.

## What it needs from the machine
- the vllm-ascend git checkout, default `/vllm-workspace/vllm-ascend`
  (`VA=/path`), with the cell branches reachable. A branch is resolved as
  `<name>`, then `mmo/<name>`, `mmo/misha_<name>`, `origin/<name>`,
  `tianchen/<name>`, so a fresh clone where they exist only as
  `mmo/misha_night_z1z2` works; the local branch is created from the remote.
- `vllm`, `python3` with `torch_npu`, `git`, `curl`, 8 NPUs.
- the base model and two LoRA adapters:
  `MODEL=/data/models/DeepSeek-V4-Flash-0731-w8a8`
  `LORA_ZERO=/data/models/DeepSeek-V4-Flash-0731-lora-zero-rank16`
  `LORA_RAND=/data/models/DeepSeek-V4-Flash-0731-lora-rand16`
  `--smoke` prints the exact `gen_zero_lora_ds.py` / `gen_rand_lora_ds.py`
  command when one is missing. Both are needed: the zero adapter is the perf
  arm (it adds no numerical effect, so the cost it adds is pure overhead), the
  rand adapter is the falsifiability check -- with zero weights alone a DROPPED
  LoRA delta is invisible and reads as a speedup.
- ~40 GB free where the traces land, and ~30 GB free under `PROF_ROOT` while a
  window is live (all 8 ranks exist until the prune).

Everything else is in this folder. No dataset download, no `load.py`, no
`n22_delta_probe.sh`, no network.

## Files
    n36_prof_matrix.sh     the driver -- the only thing you run
    n35_serve_matrix.sh    the serve, LoRA on/off and overlap on/off
    n37_prof_compare.py    cross-cell comparison (self-contained)
    clean_rebuild_va.sh    clean rebuild incl. the AscendC kernels, with the
                           kernel-freshness guard. If your container already has
                           one you trust, use it: REBUILD=/path/to/yours
    lora_prof_analyze_2.py optional, per-cell report; delete it and the run still
                           works, you just lose the per-arm tables
    data/dataset.jsonl     the load: 8 LongAlpaca-12k prompts, 24.6k-36.2k chars
                           (~6.2k-9.0k tokens), one per concurrent slot

## The load, and why it is this load
8 real LongAlpaca-12k instructions (paper + question), natural long-context
prose. Required -- there is deliberately no filler fallback, and this is not the
same load as the earlier `fast_*` scoreboards used.

Filler text, a sentence repeated to length, is trivially predictable, so the
dspark MTP draft accepts nearly every token of it. Acceptance is what TPOT is
divided by, so a filler load reports an acceptance and a per-step cost that no
real traffic produces. The `data/dataset.jsonl` shipped with the earlier rounds
was a Python-completion prompt padded with the sentence
"这是一个填充句子，用于扩展token长度，请忽略此内容。" repeated to length, which has
exactly that problem. Numbers from this kit are therefore NOT comparable with
those earlier scoreboards; they are comparable with each other, which is what
the matrix is for.

Pass `DATASET=/path/to/other.jsonl` to use something else. Format: one JSON
object per line, `{"messages": [{"role": "user", "content": "..."}]}`.

## Nothing is captured cold
Per cell: a 16-token burst on every arm first (`GLOBAL_WARMUP=1`, so the arm
measured first is not the one that paid for first-touch weight and adapter
loading), then a 128-token warmup load per arm that is allowed to finish, then
`reset_prefix_cache` (the serve runs with `--enable-prefix-caching` and the
warmup just cached these very prompts), then the real load starts and runs
`PRE_WAIT` seconds -- 30 in decode, 3 in prefill -- before `start_profile`, so
the window opens on a steady batch.

## The 10 cells (4 clean rebuilds, grouped by branch)
    test_nolora       test           LORA=0            stock baseline
    test_lora         test           LORA=1            stock bgmv/sgmv LoRA path
    addlora_lora_ov0  addlora_path   LORA=1 OVERLAP=0
    addlora_lora_ov1  addlora_path   LORA=1 OVERLAP=1
    z1z2_lora_ov0     night_z1z2     LORA=1 OVERLAP=0
    z1z2_lora_ov1     night_z1z2     LORA=1 OVERLAP=1
    z1z2_nolora       night_z1z2     LORA=0   no-lora WITHOUT scatter
    scatter_lora_ov0  night_scatter  LORA=1 OVERLAP=0
    scatter_lora_ov1  night_scatter  LORA=1 OVERLAP=1
    scatter_nolora    night_scatter  LORA=0   no-lora WITH scatter

`night_scatter` is `night_z1z2` plus exactly the two ScatterNdUpdateV2 commits
and nothing else, so the two LORA=0 cells isolate the scatter kernel on the base
model path. `OVERLAP` is "-" on every LORA=0 cell because
`VLLM_ASCEND_LORA_MOE_OVERLAP` is only read on the LoRA path, and on `test`
because that branch has no such env and no `add_lora_fused` /
`combined_lora_idx` kernels at all. `addlora_path` LORA=0 is omitted: with LoRA
off it is the same code as `night_z1z2` LORA=0.

Each cell fails loudly rather than producing a number you cannot trust: the
build must print `KERNEL FRESHNESS: OK`, the serve must come up, and on LoRA
cells the delta probe must show `lora-rand` differing from `ds`.

## Rank0 only
One capture window writes ~3.2 GB per rank; TP=8 makes that ~26 GB, and a
previous run's profiler directory had reached 174 GB. `torch_npu`'s `analyse()`
re-parses every rank directory it finds, which is the bulk of the ~20 min
attribution cost. Nothing downstream needs the other seven: the analysis picks
ONE primary rank for every number it reports. So every `rank != 0` capture
directory is deleted after each window, BEFORE `analyse()`. A trace here is one
rank's timeline, not the collective.

## Output
    logs/prof_matrix/compare.txt     <- read this first
    logs/prof_matrix/compare.json
    logs/prof_matrix/traces.tsv      tag, phase, arm -> trace_view.json
    logs/prof_matrix/matrix_<ts>.log the whole run
    logs/prof_matrix/<tag>/
        build.log serve.log git_info.txt delta_probe.txt analyse.log
        <phase>/report.txt report.json meta.json
        <phase>/<arm>/kernel_details_rank0.csv       flat, what the analysis reads
        <phase>/<arm>/load.log warmup.log acc.txt
        <phase>/<arm>/ascend_pt_rank0/ASCEND_PROFILER_OUTPUT/trace_view.json
                                                             kernel_details.csv

`compare.txt` has: one row per cell/arm (steps, device busy per step, busiest and
second stream, wall per step, client TPOT, acceptance); the LoRA tax per cell
(lora arm minus base arm, split into lora/comm/other kernels, with the families
that moved); a per-family us/step matrix across all cells; and the named A/Bs --
overlap on vs off per branch, scatter on vs off, and each branch against
`addlora_path`.

Every trace is kept for manual review. `KEEP=traces` (default) keeps the full
analysed output including `trace_view.json`, ~2.3 GB per window, ~39 GB for the
default matrix. `KEEP=all` also keeps the unparsed `PROF_*`/`FRAMEWORK` so a
capture can be re-analysed with other options. `KEEP=csv` keeps only the csvs,
~150 MB per window, and leaves you no timeline.

## Time
~4-5 h for the default matrix: 4 clean rebuilds at ~20 min, then per cell ~8 min
of server startup, ~2 min per capture window, ~4 min of `analyse()`. The run
kills any `vllm serve` it finds and holds all 8 NPUs until it is done.

## Useful subsets
    CELLS="z1z2_nolora scatter_nolora" bash n36_prof_matrix.sh   # the scatter A/B
    PHASES="decode prefill" bash n36_prof_matrix.sh              # both windows
    DO_BUILD=0 CELLS=scatter_lora_ov1 bash n36_prof_matrix.sh    # already built
    RUN_UT=1 ...                        op-level numerics while the NPUs are free
    KEEP=all ...                        keep the unparsed capture too

Resumable: each finished cell writes a `.done` marker and is skipped on a rerun
(`SKIP_DONE=0` to force). Re-running the comparison alone, any time:

    python3 n37_prof_compare.py --root logs/prof_matrix

## Caveat
A profile window perturbs what it measures, so the per-step device numbers are
for attribution, not go/no-go. The client TPOT column is the sanity reference;
TPOT medians across server restarts are a separate job (`n23_measure_repeats.sh`
in the node1 scripts).
