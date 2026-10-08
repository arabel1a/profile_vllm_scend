#!/usr/bin/env python3
"""Decode-window analyzer for Ascend torch_npu kernel_details.csv.

Usage:
  python3 lora_prof_analyze_2.py --meta <meta.json> [--out report.json]
      meta.json is produced by lora_prof_capture_2.sh (two arms by default:
      lora-adapter1 and ds on the same serve). Prints per-arm tables and,
      when two arms are present, a per-kernel-family device-time delta table
      (lora minus base) with a classification verdict.
  python3 lora_prof_analyze_2.py file1.csv [file2.csv ...]
      standalone mode: each file is one arm; verdict falls back to hardcoded
      pre-fused-routing reference durations (approximate).

Per arm it reports:
  - top kernel families by device time (us/step, avg/p50/p90, core type)
  - step decomposition: wall ms/step, busiest-stream busy ms/step, second
    stream busy (aux-stream overlap evidence), inter-kernel gap percentiles
  - load metrics from the arm's load.log: delivered tokens, client TPOT,
    acceptance estimate
  - one summary line per rank csv (straggler detection)

Step counting: combined_lora_idx_finalize count / IDX_PER_STEP for lora arms;
HcPre count / HC_PER_STEP for base arms. Both divisors must be validated on a
B3 reference capture (the raw counts are printed so calibration is checkable).
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

TYPE_TAILS = re.compile(r"_(?:bfloat16|float16|fp16|fp32|i8|i16|i32|int32|i64|t)_")
COMM_RE = re.compile(
    r"HcPre|HcPost|AivKernel|Hccl|AllGather|AllReduce|AllSend|AllWrite"
    r"|Receive|Broadcast|ReduceScatter|Exchange")
IDX_FAM = "combined_lora_idx_finalize"
HC_FAM = "HcPre"
IDX_PER_STEP = 40
HC_PER_STEP = 86

BASELINE_AVG_US = {
    "bgmv_shrink": 71.5,
    "bgmv_expand": 29.7,
    "add_lora_z1": 26.5,
    "add_lora_z2": 13.9,
    "combined_lora_idx_finalize": 12.7,
    "aclnnSort": 4.2,
}
BASELINE_P50_US = {
    "bgmv_shrink": 70.0,
    "bgmv_expand": 29.0,
    "add_lora_z1": 25.0,
    "add_lora_z2": 13.5,
    "combined_lora_idx_finalize": 13.0,
    "aclnnSort": 2.8,
}
BASELINE_GAPS_P50_US = 0.5


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100.0))]


def family(name):
    f = TYPE_TAILS.split(name, 1)[0] or name
    if f.startswith("aclnn"):
        f = f.split("_", 1)[0]
    return f


def is_lora_fam(f):
    return ("add_lora" in f or "bgmv" in f or "combined_lora_idx" in f
            or f == "aclnnSort")


def classify(f, core, name):
    if core == "COMMUNICATION" or COMM_RE.search(name):
        return "comm"
    if is_lora_fam(f):
        return "lora"
    return "other"


def parse_csv(path):
    fams = {}
    streams = defaultdict(list)
    min_s = max_e = None
    busy = 0.0
    try:
        fh = open(path, newline="")
    except OSError as e:
        print(f"  !! cannot open {path}: {e}")
        return None
    with fh:
        rd = csv.reader(fh)
        hdr = next(rd, None)
        if not hdr:
            return None
        try:
            ix = {k: hdr.index(k) for k in
                  ("Name", "Accelerator Core", "Duration(us)",
                   "Start Time(us)", "Stream ID")}
        except ValueError as e:
            print(f"  !! unexpected csv header in {path}: {e}")
            return None
        for row in rd:
            try:
                dur = float(row[ix["Duration(us)"]])
                start = float(row[ix["Start Time(us)"]])
            except (ValueError, IndexError):
                continue
            name = row[ix["Name"]]
            core = row[ix["Accelerator Core"]]
            stream = row[ix["Stream ID"]]
            busy += dur
            streams[stream].append((start, dur))
            if min_s is None or start < min_s:
                min_s = start
            if max_e is None or start + dur > max_e:
                max_e = start + dur
            f = family(name)
            d = fams.get(f)
            if d is None:
                d = fams[f] = {"cnt": 0, "tot": 0.0, "durs": [],
                               "core": core, "name": name}
            d["cnt"] += 1
            d["tot"] += dur
            d["durs"].append(dur)
    return {"fams": fams, "streams": dict(streams),
            "min": min_s or 0.0, "max": max_e or 0.0, "busy": busy}


def steps_of(parsed):
    idx_cnt = parsed["fams"].get(IDX_FAM, {}).get("cnt", 0)
    hc_cnt = parsed["fams"].get(HC_FAM, {}).get("cnt", 0)
    if idx_cnt:
        return idx_cnt / IDX_PER_STEP, f"{IDX_FAM}={idx_cnt}/{IDX_PER_STEP}"
    if hc_cnt:
        return hc_cnt / HC_PER_STEP, f"{HC_FAM}={hc_cnt}/{HC_PER_STEP}"
    return 0.0, "no marker kernel"


def stream_rank(parsed):
    out = []
    for s, evs in parsed["streams"].items():
        evs.sort()
        b = sum(d for _, d in evs)
        gaps = [s1 - (s0 + d0) for (s0, d0), (s1, _) in zip(evs, evs[1:])]
        gaps = [g for g in gaps if 0 <= g < 1000000]
        out.append({"stream": s, "busy": b, "gaps": gaps})
    out.sort(key=lambda x: -x["busy"])
    return out


def load_metrics(path):
    m = {"delivered": None, "mean_tpot_ms": None, "wall_s": None,
         "requests": None, "ttft_mean_s": None, "ttft_p50_s": None,
         "ttft_p90_s": None, "ttft_max_s": None, "ttft_min_s": None,
         "decode_tpot_ms": None}
    if not path or not os.path.exists(path):
        return m
    for line in open(path, errors="replace"):
        mm = re.search(
            r"TOTAL delivered=(\d+) tok requests=(\d+) wall=([\d.]+)s "
            r"mean_tpot=([\d.]+) ms/tok ttft_mean=([\d.]+)s "
            r"ttft_p50=([\d.]+)s ttft_p90=([\d.]+)s ttft_max=([\d.]+)s "
            r"ttft_min=([\d.]+)s decode_tpot=([\d.]+) ms/tok", line)
        if mm:
            m.update(delivered=int(mm[1]), requests=int(mm[2]),
                     wall_s=float(mm[3]), mean_tpot_ms=float(mm[4]),
                     ttft_mean_s=float(mm[5]), ttft_p50_s=float(mm[6]),
                     ttft_p90_s=float(mm[7]), ttft_max_s=float(mm[8]),
                     ttft_min_s=float(mm[9]), decode_tpot_ms=float(mm[10]))
            return m
    for line in open(path, errors="replace"):
        mm = re.search(r"TOTAL delivered=(\d+) tok requests=(\d+) "
                       r"wall=([\d.]+)s mean_tpot=([\d.]+)", line)
        if mm:
            m.update(delivered=int(mm[1]), requests=int(mm[2]),
                     wall_s=float(mm[3]), mean_tpot_ms=float(mm[4]))
            return m
    tot = 0
    n = 0
    for line in open(path, errors="replace"):
        mm = re.match(r"req\d+: (\d+) tok in ([\d.]+)s", line)
        if mm:
            tot += int(mm[1])
            n += 1
    if n:
        m.update(delivered=tot, requests=n)
    return m


def analyze_arm(label, csv_paths, load_log, parallel, rank_hint=0,
                acc_tel=None, acc_lines=0):
    # Parse every rank csv once; the profiler randomly drops events on some
    # ranks per run, so the primary rank is chosen by highest marker count
    # (proxy for most complete recording), not by rank id.
    parsed = []
    for c in sorted(csv_paths, key=lambda c: c.get("rank", 99)):
        p = parse_csv(c["path"])
        if p:
            p["_csv"] = c
            parsed.append(p)
    if not parsed:
        return None
    per_rank = []
    for p in parsed:
        st, mk = steps_of(p)
        per_rank.append((st, mk, p))
    steps, marker, main = max(per_rank, key=lambda x: x[0])
    wall_ms = (main["max"] - main["min"]) / 1000.0
    wall_s = wall_ms / 1000.0
    prefill = steps < 5
    ms_step = wall_ms / steps if steps else 0.0
    sps = steps / wall_s if wall_s else 0.0
    stranks = stream_rank(main)
    busy_step_ms = main["busy"] / 1000.0 / steps if steps else 0.0
    lm = load_metrics(load_log)
    acc = None
    acc_warn = None
    client_ms_step = None
    if lm["delivered"] and lm["wall_s"] and steps and parallel:
        acc = (lm["delivered"] / lm["wall_s"]) / (sps * parallel) if sps else None
    if acc and acc > 6.5:
        acc_warn = f"acc={acc:.2f} is physically impossible " \
                   f"(max 6 at k=5): marker counts unreliable, ignoring"
        acc = None
    if lm["mean_tpot_ms"] and acc:
        client_ms_step = lm["mean_tpot_ms"] * acc
    fam_rows = []
    for f, d in main["fams"].items():
        fam_rows.append({
            "fam": f, "core": d["core"], "name": d["name"], "cnt": d["cnt"],
            "tot": d["tot"], "avg": d["tot"] / d["cnt"],
            "p50": pct(d["durs"], 50), "p90": pct(d["durs"], 90),
            "per_step": d["tot"] / steps if steps else 0.0,
            "cls": classify(f, d["core"], d["name"])})
    fam_rows.sort(key=lambda r: -r["tot"])
    sts = [st for st, _, _ in per_rank]
    hi = max(sts) if sts else 0
    lo = min(sts) if sts else 0
    spread_warn = hi > 0 and lo < hi / 2.0
    rank_lines = []
    for st, mk, p in per_rank:
        sr = stream_rank(p)
        top_share = (sr[0]["busy"] / p["busy"] * 100.0) if sr and p["busy"] else 0
        flag = "  <-- DROPPED EVENTS" if (hi > 0 and st < hi / 2.0) else ""
        rank_lines.append(
            f"  rank{p['_csv'].get('rank', '?')}: "
            f"wall={(p['max']-p['min'])/1e6:.1f}s "
            f"busy={p['busy']/1e6:.1f}s steps~{st:.0f}({mk.split('=')[0]}) "
            f"top-stream {top_share:.0f}%{flag}")
    return {"label": label, "csv": main["_csv"]["path"],
            "primary_rank": main["_csv"].get("rank"),
            "steps_min_rank": lo, "steps_max_rank": hi,
            "spread_warn": spread_warn,
            "wall_s": wall_s, "steps": steps, "marker": marker,
            "prefill_window": prefill,
            "ms_step": ms_step, "steps_per_s": sps,
            "busy_ms": main["busy"] / 1000.0, "busy_step_ms": busy_step_ms,
            "streams": stranks, "fams": fam_rows, "load": lm,
            "acc": acc if acc_tel is None else acc_tel,
            "acc_warn": acc_warn,
            "acc_is_telemetry": acc_tel is not None, "acc_lines": acc_lines,
            "client_ms_step": client_ms_step,
            "rank_lines": rank_lines}


def print_arm(a):
    print(f"\n===== arm {a['label']}  ({a['csv'].split('/')[-1]}) =====")
    print(f"primary rank {a.get('primary_rank')} selected by max marker "
          f"count; per-rank steps {a.get('steps_min_rank', 0):.0f}.."
          f"{a.get('steps_max_rank', 0):.0f}")
    if a.get("spread_warn"):
        print("-> WARNING: marker counts differ >2x across ranks: some "
              "ranks dropped events; per-rank lines flagged below")
    if a.get("acc_warn"):
        print(f"-> WARNING: {a['acc_warn']}")
    if a["prefill_window"]:
        print(f"PREFILL/MIXED window (decode steps~{a['steps']:.0f} only: "
              f"[{a['marker']}]); step-normalized metrics not meaningful")
        print(f"window: wall={a['wall_s']:.2f}s device busy sum="
              f"{a['busy_ms']/1000:.1f}s")
    else:
        print(f"window: wall={a['wall_s']:.2f}s steps~{a['steps']:.0f} "
              f"[{a['marker']}] -> {a['ms_step']:.2f} ms/step, "
              f"{a['steps_per_s']:.1f} steps/s")
        print(f"device busy sum: {a['busy_ms']/1000:.1f}s = "
              f"{a['busy_step_ms']:.2f} ms/step")
    for i, s in enumerate(a["streams"][:2]):
        share = s["busy"] / (a["busy_ms"] * 1000.0) * 100.0 if a["busy_ms"] else 0
        ps = s["busy"] / 1000.0 / a["steps"] if a["steps"] else \
            s["busy"] / 1000.0
        print(f"stream#{i} id={s['stream']}: "
              f"{ps:.2f} {'ms/step' if a['steps'] else 's/window'} "
              f"({share:.0f}% of busy)")
    if a["streams"]:
        g = a["streams"][0]["gaps"]
        print(f"gaps on busiest stream: p50={pct(g, 50):.2f}us "
              f"p90={pct(g, 90):.2f}us  >1ms:{sum(1 for x in g if x > 1000)} "
              f">10ms:{sum(1 for x in g if x > 10000)}")
    lm = a["load"]
    ld = (f"delivered={lm['delivered']}" if lm["delivered"] else "delivered=?")
    tp = (f"client_tpot={lm['mean_tpot_ms']:.2f} ms/tok"
          if lm["mean_tpot_ms"] else "client_tpot=?")
    dp = (f"decode_tpot(excl_ttft)={lm['decode_tpot_ms']:.2f} ms/tok"
          if lm["decode_tpot_ms"] else "")
    tf = (f"ttft p50/p90/max={lm['ttft_p50_s']:.1f}/{lm['ttft_p90_s']:.1f}/"
          f"{lm['ttft_max_s']:.1f}s (min {lm['ttft_min_s']:.1f}s)"
          if lm["ttft_p50_s"] else "")
    if a["acc"] is not None:
        ac = (f"acc={a['acc']:.2f} (telemetry, {a['acc_lines']} lines)"
              if a["acc_is_telemetry"] else f"acc~{a['acc']:.2f} (est)")
    else:
        ac = "acc=?"
    print(f"load: {ld} {tp} {dp} {tf} {ac}")
    for line in a["rank_lines"]:
        print(line)
    print(f"top families by device time (% of busy in parens):")
    if a["prefill_window"]:
        print(f"  {'core':<16}{'family':<34}{'cnt':>7}{'tot_ms':>9}"
              f"{'avg':>8}{'p50':>8}{'p90':>8}")
        for r in a["fams"][:20]:
            print(f"  {r['core'][:16]:<16}{r['fam'][:34]:<34}{r['cnt']:>7}"
                  f"{r['tot']/1000:>9.1f}{r['avg']:>8.1f}{r['p50']:>8.1f}"
                  f"{r['p90']:>8.1f}  ({100*r['tot']/a['busy_ms']/1000:.1f}%)")
    else:
        print(f"  {'core':<16}{'family':<34}{'cnt':>7}{'us/step':>9}"
              f"{'avg':>8}{'p50':>8}{'p90':>8}")
        for r in a["fams"][:20]:
            print(f"  {r['core'][:16]:<16}{r['fam'][:34]:<34}{r['cnt']:>7}"
                  f"{r['per_step']:>9.1f}{r['avg']:>8.1f}{r['p50']:>8.1f}"
                  f"{r['p90']:>8.1f}  ({100*r['tot']/a['busy_ms']/1000:.1f}%)")


def pair_diff(base, lora):
    if base["prefill_window"] or lora["prefill_window"]:
        print(f"\n===== device-time delta ({lora['label']} minus {base['label']}),"
              f" us per WINDOW (prefill window: not step-normalized) =====")
        fb = {r["fam"]: r for r in base["fams"]}
        fl = {r["fam"]: r for r in lora["fams"]}
        rows = []
        for f in set(fb) | set(fl):
            rb, rl = fb.get(f), fl.get(f)
            rows.append({"fam": f, "cls": (rl or rb)["cls"],
                         "core": (rl or rb)["core"],
                         "per_b": rb["tot"] if rb else 0.0,
                         "per_l": rl["tot"] if rl else 0.0,
                         "delta": (rl["tot"] if rl else 0.0) -
                                  (rb["tot"] if rb else 0.0)})
        rows.sort(key=lambda r: -abs(r["delta"]))
        cls_sum = defaultdict(float)
        for r in rows:
            cls_sum[r["cls"]] += r["delta"]
        print(f"  {'core':<16}{'family':<34}{'base_us':>10}{'lora_us':>10}"
              f"{'delta_us':>10}  class")
        for r in rows[:15]:
            print(f"  {r['core'][:16]:<16}{r['fam'][:34]:<34}{r['per_b']:>10.0f}"
                  f"{r['per_l']:>10.0f}{r['delta']:>+10.0f}  {r['cls']}")
        print(f"class sums (us/window): comm={cls_sum['comm']:+.0f} "
              f"lora={cls_sum['lora']:+.0f} other={cls_sum['other']:+.0f}")
        return {"rows": rows[:30], "cls_sum_us": dict(cls_sum)}
    step_b = base["steps"] or 1
    step_l = lora["steps"] or 1
    fb = {r["fam"]: r for r in base["fams"]}
    fl = {r["fam"]: r for r in lora["fams"]}
    rows = []
    for f in set(fb) | set(fl):
        rb, rl = fb.get(f), fl.get(f)
        per_b = rb["tot"] / step_b if rb else 0.0
        per_l = rl["tot"] / step_l if rl else 0.0
        cls = (rl or rb)["cls"]
        rows.append({"fam": f, "cls": cls,
                     "core": (rl or rb)["core"],
                     "per_b": per_b, "per_l": per_l,
                     "delta": per_l - per_b})
    rows.sort(key=lambda r: -abs(r["delta"]))
    cls_sum = defaultdict(float)
    for r in rows:
        cls_sum[r["cls"]] += r["delta"]
    print(f"\n===== device-time delta ({lora['label']} minus {base['label']}),"
          f" us/step =====")
    print(f"  {'core':<16}{'family':<34}{'base':>9}{'lora':>9}{'delta':>9}"
          f"  class")
    for r in rows[:15]:
        print(f"  {r['core'][:16]:<16}{r['fam'][:34]:<34}{r['per_b']:>9.1f}"
              f"{r['per_l']:>9.1f}{r['delta']:>+9.1f}  {r['cls']}")
    dev_tax = sum(cls_sum.values()) / 1000.0
    print(f"class sums (us/step): comm={cls_sum['comm']:+.0f} "
          f"lora={cls_sum['lora']:+.0f} other={cls_sum['other']:+.0f} "
          f"total={cls_sum['comm']+cls_sum['lora']+cls_sum['other']:+.0f}")
    print(f"base {base['ms_step']:.2f} ms/step vs lora {lora['ms_step']:.2f} "
          f"ms/step wall; device tax {dev_tax:+.2f} ms/step")
    tb = base["load"]["mean_tpot_ms"]
    tl = lora["load"]["mean_tpot_ms"]
    if tb and tl:
        acc = lora["acc"] or 1.0
        print(f"client tpot {tb:.2f} -> {tl:.2f} ms/tok; client tax "
              f"{(tl - tb) * acc:+.2f} ms/step (acc~{acc:.2f})")
    sr = (lora["steps"] / base["steps"]) if base["steps"] else 0
    if sr > 1.10:
        print(f"-> WARNING: steps grew {100 * (sr - 1):.0f}% at equal load: "
              f"acceptance/scheduling drop, not per-kernel cost")
    tot = cls_sum["comm"] + cls_sum["lora"] + cls_sum["other"]
    if tot > 0:
        if cls_sum["comm"] > 0.5 * tot:
            print("-> comm-dominated lora tax: inspect lora-only collectives "
                  "(idx allgather, delta fold sync) and HCCL config/topology")
        elif cls_sum["lora"] > 0.5 * tot:
            print("-> lora-kernel-dominated tax: compare per-kernel avgs "
                  "above vs the B3 report (codegen/core-count sensitivity)")
        else:
            print("-> tax spread across non-lora kernels: compare top "
                  "families vs the B3 report")
    if tb and tl and lora["acc"] and dev_tax < 0.6 * (tl - tb) * lora["acc"]:
        print("-> device tax explains <60% of client tax: host/launch/sync "
              "overhead likely; check gap percentiles and stream count")
    return {"rows": rows[:30], "cls_sum_us": dict(cls_sum),
            "device_tax_ms_step": dev_tax}


def fallback_verdict(a):
    ratios = []
    for r in a["fams"]:
        b = BASELINE_AVG_US.get(r["fam"])
        if b:
            ratios.append((r["fam"], r["avg"] / b, r["p50"] / BASELINE_P50_US[r["fam"]]))
    if not ratios:
        return
    med = sorted(r[1] for r in ratios)[len(ratios) // 2]
    g = a["streams"][0]["gaps"] if a["streams"] else []
    gp = pct(g, 50)
    print(f"\n-- single-arm fallback vs pre-fused-routing reference --")
    for fam, ra, rp in ratios:
        print(f"   {fam:<34} avg x{ra:.2f} p50 x{rp:.2f}")
    print(f"median avg ratio {med:.2f}; busiest-stream gap p50 {gp:.2f}us")
    if med > 2.0:
        print("-> lora kernels ~Nx slower than reference: vector-execution / "
              "SOC build issue")
    elif gp > 3.0 * BASELINE_GAPS_P50_US:
        print("-> kernel durations near reference but gaps inflated: "
              "launch/host overhead")
    else:
        print("-> durations and gaps near reference: compare steps/s and "
              "top families against the other node's report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta")
    ap.add_argument("--out")
    ap.add_argument("csvs", nargs="*")
    args = ap.parse_args()
    arms = []
    if args.meta:
        meta = json.load(open(args.meta))
        for e in meta["arms"]:
            a = analyze_arm(e["arm"], e["csvs"], e.get("load_log"),
                            meta.get("parallel", 8),
                            acc_tel=e.get("acc_telemetry"),
                            acc_lines=e.get("acc_lines", 0))
            if a:
                a["_meta"] = e
                arms.append(a)
        print(f"meta: parallel={meta.get('parallel')} "
              f"max_tokens={meta.get('max_tokens')} "
              f"mode={meta.get('load_mode')} dataset={meta.get('dataset')}")
        if meta.get("node", {}).get("git"):
            print(f"node: {meta['node']['git']}")
    else:
        for p in args.csvs:
            a = analyze_arm(os.path.basename(p), [{"path": p, "rank": 0}], None, 0)
            if a:
                arms.append(a)
    if not arms:
        print("no parsable arms")
        sys.exit(1)
    for a in arms:
        print_arm(a)
    result = {"arms": [{k: v for k, v in a.items() if k not in
                        ("streams", "fams", "_meta", "rank_lines")}
                       for a in arms]}
    base = next((a for a in arms if not a["fams"] or
                 not any(r["fam"] == IDX_FAM for r in a["fams"])), arms[0])
    others = [a for a in arms if a is not base]
    if args.meta and others:
        result["diff"] = pair_diff(base, others[0])
    elif arms:
        fallback_verdict(arms[0])
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
