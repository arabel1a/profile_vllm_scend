#!/usr/bin/env python3
"""Cross-cell comparison for the profiler matrix produced by n36_prof_matrix.sh.

Reads every <root>/<tag>/<phase>/<arm>/kernel_details_rank0.csv (rank0 is the
only rank kept -- see the header of n36_prof_matrix.sh) and prints

  1. one summary row per cell/phase/arm: steps in the window, device busy and
     busiest-stream busy per step, wall per step, client TPOT, acceptance
  2. the LoRA tax of every LoRA cell: lora arm minus base arm, in device us per
     step, split into lora / comm / other kernels
  3. a per-family us/step matrix across all cells, so a family that moved
     between branches is visible without diffing two reports by hand
  4. the three named A/Bs: overlap on vs off per branch, scatter on vs off
     (night_scatter vs night_z1z2, which differ by nothing else), and every
     branch against addlora_path

Self-contained: it needs nothing but these csvs (no lora_prof_analyze_2.py,
no torch). Normalisation is per decode step, using the same marker kernels that
analyzer uses (combined_lora_idx_finalize/40, else HcPre/86), so cells
whose capture windows contained different numbers of steps stay comparable.
Absolute per-step numbers are still profiled numbers: a capture window perturbs
what it measures, and the client TPOT column is the unperturbed reference.

  python3 n37_prof_compare.py --root logs/prof_matrix [--out compare.json]
"""
import argparse
import collections
import csv as _csv
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Self-contained copies of the four primitives lora_prof_analyze_2.py uses, so
# this script runs on a machine that only has the matrix scripts on it. The
# family grouping, the marker divisors and the lora/comm classification are kept
# identical to that analyzer on purpose: the two must agree or the per-cell
# reports and this table would disagree about the same capture.
# ---------------------------------------------------------------------------
TYPE_TAILS = re.compile(r"_(?:bfloat16|float16|fp16|fp32|i8|i16|i32|int32|i64|t)_")
COMM_RE = re.compile(
    r"HcPre|HcPost|AivKernel|Hccl|AllGather|AllReduce|AllSend|AllWrite"
    r"|Receive|Broadcast|ReduceScatter|Exchange")
IDX_FAM = "combined_lora_idx_finalize"
HC_FAM = "HcPre"
IDX_PER_STEP = 40
HC_PER_STEP = 86


def family(name):
    f = TYPE_TAILS.split(name, 1)[0] or name
    if f.startswith("aclnn"):
        f = f.split("_", 1)[0]
    return f


def classify(f, core, name):
    if COMM_RE.search(f) or COMM_RE.search(name):
        return "comm"
    if ("add_lora" in f or "bgmv" in f or "sgmv" in f
            or "combined_lora_idx" in f or f == "aclnnSort"):
        return "lora"
    return "other"


def parse_csv(path):
    fams, streams = {}, collections.defaultdict(list)
    min_s = max_e = None
    busy = 0.0
    try:
        fh = open(path, newline="")
    except OSError as e:
        print(f"  !! cannot open {path}: {e}")
        return None
    with fh:
        rd = _csv.reader(fh)
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
            name, core = row[ix["Name"]], row[ix["Accelerator Core"]]
            busy += dur
            streams[row[ix["Stream ID"]]].append((start, dur))
            if min_s is None or start < min_s:
                min_s = start
            if max_e is None or start + dur > max_e:
                max_e = start + dur
            d = fams.setdefault(family(name), {"cnt": 0, "tot": 0.0,
                                               "core": core, "name": name})
            d["cnt"] += 1
            d["tot"] += dur
    return {"fams": fams, "streams": dict(streams), "min": min_s or 0.0,
            "max": max_e or 0.0, "busy": busy}


def steps_of(parsed):
    idx = parsed["fams"].get(IDX_FAM, {}).get("cnt", 0)
    hc = parsed["fams"].get(HC_FAM, {}).get("cnt", 0)
    if idx:
        return idx / IDX_PER_STEP, f"{IDX_FAM}={idx}/{IDX_PER_STEP}"
    if hc:
        return hc / HC_PER_STEP, f"{HC_FAM}={hc}/{HC_PER_STEP}"
    return 0.0, "no marker kernel"


def stream_rank(parsed):
    out = []
    for s, evs in parsed["streams"].items():
        evs.sort()
        out.append({"stream": s, "busy": sum(d for _, d in evs)})
    out.sort(key=lambda x: -x["busy"])
    return out


def load_metrics(path):
    m = {"mean_tpot_ms": None, "decode_tpot_ms": None, "delivered": None}
    if not path or not os.path.exists(path):
        return m
    for line in open(path, errors="replace"):
        mm = re.search(r"TOTAL delivered=(\d+) tok .*?mean_tpot=([\d.]+)", line)
        if mm:
            m["delivered"] = int(mm[1])
            m["mean_tpot_ms"] = float(mm[2])
            md = re.search(r"decode_tpot=([\d.]+)", line)
            if md:
                m["decode_tpot_ms"] = float(md[1])
    return m

# cell order = the matrix order in n36_prof_matrix.sh
CELL_ORDER = ["test_nolora", "test_lora",
              "addlora_lora_ov0", "addlora_lora_ov1",
              "z1z2_lora_ov0", "z1z2_lora_ov1", "z1z2_nolora",
              "scatter_lora_ov0", "scatter_lora_ov1", "scatter_nolora"]

# (label, cell_a, cell_b, arm) -> reports a - b
NAMED_ABS = [
    ("overlap on-off, addlora_path", "addlora_lora_ov1", "addlora_lora_ov0", "lora-adapter1"),
    ("overlap on-off, night_z1z2", "z1z2_lora_ov1", "z1z2_lora_ov0", "lora-adapter1"),
    ("overlap on-off, night_scatter", "scatter_lora_ov1", "scatter_lora_ov0", "lora-adapter1"),
    ("scatter on-off, NO lora", "scatter_nolora", "z1z2_nolora", "ds"),
    ("scatter on-off, lora ov0", "scatter_lora_ov0", "z1z2_lora_ov0", "lora-adapter1"),
    ("scatter on-off, lora ov1", "scatter_lora_ov1", "z1z2_lora_ov1", "lora-adapter1"),
    ("z1z2 vs addlora_path, ov0", "z1z2_lora_ov0", "addlora_lora_ov0", "lora-adapter1"),
    ("z1z2 vs addlora_path, ov1", "z1z2_lora_ov1", "addlora_lora_ov1", "lora-adapter1"),
    ("addlora_path vs stock test", "addlora_lora_ov0", "test_lora", "lora-adapter1"),
    ("no-lora: night_z1z2 vs stock test", "z1z2_nolora", "test_nolora", "ds"),
]


def load_arm(adir):
    csv = os.path.join(adir, "kernel_details_rank0.csv")
    if not os.path.exists(csv):
        return None
    p = parse_csv(csv)
    if not p:
        return None
    steps, how = steps_of(p)
    if steps <= 0:
        print(f"  !! {adir}: no marker kernel, cannot normalise per step")
        return None
    streams = stream_rank(p)
    lm = load_metrics(os.path.join(adir, "load.log"))
    acc = None
    accf = os.path.join(adir, "acc.txt")
    if os.path.exists(accf):
        parts = open(accf).read().split()
        if parts:
            acc = float(parts[0])
    fams = {}
    for f, d in p["fams"].items():
        fams[f] = {"us_per_step": d["tot"] / steps, "cnt_per_step": d["cnt"] / steps,
                   "avg_us": d["tot"] / max(d["cnt"], 1),
                   "cls": classify(f, d["core"], d["name"]), "core": d["core"]}
    return {"steps": steps, "marker": how,
            "busy_us_per_step": p["busy"] / steps,
            "stream0_us_per_step": (streams[0]["busy"] / steps) if streams else 0.0,
            "stream1_us_per_step": (streams[1]["busy"] / steps) if len(streams) > 1 else 0.0,
            "wall_ms_per_step": (p["max"] - p["min"]) / 1000.0 / steps,
            "client_tpot_ms": lm.get("mean_tpot_ms"),
            "decode_tpot_ms": lm.get("decode_tpot_ms"),
            "acceptance": acc, "fams": fams}


def collect(root):
    cells = {}
    for tag in sorted(os.listdir(root)):
        cdir = os.path.join(root, tag)
        if not os.path.isdir(cdir):
            continue
        for phase in sorted(os.listdir(cdir)):
            pdir = os.path.join(cdir, phase)
            meta_p = os.path.join(pdir, "meta.json")
            if not os.path.isdir(pdir) or not os.path.exists(meta_p):
                continue
            meta = json.load(open(meta_p))
            arms = {}
            for e in meta["arms"]:
                a = load_arm(os.path.join(pdir, e["arm"]))
                if a:
                    arms[e["arm"]] = a
            if arms:
                cells[(tag, phase)] = {"meta": meta.get("cell", {}), "arms": arms,
                                       "git": meta.get("node", {}).get("git", "")}
    return cells


def order_key(k):
    tag, phase = k
    return (phase, CELL_ORDER.index(tag) if tag in CELL_ORDER else 99, tag)


def print_summary(cells):
    print("\n=== per cell/phase/arm (rank0, per decode step) ===")
    print(f"{'cell':<18}{'phase':<9}{'arm':<15}{'steps':>7}{'busy us':>10}"
          f"{'strm0':>9}{'strm1':>8}{'wall ms':>9}{'tpot ms':>9}{'acc':>6}  marker")
    for k in sorted(cells, key=order_key):
        tag, phase = k
        for arm, a in cells[k]["arms"].items():
            print(f"{tag:<18}{phase:<9}{arm:<15}{a['steps']:>7.0f}"
                  f"{a['busy_us_per_step']:>10.0f}{a['stream0_us_per_step']:>9.0f}"
                  f"{a['stream1_us_per_step']:>8.0f}{a['wall_ms_per_step']:>9.2f}"
                  f"{(a['client_tpot_ms'] or 0):>9.2f}{(a['acceptance'] or 0):>6.2f}"
                  f"  {a['marker']}")


def print_tax(cells, out):
    print("\n=== LoRA tax per cell (lora arm minus base arm, device us/step) ===")
    print(f"{'cell':<18}{'phase':<9}{'total':>9}{'lora':>9}{'comm':>9}{'other':>9}"
          f"{'wall ms':>9}  top movers")
    for k in sorted(cells, key=order_key):
        tag, phase = k
        arms = cells[k]["arms"]
        base = "ds"
        lora = next((x for x in arms if x != base), None)
        if base not in arms or lora is None:
            continue
        b, l = arms[base], arms[lora]
        allf = set(b["fams"]) | set(l["fams"])
        per_cls = {"lora": 0.0, "comm": 0.0, "other": 0.0}
        movers = []
        for f in allf:
            db = b["fams"].get(f, {}).get("us_per_step", 0.0)
            dl = l["fams"].get(f, {}).get("us_per_step", 0.0)
            d = dl - db
            cls = (l["fams"].get(f) or b["fams"].get(f))["cls"]
            per_cls[cls if cls in per_cls else "other"] += d
            if abs(d) >= 5.0:
                movers.append((d, f))
        movers.sort(key=lambda x: -abs(x[0]))
        tot = sum(per_cls.values())
        out.setdefault("tax", {})[f"{tag}/{phase}"] = {
            "total_us_per_step": tot, **{f"{c}_us_per_step": v for c, v in per_cls.items()},
            "wall_delta_ms_per_step": l["wall_ms_per_step"] - b["wall_ms_per_step"],
            "movers": [{"fam": f, "d_us_per_step": d} for d, f in movers[:12]]}
        print(f"{tag:<18}{phase:<9}{tot:>9.0f}{per_cls['lora']:>9.0f}"
              f"{per_cls['comm']:>9.0f}{per_cls['other']:>9.0f}"
              f"{l['wall_ms_per_step'] - b['wall_ms_per_step']:>9.2f}  "
              + ", ".join(f"{f} {d:+.0f}" for d, f in movers[:4]))


def print_family_matrix(cells, phase, arm_pref, thresh=10.0):
    ks = [k for k in sorted(cells, key=order_key) if k[1] == phase]
    cols = []
    for k in ks:
        arms = cells[k]["arms"]
        arm = next((a for a in arm_pref if a in arms), None)
        if arm:
            cols.append((k[0], arm, arms[arm]))
    if not cols:
        return
    fams = {}
    for _, _, a in cols:
        for f, d in a["fams"].items():
            fams[f] = max(fams.get(f, 0.0), d["us_per_step"])
    keep = [f for f, v in sorted(fams.items(), key=lambda x: -x[1]) if v >= thresh][:28]
    print(f"\n=== {phase}: device us/step per kernel family "
          f"(arm preference {'/'.join(arm_pref)}; '-' = family absent) ===")
    print("family".ljust(34) + "".join(f"{t[:11]:>12}" for t, _, _ in cols))
    print("arm".ljust(34) + "".join(f"{a[:11]:>12}" for _, a, _ in cols))
    for f in keep:
        row = "".join(
            (f"{a['fams'][f]['us_per_step']:>12.1f}" if f in a["fams"] else f"{'-':>12}")
            for _, _, a in cols)
        print(f"{f[:33]:<34}{row}")


def print_named(cells, phase, out):
    print(f"\n=== {phase}: named A/Bs (a minus b, device us/step; "
          f"neg = a is cheaper) ===")
    for label, ca, cb, arm in NAMED_ABS:
        ka, kb = (ca, phase), (cb, phase)
        if ka not in cells or kb not in cells:
            continue
        aa = cells[ka]["arms"].get(arm) or next(iter(cells[ka]["arms"].values()))
        ab = cells[kb]["arms"].get(arm) or next(iter(cells[kb]["arms"].values()))
        d_busy = aa["busy_us_per_step"] - ab["busy_us_per_step"]
        d_wall = aa["wall_ms_per_step"] - ab["wall_ms_per_step"]
        allf = set(aa["fams"]) | set(ab["fams"])
        movers = sorted(
            ((aa["fams"].get(f, {}).get("us_per_step", 0.0)
              - ab["fams"].get(f, {}).get("us_per_step", 0.0), f) for f in allf),
            key=lambda x: -abs(x[0]))
        out.setdefault("named", {})[f"{label} [{phase}]"] = {
            "arm": arm, "a": ca, "b": cb,
            "d_busy_us_per_step": d_busy, "d_wall_ms_per_step": d_wall,
            "movers": [{"fam": f, "d_us_per_step": d} for d, f in movers[:10]
                       if abs(d) >= 2.0]}
        print(f"{label:<36} arm={arm:<14} busy {d_busy:+8.0f} us  "
              f"wall {d_wall:+6.2f} ms")
        for d, f in movers[:5]:
            if abs(d) >= 2.0:
                print(f"      {f[:40]:<42}{d:+8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/prof_matrix")
    ap.add_argument("--out")
    args = ap.parse_args()
    if not os.path.isdir(args.root):
        print(f"no such root: {args.root}")
        sys.exit(1)
    cells = collect(args.root)
    if not cells:
        print(f"no finished cells under {args.root}")
        sys.exit(1)
    print(f"cells found: {len(cells)}")
    for k in sorted(cells, key=order_key):
        m = cells[k]["meta"]
        print(f"  {k[0]:<18} {k[1]:<8} branch={m.get('branch','?'):<16}"
              f" lora={m.get('lora','?')} overlap={m.get('overlap','?')}"
              f" arms={list(cells[k]['arms'])}")
        if cells[k]["git"]:
            print(f"      {cells[k]['git'].splitlines()[0]}")
    out = {"root": os.path.abspath(args.root),
           "cells": {f"{t}/{p}": {"meta": c["meta"], "git": c["git"],
                                  "arms": {a: {k2: v2 for k2, v2 in v.items() if k2 != "fams"}
                                           for a, v in c["arms"].items()},
                                  "fams": {a: v["fams"] for a, v in c["arms"].items()}}
                     for (t, p), c in cells.items()}}
    print_summary(cells)
    print_tax(cells, out)
    for phase in sorted({p for _, p in cells}):
        print_family_matrix(cells, phase, ["lora-adapter1", "ds"])
        print_family_matrix(cells, phase, ["ds"])
        print_named(cells, phase, out)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
