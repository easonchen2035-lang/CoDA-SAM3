"""Assemble the TEST summary table (all 4 datasets). Every method uses the tau selected on VAL:
  final CoDA spec + its variants -> tau from the matching val result of the same pipeline (results/jvote_val_*, gate2_val_*)
  baselines from runs/main     -> tau from results/val_<ds>_summary.json (same spec, val, K=3 random draws)
Missing entries are printed as '--' (not yet computed). Writes results/test_table.md and results/test_table.json."""
import json, os, subprocess
import numpy as np

DS = ["kvasir", "isic2018", "nwpu", "isaid"]
ROWS = [  # (label, test file pattern, spec, val tau source files)
    ("SAM3 text-only", "test_final_{d}.json", "src:T", ["gate2_val_{d}.json", "val_{d}.json"]),
    ("SAM3 text + support affinity", "test_final_{d}.json", "src:T|lam=1", ["gate2_val_{d}.json"]),
    ("Feature matching (FM)", "test_baselines_{d}.json", "fm", ["val_{d}.json"]),
    ("Geometric prompt (G)", "test_baselines_{d}.json", "src:G", ["val_{d}.json"]),
    ("Text + geometric prompt (TG)", "test_baselines_{d}.json", "src:TG", ["val_{d}.json"]),
    ("Cross-image exemplar tokens (E)", "test_baselines_{d}.json", "src:E", ["val_{d}.json"]),
    ("Cross-image tokens + consensus (E1 vote)", "test_baselines_{d}.json", "vote:E1|vote=a|lam=0|mv=1", ["val_{d}.json", "val_{d}_mv.json"]),
    ("Native mosaic, visual (M1 vote)", "test_final_{d}.json", "vote:M1|vote=a|lam=1", ["vtau_val_{d}.json"]),
    ("Native mosaic, text (TM1 vote)", "test_final_{d}.json", "vote:TM1|vote=a|lam=1", ["vtau_val_{d}.json"]),
    ("CoDA-SAM3 (ours, frozen)", "test_final_{d}.json", "jvote:M1+TM1|vote=a|lam=1", ["jvote_val_{d}.json"]),
]


def val_tau(d, spec, srcs):
    for s in srcs:
        f = f"results/{s.format(d=d)}"
        if not os.path.exists(f):
            continue
        sf = f.replace(".json", "_summary.json")
        if not os.path.exists(sf) or os.path.getmtime(sf) < os.path.getmtime(f):
            subprocess.run([".venv/bin/python", "scripts/summarize.py", f, "--out", sf], capture_output=True)
        S = json.load(open(sf))["specs"]
        if spec in S:
            return S[spec]["tau"]
    return None


def need_val_tau_for_mosaic(d, spec):
    # M1/TM1 with lam=1 were not in the val grid at lam=1 for all sets; fall back to the closest evaluated val spec family
    for s in ["gate2_val_{d}.json", "jvote_val_{d}.json", "sel_val_mosaic_retr_{d}.json"]:
        t = val_tau(d, spec, [s])
        if t is not None:
            return t
    return None


out, md = {}, []
hdr = "| Method | " + " | ".join(f"{d} AP50box / AP50mask / mIoU / mDice" for d in DS) + " |"
md += [hdr, "|" + "---|" * (len(DS) + 1)]
for label, pat, spec, srcs in ROWS:
    cells = []
    for d in DS:
        f = f"results/{pat.format(d=d)}"
        tau = val_tau(d, spec, srcs) or need_val_tau_for_mosaic(d, spec)
        if not os.path.exists(f) or tau is None:
            cells.append("--"); continue
        runs = [r for r in json.load(open(f))["runs"] if r["spec"] == spec]
        if not runs:
            cells.append("--"); continue
        v = {k: [r[k]["AP50"] for r in runs] for k in ["ap50_box", "ap50_mask"]}
        v["mIoU"] = [r["sem"][tau]["mIoU"] for r in runs]; v["mDice"] = [r["sem"][tau]["mDice"] for r in runs]
        out.setdefault(label, {})[d] = {"tau_from_val": tau, "n_draws": len(runs), **{k: list(map(float, x)) for k, x in v.items()}}
        fmt = lambda x: f"{100*np.mean(x):.1f}" + (f"±{100*np.std(x):.1f}" if len(x) > 1 else "")
        cells.append(" / ".join(fmt(v[k]) for k in ["ap50_box", "ap50_mask", "mIoU", "mDice"]))
    md.append(f"| {label} | " + " | ".join(cells) + " |")
open("results/test_table.md", "w").write("\n".join(md) + "\n")
json.dump(out, open("results/test_table.json", "w"), indent=1)
print("\n".join(md))
