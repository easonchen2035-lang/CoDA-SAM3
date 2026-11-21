"""Ablation / K-sweep / selection tables on TEST. All rows of the final pipeline family use the FROZEN per-dataset tau
(configs/frozen_selection.json, chosen on val for the final spec); AP50 is threshold-free. Missing -> '--'.
Writes results/ablation_table.md (+ .json)."""
import json, os
import numpy as np

DS = ["kvasir", "isic2018", "nwpu", "isaid"]
TAU = json.load(open("configs/frozen_selection.json"))["tau"]
FULL = "jvote:M1+TM1|vote=a|lam=1"


def cell(f, spec, d):
    if not os.path.exists(f):
        return None
    runs = [r for r in json.load(open(f))["runs"] if r["spec"] == spec]
    if not runs:
        return None
    t = TAU[d]
    return {"ap50_box": [r["ap50_box"]["AP50"] for r in runs], "ap50_mask": [r["ap50_mask"]["AP50"] for r in runs],
            "mIoU": [r["sem"][t]["mIoU"] for r in runs], "mDice": [r["sem"][t]["mDice"] for r in runs]}


def fmt(v, keys=("ap50_box", "mIoU")):
    if v is None:
        return "--"
    f = lambda x: f"{100*np.mean(x):.1f}" + (f"±{100*np.std(x):.1f}" if len(x) > 1 else "")
    return " / ".join(f(v[k]) for k in keys)


out, md = {}, []


def table(title, rows):
    md.extend([f"\n### {title}\n", "| Variant | " + " | ".join(f"{d} AP50box / mIoU" for d in DS) + " |",
               "|" + "---|" * (len(DS) + 1)])
    for label, pat, spec in rows:
        cs = []
        for d in DS:
            v = cell(pat.format(d=d), spec, d)
            if v is not None:
                out.setdefault(title, {}).setdefault(label, {})[d] = v
            cs.append(fmt(v))
        md.append(f"| {label} | " + " | ".join(cs) + " |")


A = "results/ablation_test_{d}.json"
table("D1 component ablation (retrieved K=3)", [
    ("Full (joint alpha-voting M1+TM1, affinity lam=1)", A, FULL),
    ("- affinity re-ranking (lam=0)", A, "jvote:M1+TM1|vote=a|lam=0"),
    ("affinity lam=0.5", A, "jvote:M1+TM1|vote=a|lam=0.5"),
    ("- routing weights (uniform 1/K)", A, "jvote:M1+TM1|vote=u|lam=1"),
    ("+ mask voting", A, "jvote:M1+TM1|vote=a|lam=1|mv=1"),
    ("- consensus (union of decodes + NMS)", A, "pool:M1+TM1|lam=1"),
    ("- consensus - affinity", A, "pool:M1+TM1|lam=0"),
    ("top-1 routed support only (M1+TM1)", A, "top1:M1+TM1|lam=1"),
    ("top-1 routed support only (M1)", A, "top1:M1|lam=1"),
    ("visual mosaic only (vote M1)", A, "vote:M1|vote=a|lam=1"),
    ("text mosaic only (vote TM1)", A, "vote:TM1|vote=a|lam=1"),
])
K = lambda k: ("results/test_final_{d}.json" if k == 3 else f"results/ksweep_test_K{k}_" + "{d}.json")
table("D4 number of retrieved supports K (full method)", [(f"K={k}", K(k), FULL) for k in (1, 3, 5, 10)])
table("D4b number of retrieved supports K (visual mosaic M1)", [(f"K={k}", K(k), "vote:M1|vote=a|lam=1") for k in (1, 3, 5, 10)])
table("D3 support selection (K=3, full method)", [
    ("random K=3 (3 draws, mean±std)", "results/rand_test_{d}.json", FULL),
    ("query-adaptive retrieval K=3", "results/test_final_{d}.json", FULL),
])
open("results/ablation_table.md", "w").write("\n".join(md) + "\n")
json.dump(out, open("results/ablation_table.json", "w"), indent=1)
print("\n".join(md))
