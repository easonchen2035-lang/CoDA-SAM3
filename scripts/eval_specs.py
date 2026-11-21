"""Evaluate a list of method specs on one split's raw candidates, for every seed and a grid of thresholds.

Writes one JSON with per-(spec, seed, tau) metrics. Seed-independent specs (src:T) are evaluated once (seed 0 supplies
nothing for them) and replicated. Usage:
  python scripts/eval_specs.py --dataset kvasir --split val --raw runs/main/kvasir/raw_val_*.pkl --specs-file configs/val_grid.txt --out results/val_kvasir.json
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, glob, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, str(PROJECT_ROOT))
from coda.evaluate import load_raw, run_method

TAUS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
G = {}


def init(raw, gt):
    G["res"], G["draws"], _ = load_raw(raw)
    G["gt"] = json.load(open(gt))


def job(args):
    spec, seed = args
    t = time.time()
    r = run_method(G["res"], G["gt"], spec, seed, taus=TAUS)
    r["sem"] = {str(k): v for k, v in r["sem"].items()}
    return spec, seed, r, time.time() - t


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--dataset", required=True); P.add_argument("--split", required=True)
    P.add_argument("--raw", nargs="+", required=True); P.add_argument("--specs", nargs="*", default=[])
    P.add_argument("--specs-file"); P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    P.add_argument("--procs", type=int, default=24); P.add_argument("--out", required=True)
    a = P.parse_args()
    raw = sorted(set(sum([glob.glob(x) for x in a.raw], [])))
    specs = list(a.specs)
    if a.specs_file:
        specs += [l.strip() for l in open(a.specs_file) if l.strip() and not l.startswith("#")]
    gt = f"{PROJECT_ROOT}/data/proc/{a.dataset}/{a.split}.json"
    jobs = [(s, sd) for s in specs for sd in a.seeds]
    out = {"dataset": a.dataset, "split": a.split, "raw": raw, "taus": TAUS, "runs": []}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with Pool(a.procs, initializer=init, initargs=(raw, gt)) as pool:
        for spec, seed, r, dt in pool.imap_unordered(job, jobs):
            out["runs"].append({"spec": spec, "seed": seed, **r})
            best = max(r["sem"].values(), key=lambda v: v["mIoU"])
            print(f"{spec:45s} s{seed} AP50b {r['ap50_box']['AP50']:.3f} AP50m {r['ap50_mask']['AP50']:.3f} "
                  f"best-tau mIoU {best['mIoU']:.3f}  ({dt:.0f}s)", flush=True)
    json.dump(out, open(a.out, "w"))
    print("WROTE", a.out)
