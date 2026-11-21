"""Memory-light parallel evaluation: tasks = (raw shard file, seed, chunk of specs).
Workers load ONE raw shard, assemble final instances for each spec, and return per-image match records
(score, TP@0.5 box, TP@0.5 mask) + semantic (I, |P|, |G|) tables. The master merges shards and computes
COCO-identical AP50 (coda/ap.py) and the semantic metrics (coda/evaluate.semantic). Output = eval_specs.py format."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, glob, json, os, pickle, sys, time
from multiprocessing import Pool
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from pycocotools import mask as mu
from coda.evaluate import assemble, semantic
from coda.ap import image_records, ap_from_records

TAUS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
G = {"file": None}


def init(gt_path):
    gt = json.load(open(gt_path))
    rp = os.path.join(os.path.dirname(gt_path), "pool_text_reliability.json")
    G["pool_rel"] = json.load(open(rp)) if os.path.exists(rp) else None
    G["imgs"] = {im["id"]: im for im in gt["images"]}
    G["cats"] = [c["id"] for c in gt["categories"]]
    G["gta"] = {}
    for a in gt["annotations"]:
        G["gta"].setdefault((a["image_id"], a["category_id"]), []).append(a)


def load(f):
    """f may be 'main.pkl' or 'main.pkl::extra.pkl' (extra = TM1-only run of the same queries/supports; its per-seed
    configs are merged into the main records, after checking that the selected support ids are identical)."""
    if G["file"] != f:
        parts = f.split("::")
        d = pickle.load(open(parts[0], "rb"))
        res = d["results"]
        for extra in parts[1:]:
            e = {r["image_id"]: r for r in pickle.load(open(extra, "rb"))["results"]}
            for r in res:
                er = e[r["image_id"]]
                for cid, c in r["classes"].items():
                    for sd, sr in c["seeds"].items():
                        es = er["classes"][cid]["seeds"][sd]
                        assert es.get("support_ids") == sr.get("support_ids"), "support mismatch in merge"
                        for k, v in es.items():
                            if isinstance(v, dict) and "per_support" in v and k not in sr:
                                sr[k] = v
        G["res"], G["draws"] = res, d["draws"]; G["file"] = f
    return G["res"]


def work(task):
    f, seed, specs = task
    t = time.time()
    res = load(f)
    out = {}
    for spec in specs:
        recs, sem = [], {tau: {} for tau in TAUS}
        for rec in res:
            iid = rec["image_id"]; im = G["imgs"][iid]; H, W = im["height"], im["width"]
            for cid in G["cats"]:
                rc = rec["classes"].get(cid, rec["classes"].get(str(cid)))
                inst = assemble(rc, seed, spec, H, W, cid=cid, ctx={"draws": G["draws"], "pool_rel": G["pool_rel"], "cats": G["cats"]})
                gts = G["gta"].get((iid, cid), [])
                sb, tb = image_records(inst, gts, "bbox")
                sm, tm = image_records(inst, gts, "segm")
                recs.append((iid, cid, sb, tb, tm))
                gm = mu.merge([g["segmentation"] for g in gts]) if gts else None
                gA = int(mu.area(gm)) if gm else 0
                for tau in TAUS:
                    ps = [d["rle"] for d in inst if d["s"] >= tau]
                    pm = mu.merge(ps) if ps else None
                    pA = int(mu.area(pm)) if pm else 0
                    I = int(mu.area(mu.merge([pm, gm], intersect=True))) if (pm and gm) else 0
                    sem[tau][(iid, cid)] = (I, pA, gA)
        out[spec] = (recs, sem)
    return f, seed, out, time.time() - t


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--dataset", required=True); P.add_argument("--split", required=True)
    P.add_argument("--raw", nargs="+", required=True); P.add_argument("--specs", nargs="*", default=[])
    P.add_argument("--specs-file"); P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    P.add_argument("--chunk", type=int, default=8); P.add_argument("--procs", type=int, default=48)
    P.add_argument("--out", required=True)
    a = P.parse_args()
    raw = sorted(set(sum([glob.glob(x) if "::" not in x else [x] for x in a.raw], [])))
    specs = list(a.specs)
    if a.specs_file:
        specs += [l.strip() for l in open(a.specs_file) if l.strip() and not l.startswith("#")]
    gt_path = f"{PROJECT_ROOT}/data/proc/{a.dataset}/{a.split}.json"
    gt = json.load(open(gt_path))
    cats = [c["id"] for c in gt["categories"]]
    ngt = {c: sum(1 for x in gt["annotations"] if x["category_id"] == c) for c in cats}
    chunks = [specs[i:i + a.chunk] for i in range(0, len(specs), a.chunk)]
    tasks = [(f, s, ch) for f in raw for s in a.seeds for ch in chunks]
    acc = {}
    t0 = time.time()
    with Pool(a.procs, initializer=init, initargs=(gt_path,)) as pool:
        for n, (f, seed, out, dt) in enumerate(pool.imap_unordered(work, tasks)):
            for spec, (recs, sem) in out.items():
                A = acc.setdefault((spec, seed), {"recs": [], "sem": {t: {} for t in TAUS}})
                A["recs"] += recs
                for t in TAUS:
                    A["sem"][t].update(sem[t])
            print(f"task {n+1}/{len(tasks)} {os.path.basename(f)} s{seed} ({dt:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    result = {"dataset": a.dataset, "split": a.split, "raw": raw, "taus": TAUS, "runs": []}
    for (spec, seed), A in sorted(acc.items()):
        recs = sorted(A["recs"], key=lambda r: (r[0], r[1]))
        r = {"spec": spec, "seed": seed}
        for key, idx in [("ap50_box", 3), ("ap50_mask", 4)]:
            pc = {}
            for c in cats:
                rr = [x for x in recs if x[1] == c]
                sc = np.concatenate([x[2] for x in rr]) if rr else np.zeros(0)
                tp = np.concatenate([x[idx] for x in rr]) if rr else np.zeros(0, bool)
                v = ap_from_records(sc, tp, ngt[c])
                if v is not None:
                    pc[int(c)] = v
            r[key] = {"AP50": float(np.mean(list(pc.values()))), "per_class": pc}
        r["sem"] = {str(t): semantic(A["sem"][t], cats) for t in TAUS}
        result["runs"].append(r)
        best = max(r["sem"].values(), key=lambda v: v["mIoU"])
        print(f"{spec:45s} s{seed} AP50b {r['ap50_box']['AP50']:.3f} AP50m {r['ap50_mask']['AP50']:.3f} best-tau mIoU {best['mIoU']:.3f}")
    json.dump(result, open(a.out, "w"))
    print("WROTE", a.out)
