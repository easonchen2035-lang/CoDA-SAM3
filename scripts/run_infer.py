"""GPU stage: run frozen SAM3 under several prompt configurations and save ALL raw candidates.

For every query image q (split val|test), class c, support draw seed s (K supports per class from the pool):
  T    : text phrase only                                   (seed independent; generalist baseline)
  E    : 'visual' + cross-image exemplar tokens of all K supports  (direct exemplar baseline; no class name)
  TE   : text + exemplar tokens of all K supports
  TEr  : text + exemplar tokens of the top-r routed supports (r=1)
  G    : 'visual' + in-image boxes from the support response map (feature-matching -> geometric prompt baseline)
  TG   : text + response-map boxes
  TEG  : text + all-K exemplar tokens + response-map boxes
Saved per candidate: SAM3 score, box (eval-res xyxy), mask RLE (eval res, from bilinear-upsampled logits > 0),
affinities A_w / A_u (mean of weighted / uniform-routing response inside the mask on the 72x72 grid).
Saved per (q,c,s): routing weights, response maps R_w, R_u (float16, 72x72), geometric boxes.
No query ground truth is read by this script except to list query images.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, json, os, pickle, sys, time
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mu
from coda.sam3core import FrozenSAM3, cat_prompts
from coda import gpu_rle
from coda.support import draw_supports, build_memory, route, response, response_boxes

P = argparse.ArgumentParser()
P.add_argument("--dataset", required=True)
P.add_argument("--split", required=True)
P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
P.add_argument("--K", type=int, default=3)
P.add_argument("--configs", nargs="+", default=["T", "E", "TE", "TEr", "TE1", "E1", "G", "TG", "TEG"])
P.add_argument("--E", type=int, default=3, help="max exemplar instances per support image")
P.add_argument("--coord_mode", default="nopos")
P.add_argument("--feat_source", default="fpn72", choices=["fpn72", "vit"])
P.add_argument("--resp", default="knn", choices=["knn", "proto"])
P.add_argument("--rel_thr", type=float, default=0.5)
P.add_argument("--max_boxes", type=int, default=3)
P.add_argument("--route_temp", type=float, default=0.05)
P.add_argument("--keep_max", type=int, default=None)
P.add_argument("--keep_floor", type=float, default=0.01)
P.add_argument("--keep_top", type=int, default=3)
P.add_argument("--limit", type=int, default=None)
P.add_argument("--shard", type=int, default=0)
P.add_argument("--nshards", type=int, default=1)
P.add_argument("--out", required=True)
a = P.parse_args()

D = f"{PROJECT_ROOT}/data/proc/{a.dataset}"
pool = json.load(open(f"{D}/pool.json"))
qco = json.load(open(f"{D}/{a.split}.json"))
cats = [(c["id"], c["name"]) for c in qco["categories"]]
multi = len(cats) > 1
keep_max = a.keep_max or (100 if multi else 10)
queries = qco["images"][: a.limit] if a.limit else qco["images"]
queries = queries[a.shard::a.nshards]
os.makedirs(a.out, exist_ok=True)
tag = f"{a.split}_K{a.K}_s{'-'.join(map(str, a.seeds))}_sh{a.shard}of{a.nshards}"
json.dump(vars(a), open(f"{a.out}/args_{tag}.json", "w"), indent=1)

torch.manual_seed(0)
S = FrozenSAM3(feat_source=a.feat_source)
t0 = time.time()

# ---------------- support memories (cached per seed/class); support encodings are released afterwards
mems, draws = {}, {}
enc_cache = {}
for s in a.seeds:
    for cid, _ in cats:
        ids = draw_supports(pool, cid, a.K, s, a.dataset)
        draws[f"{s}|{cid}"] = ids
        mems[(s, cid)] = build_memory(S, pool, f"{D}/images", cid, ids, enc_cache, E=a.E, coord_mode=a.coord_mode)
enc_cache.clear(); torch.cuda.empty_cache()
json.dump(draws, open(f"{a.out}/draws_{tag}.json", "w"))
print(f"support memory built in {time.time()-t0:.1f}s", flush=True)

texts = {cid: S.encode_text(name) for cid, name in cats}
visual = S.encode_text("visual")


def candidates(o, H, W, Rw, Ru, return_m72=False):
    sc = o["scores"]
    order = torch.argsort(sc, descending=True)
    keep = [int(j) for r, j in enumerate(order[:keep_max]) if r < a.keep_top or sc[j] >= a.keep_floor]
    if not keep:
        return ([], None) if return_m72 else []
    ml = o["mask_logits"][keep]                                   # [n,288,288]
    # affinity grid: mask logits bilinearly resized to the 72x72 response grid, > 0
    m72 = (F.interpolate(ml[None], size=(72, 72), mode="bilinear", align_corners=False)[0] > 0).float()
    if Rw is not None:
        area72 = m72.sum((1, 2)).clamp_min(1)
        Aw = ((m72 * Rw).sum((1, 2)) / area72).tolist()
        Au = ((m72 * Ru).sum((1, 2)) / area72).tolist()
    else:
        Aw = Au = [None] * len(keep)
    full = F.interpolate(ml[None], size=(H, W), mode="bilinear", align_corners=False)[0] > 0
    areas = full.sum((1, 2)).tolist()
    rles = gpu_rle.encode(full)
    boxes = (o["boxes"][keep] * torch.tensor([W, H, W, H], device=sc.device)).tolist()
    out = [{"score": float(sc[j]), "box": boxes[n], "rle": rles[n], "Aw": Aw[n], "Au": Au[n], "area": int(areas[n])}
           for n, j in enumerate(keep)]
    return (out, m72) if return_m72 else out


results = []
timing = []
for qi, im in enumerate(queries):
    tq = time.time()
    H, W = im["height"], im["width"]
    enc = S.encode_image(Image.open(f"{D}/images/{im['file_name']}"))
    rec = {"image_id": im["id"], "file_name": im["file_name"], "classes": {}}
    for cid, _ in cats:
        T = texts[cid]
        crec = {"T": None, "seeds": {}}
        if "T" in a.configs:
            tc, tm72 = candidates(S.decode(enc, *T), H, W, None, None, return_m72=True)
            crec["T"] = {"cands": tc}
        for s in a.seeds:
            mem = mems[(s, cid)]
            alpha = route(mem, enc.gfeat, a.route_temp)
            uni = torch.full_like(alpha, 1.0 / len(alpha))
            Rw, Ru = response(mem, enc.feat, alpha, a.resp), response(mem, enc.feat, uni, a.resp)
            gb = response_boxes(Rw, a.rel_thr, a.max_boxes)
            gtok = S.geometry_tokens(enc, [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in gb]) if gb else None
            ex_all = cat_prompts(mem.tokens)
            top = int(alpha.argmax())
            ex_r = mem.tokens[top]
            prompts = {"E": [visual, ex_all], "TE": [T, ex_all], "TEr": [T, ex_r],
                       "G": [visual, gtok] if gtok else None, "TG": [T, gtok] if gtok else None,
                       "TEG": [T, ex_all, gtok] if gtok else [T, ex_all]}
            srec = {"alpha": alpha.tolist(), "gboxes": gb, "Rw": Rw.half().cpu().numpy(), "Ru": Ru.half().cpu().numpy(),
                    "Rmax": float(Rw.max())}
            for cfg in a.configs:
                if cfg == "T":
                    continue
                if cfg in ("TE1", "E1"):   # one decode per support entry (its own exemplar tokens only)
                    srec[cfg] = {"per_support": []}
                    for i, tk in enumerate(mem.tokens):
                        o = S.decode(enc, *cat_prompts([T if cfg == "TE1" else visual, tk]))
                        srec[cfg]["per_support"].append({"cands": candidates(o, H, W, Rw, Ru), "presence": o["presence"]})
                    continue
                parts = prompts[cfg]
                if parts is None:   # no geometric prior available (max response <= 0): no candidates
                    srec[cfg] = {"cands": [], "empty_prompt": True}
                    continue
                o = S.decode(enc, *cat_prompts(parts))
                srec[cfg] = {"cands": candidates(o, H, W, Rw, Ru), "presence": o["presence"]}
            # affinities for T candidates under this seed's response (T itself is seed independent)
            if crec["T"] is not None:
                if tc:
                    ar = tm72.sum((1, 2)).clamp_min(1)
                    srec["T_aff"] = ((tm72 * Rw).sum((1, 2)) / ar).tolist()
                    srec["T_affu"] = ((tm72 * Ru).sum((1, 2)) / ar).tolist()
                else:
                    srec["T_aff"], srec["T_affu"] = [], []
            crec["seeds"][s] = srec
        rec["classes"][cid] = crec
    results.append(rec)
    torch.cuda.synchronize()
    timing.append(time.time() - tq)
    if qi % 20 == 0:
        print(f"[{qi+1}/{len(queries)}] {np.mean(timing):.2f}s/img  elapsed {time.time()-t0:.0f}s", flush=True)

with open(f"{a.out}/raw_{tag}.pkl", "wb") as f:
    pickle.dump({"args": vars(a), "results": results, "draws": draws}, f)
print("DONE", len(results), f"{time.time()-t0:.0f}s", flush=True)
