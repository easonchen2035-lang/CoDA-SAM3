"""GPU stage for in-context SCHEME / SUPPORT-SELECTION studies.

Support selection schemes (k supports per class, chosen from a labelled pool subset of size N drawn with `seed`):
  rand : k random pool images (identical to run_infer draws when N=all)                [query independent]
  retr : top-k pool images by cos(g_q, g_i) (SAM3 FPN global descriptor)               [query adaptive]
  div  : k-center greedy (farthest-point) over pool descriptors, start = medoid        [query independent]
  mmr  : maximal marginal relevance, score = 0.7*cos(g_q,g_i) - 0.3*max_j∈S cos(g_i,g_j) [query adaptive + diverse]
In-context schemes stored for the selected supports (same record format as run_infer, readable by eval_fast):
  T (text), E1/TE1 (per-support cross-image exemplar decodes, for consensus voting),
  M1 (per-support NATIVE mosaic: support | query stitched on one canvas, the support's class boxes given as SAM3's
      native in-image exemplar boxes with the 'visual' prompt; only detections whose box centre lies in the query
      half are kept and mapped back to query coordinates).
Per-(q,c,s) record also stores `support_ids` (the selected pool image ids), used by the text gate."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, json, os, pickle, random, sys, time
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from coda.sam3core import FrozenSAM3, cat_prompts
from coda import gpu_rle
from coda.support import build_memory, route, response, ClassMemory

P = argparse.ArgumentParser()
P.add_argument("--dataset", required=True); P.add_argument("--split", required=True)
P.add_argument("--scheme", required=True, choices=["rand", "retr", "div", "mmr"])
P.add_argument("--N", type=int, default=0, help="pool subset size per class (0 = whole pool)")
P.add_argument("--K", type=int, default=3); P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
P.add_argument("--configs", nargs="+", default=["T", "E1", "TE1"])
P.add_argument("--keep_floor", type=float, default=0.01); P.add_argument("--keep_top", type=int, default=3)
P.add_argument("--shard", type=int, default=0); P.add_argument("--nshards", type=int, default=1)
P.add_argument("--limit", type=int, default=None); P.add_argument("--out", required=True)
a = P.parse_args()

D = f"{PROJECT_ROOT}/data/proc/{a.dataset}"
pool = json.load(open(f"{D}/pool.json")); qco = json.load(open(f"{D}/{a.split}.json"))
cats = [(c["id"], c["name"]) for c in qco["categories"]]
keep_max = 100 if len(cats) > 1 else 10
queries = (qco["images"][: a.limit] if a.limit else qco["images"])[a.shard::a.nshards]
os.makedirs(a.out, exist_ok=True)
tag = f"{a.split}_K{a.K}_s{'-'.join(map(str, a.seeds))}_sh{a.shard}of{a.nshards}"
json.dump(vars(a), open(f"{a.out}/args_{tag}.json", "w"), indent=1)
S = FrozenSAM3(feat_source="fpn72")
t0 = time.time()
pim = {im["id"]: im for im in pool["images"]}

# ---------------- whole-pool memory, one entry per (class, pool image); encodings released immediately
pool_mem = {cid: [] for cid, _ in cats}      # cid -> list of (image_id, single-entry ClassMemory)
cls_imgs = {cid: sorted({x["image_id"] for x in pool["annotations"] if x["category_id"] == cid}) for cid, _ in cats}
for iid in sorted(pim):
    cids = [c for c, _ in cats if iid in set(cls_imgs[c])]
    if not cids:
        continue
    cache = {}
    for cid in cids:
        pool_mem[cid].append((iid, build_memory(S, pool, f"{D}/images", cid, [iid], cache)))
    cache.clear()
torch.cuda.empty_cache()
print(f"pool memory: {sum(len(v) for v in pool_mem.values())} entries in {time.time()-t0:.0f}s", flush=True)


def stack(entries):
    ms = [m for _, m in entries]
    return ClassMemory(ms[0].cat_id, [i for i, _ in entries], torch.cat([m.v for m in ms]), torch.cat([m.b for m in ms]),
                       torch.cat([m.g for m in ms]), sum([m.tokens for m in ms], []),
                       np.concatenate([m.struct for m in ms]), sum([m.n_inst for m in ms], []),
                       sum([m.fg_patches for m in ms], []), sum([m.bg_patches for m in ms], []))


def pool_subset(cid, seed):
    ent = pool_mem[cid]
    if a.N and len(ent) > a.N:
        idx = sorted(random.Random(f"{a.dataset}|{cid}|N{a.N}|{seed}").sample(range(len(ent)), a.N))
        ent = [ent[i] for i in idx]
    return ent


def select(ent, gq, seed, cid):
    k = min(a.K, len(ent))
    G = torch.stack([m.g[0] for _, m in ent])
    if a.scheme == "rand":
        r = random.Random(f"{a.dataset}|{cid}|{a.K}|{seed}")
        ids = sorted(r.sample([i for i, _ in ent], k)) if len(ent) > k else [i for i, _ in ent]
        pos = {i: n for n, (i, _) in enumerate(ent)}
        return [pos[i] for i in ids]
    sim = (G @ gq).tolist() if gq is not None else None
    if a.scheme == "retr":
        return list(np.argsort(sim)[::-1][:k])
    GG = (G @ G.t()).cpu().numpy()
    if a.scheme == "div":
        sel = [int(np.argmax(GG.mean(1)))]
        while len(sel) < k:
            sel.append(int(np.argmin(GG[:, sel].max(1) + np.isin(np.arange(len(ent)), sel) * 9)))
        return sel
    sel = []
    while len(sel) < k:   # mmr
        sc = [(0.7 * sim[i] - 0.3 * (GG[i, sel].max() if sel else 0)) if i not in sel else -9 for i in range(len(ent))]
        sel.append(int(np.argmax(sc)))
    return sel


# fixed (query independent) selections are cached per (cid, seed)
fixed = {}
texts = {cid: S.encode_text(n) for cid, n in cats}
visual = S.encode_text("visual")


def candidates(o, H, W, Rw, Ru, region=None, return_m72=False):
    """region=(x0,y0,x1,y1) in normalised canvas coords: keep dets whose box centre is inside; crop masks/boxes to it."""
    sc = o["scores"]; boxes = o["boxes"]; ml = o["mask_logits"]
    if region is not None:
        x0, y0, x1, y1 = region
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        ok = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        sc = torch.where(ok, sc, torch.zeros_like(sc))
        gh, gw = ml.shape[-2:]
        ml = ml[:, int(round(y0 * gh)):int(round(y1 * gh)), int(round(x0 * gw)):int(round(x1 * gw))]
        boxes = ((boxes - torch.tensor([x0, y0, x0, y0], device=boxes.device)) /
                 torch.tensor([x1 - x0, y1 - y0, x1 - x0, y1 - y0], device=boxes.device)).clamp(0, 1)
    order = torch.argsort(sc, descending=True)
    keep = [int(j) for r, j in enumerate(order[:keep_max]) if (r < a.keep_top or sc[j] >= a.keep_floor) and sc[j] > 0]
    if not keep:
        return ([], None) if return_m72 else []
    ml = ml[keep]
    m72 = (F.interpolate(ml[None], size=(72, 72), mode="bilinear", align_corners=False)[0] > 0).float()
    if Rw is not None:
        ar = m72.sum((1, 2)).clamp_min(1)
        Aw, Au = ((m72 * Rw).sum((1, 2)) / ar).tolist(), ((m72 * Ru).sum((1, 2)) / ar).tolist()
    else:
        Aw = Au = [None] * len(keep)
    full = F.interpolate(ml[None], size=(H, W), mode="bilinear", align_corners=False)[0] > 0
    areas = full.sum((1, 2)).tolist(); rles = gpu_rle.encode(full)
    bx = (boxes[keep] * torch.tensor([W, H, W, H], device=sc.device)).tolist()
    out = [{"score": float(sc[j]), "box": bx[n], "rle": rles[n], "Aw": Aw[n], "Au": Au[n], "area": int(areas[n])}
           for n, j in enumerate(keep)]
    return (out, m72) if return_m72 else out


def mosaic(sup_img, sup_boxes, q_img):
    """support | query side by side at the query's height; returns canvas, query region (normalised), support boxes in canvas px."""
    Hq, Wq = q_img.height, q_img.width
    s = Hq / sup_img.height
    sw = max(1, round(sup_img.width * s))
    cv = Image.new("RGB", (sw + Wq, Hq)); cv.paste(sup_img.resize((sw, Hq), Image.BILINEAR), (0, 0)); cv.paste(q_img, (sw, 0))
    bx = [[b[0] * s, b[1] * s, b[2] * s, b[3] * s] for b in sup_boxes]
    return cv, (sw / (sw + Wq), 0.0, 1.0, 1.0), bx


def mosaic_crop(sup_img, sup_boxes_, q_img, frac=1 / 3, pad=1.0, min_side=128):
    """Crop the support around its exemplar boxes (union box expanded by `pad` x its size on each side, >= min_side px),
    fit it into a strip of width frac*Wq (height <= Hq), place it left of the query. Returns canvas, query region
    (normalised) and the exemplar boxes in canvas pixels."""
    Hq, Wq = q_img.height, q_img.width
    b = np.array(sup_boxes_, dtype=np.float64)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    w, h = max(x1 - x0, min_side / (1 + 2 * pad)), max(y1 - y0, min_side / (1 + 2 * pad))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = w * (1 + 2 * pad), h * (1 + 2 * pad)
    X0, Y0 = max(0, cx - cw / 2), max(0, cy - ch / 2)
    X1, Y1 = min(sup_img.width, cx + cw / 2), min(sup_img.height, cy + ch / 2)
    crop = sup_img.crop((int(X0), int(Y0), int(np.ceil(X1)), int(np.ceil(Y1))))
    sw = max(1, int(round(frac * Wq)))
    s = min(sw / crop.width, Hq / crop.height)
    nw, nh = max(1, round(crop.width * s)), max(1, round(crop.height * s))
    cv = Image.new("RGB", (sw + Wq, Hq), (0, 0, 0))
    oy = (Hq - nh) // 2
    cv.paste(crop.resize((nw, nh), Image.BILINEAR), (0, oy)); cv.paste(q_img, (sw, 0))
    bx = [[(bb[0] - int(X0)) * s, (bb[1] - int(Y0)) * s + oy, (bb[2] - int(X0)) * s, (bb[3] - int(Y0)) * s + oy] for bb in sup_boxes_]
    return cv, (sw / (sw + Wq), 0.0, 1.0, 1.0), bx


def sup_boxes(iid, cid, E=3):
    anns = sorted([x for x in pool["annotations"] if x["image_id"] == iid and x["category_id"] == cid], key=lambda x: -x["area"])[:E]
    return [[x["bbox"][0], x["bbox"][1], x["bbox"][0] + max(x["bbox"][2], 1), x["bbox"][1] + max(x["bbox"][3], 1)] for x in anns]


results, timing = [], []
for qi, im in enumerate(queries):
    tq = time.time()
    H, W = im["height"], im["width"]
    qimg = Image.open(f"{D}/images/{im['file_name']}").convert("RGB")
    enc = S.encode_image(qimg)
    rec = {"image_id": im["id"], "file_name": im["file_name"], "classes": {}}
    for cid, _ in cats:
        T = texts[cid]
        tc, tm72 = candidates(S.decode(enc, *T), H, W, None, None, return_m72=True)
        crec = {"T": {"cands": tc}, "seeds": {}}
        for s in a.seeds:
            ent = pool_subset(cid, s)
            if a.scheme in ("rand", "div"):
                key = (cid, s)
                if key not in fixed:
                    fixed[key] = select(ent, None, s, cid)
                sel = fixed[key]
            else:
                sel = select(ent, enc.gfeat, s, cid)
            mem = stack([ent[i] for i in sel])
            alpha = route(mem, enc.gfeat); uni = torch.full_like(alpha, 1.0 / len(alpha))
            Rw, Ru = response(mem, enc.feat, alpha, "knn"), response(mem, enc.feat, uni, "knn")
            srec = {"alpha": alpha.tolist(), "support_ids": [int(ent[i][0]) for i in sel],
                    "Rw": Rw.half().cpu().numpy(), "Rmax": float(Rw.max())}
            if tc:
                ar = tm72.sum((1, 2)).clamp_min(1)
                srec["T_aff"] = ((tm72 * Rw).sum((1, 2)) / ar).tolist(); srec["T_affu"] = ((tm72 * Ru).sum((1, 2)) / ar).tolist()
            else:
                srec["T_aff"], srec["T_affu"] = [], []
            for cfg in a.configs:
                if cfg in ("E1", "TE1"):
                    srec[cfg] = {"per_support": []}
                    for tk in mem.tokens:
                        o = S.decode(enc, *cat_prompts([T if cfg == "TE1" else visual, tk]))
                        srec[cfg]["per_support"].append({"cands": candidates(o, H, W, Rw, Ru), "presence": o["presence"]})
                elif cfg in ("M1", "TM1", "MC1"):
                    srec[cfg] = {"per_support": []}
                    for i in sel:
                        iid = ent[i][0]
                        simg = Image.open(f"{D}/images/{pim[iid]['file_name']}").convert("RGB")
                        cv, reg, bx = (mosaic_crop if cfg == "MC1" else mosaic)(simg, sup_boxes(iid, cid), qimg)
                        ec = S.encode_image(cv)
                        o = S.decode(ec, *cat_prompts([T if cfg == "TM1" else visual, S.geometry_tokens(ec, bx, coord_mode="native")]))
                        srec[cfg]["per_support"].append({"cands": candidates(o, H, W, Rw, Ru, region=reg), "presence": o["presence"]})
            crec["seeds"][s] = srec
        rec["classes"][cid] = crec
    results.append(rec)
    torch.cuda.synchronize(); timing.append(time.time() - tq)
    if qi % 20 == 0:
        print(f"[{qi+1}/{len(queries)}] {np.mean(timing):.2f}s/img elapsed {time.time()-t0:.0f}s", flush=True)
draws = {}
with open(f"{a.out}/raw_{tag}.pkl", "wb") as f:
    pickle.dump({"args": vars(a), "results": results, "draws": draws}, f)
print("DONE", len(results), f"{time.time()-t0:.0f}s", flush=True)
