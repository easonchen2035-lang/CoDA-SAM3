"""CPU stage: method assembly from raw candidates, metrics.

Method spec (string):
  "src:<CFG>"                         single prompt configuration, SAM3 score, mask-IoU NMS
  "fuse:<CFG1>+<CFG2>..|lam=<l>|aff=<w|u>|nms=<t>"
                                      union of candidates from several configurations; refined score
                                      s_hat = s + lam * A  (A = mean support response inside the mask,
                                      weighted (w) or uniform (u) routing); greedy mask-IoU NMS.
  "fm"                                pure feature matching (no SAM3 decoder): CCs of upsampled response map
Semantic prediction for class c = union of final instances with score >= tau.

Metrics:
  AP50 (box) and AP50 (mask): pycocotools COCOeval, IoU thr 0.5, maxDets 100, area 'all'; per-class AP then
     mean over classes that have GT in the split (macro; = COCO convention). Scores = final (refined) scores.
  Semantic, binary datasets (1 class): per-image foreground IoU and Dice, averaged over images
     (image with empty GT and empty prediction counts as 1.0).
  Semantic, multi-class: per class c, intersection/union/|P|/|G| accumulated over all images of the split
     (images without class c contribute false positives), IoU_c, Dice_c; mIoU / mDice = mean over classes.
"""
from __future__ import annotations

import copy
import re
import io
import contextlib
import json
import pickle
import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_raw(paths):
    res, draws, args = [], {}, None
    for p in paths:
        d = pickle.load(open(p, "rb"))
        res += d["results"]; draws.update(d["draws"]); args = d["args"]
    return res, draws, args


def parse(spec):
    if spec == "fm":
        return {"kind": "fm"}
    head, *opts = spec.split("|")
    kind, srcs = head.split(":")
    o = {"kind": kind, "srcs": srcs.split("+"), "lam": 0.0, "aff": "w", "nms": 0.5, "fm_thr": 0.5,
         "vote": "a", "extra": "", "viou": 0.5}
    for x in opts:
        k, v = x.split("=")
        o[k] = float(v) if k in ("lam", "nms", "fm_thr", "viou") else v
    o["extra"] = [e for e in o["extra"].split("+") if e]
    return o


def consensus_vote(per_support, weights, viou=0.5, mask_vote=False):
    """Context-consensus over per-support decodes (one SAM3 decode per support exemplar).
    Candidates of all supports are clustered greedily (descending score, mask IoU > viou);
    a cluster's score = sum_i w_i * (max score among its members from support i, 0 if none);
    its mask/box are those of the highest-scoring member. w = routing alpha ('a') or uniform 1/K ('u')."""
    allc = [(c["score"], i, c) for i, ps in enumerate(per_support) for c in ps["cands"]]
    if not allc:
        return []
    allc.sort(key=lambda t: -t[0])
    rles = [c["rle"] for _, _, c in allc]
    bx = _boxes(rles)
    areas = np.array([mu.area(r) for r in rles], dtype=np.float64)
    used = np.zeros(len(allc), bool)
    out = []
    for j in range(len(allc)):
        if used[j]:
            continue
        row = _row_iou(rles, bx, j, ~used, areas, viou); row[j] = 1.0
        mem = np.nonzero((row > viou) & ~used)[0]
        used[mem] = True
        best, bestm = {}, {}
        for m in mem:
            sc, i, _ = allc[m]
            if sc > best.get(i, -1):
                best[i], bestm[i] = sc, m
        c = allc[j][2]
        if mask_vote and len(best) > 1:
            # boundary consensus: pixel-wise weighted vote of each support's best member mask (present supports)
            ws = np.array([weights[i] for i in best]); ws = ws / ws.sum()
            acc = sum(w * mu.decode(allc[bestm[i]][2]["rle"]).astype(np.float32) for w, i in zip(ws, best))
            fm = acc >= 0.5
            if fm.any():
                rl = mu.encode(np.asfortranarray(fm.astype(np.uint8))); rl["counts"] = rl["counts"].decode()
                x, y, bw, bh = mu.toBbox(rl).tolist()
                c = {**c, "rle": rl, "box": [x, y, x + bw, y + bh]}
        out.append({"score": float(sum(weights[i] * v for i, v in best.items())), "c": c, "support": len(best)})
    return out


def _boxes(rles):
    b = np.array(mu.toBbox(rles), dtype=np.float64).reshape(-1, 4)
    b[:, 2:] += b[:, :2]
    return b


def _overlap(b, i, areas=None, thr=0.0):
    """indices j that can have mask IoU(i,j) > thr. Exact necessary condition:
    IoU > t  =>  |m_i & m_j| > t * max(|m_i|, |m_j|)  and  |m_i & m_j| <= box-intersection area."""
    ix = np.clip(np.minimum(b[i, 2], b[:, 2]) - np.maximum(b[i, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[i, 3], b[:, 3]) - np.maximum(b[i, 1], b[:, 1]), 0, None)
    inter = ix * iy
    if areas is None:
        return np.nonzero(inter > 0)[0]
    return np.nonzero((inter > 0) & (inter >= thr * np.maximum(areas[i], areas)))[0]


def _row_iou(rles, b, i, cand_mask, areas=None, thr=0.0):
    """mask IoU of i vs. all j; 0 where IoU <= thr is guaranteed by the bound or cand_mask is False."""
    out = np.zeros(len(rles))
    js = _overlap(b, i, areas, thr)
    js = js[cand_mask[js]]
    if len(js):
        out[js] = np.asarray(mu.iou([rles[i]], [rles[j] for j in js], [0] * len(js))).reshape(-1)
    return out


def _nms(cands, thr, max_keep=100):
    """Greedy mask-IoU NMS (identical result to the all-pairs version; IoU only evaluated for box-overlapping pairs)."""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: -c["s"])
    rles = [c["rle"] for c in cands]
    b = _boxes(rles)
    areas = np.array([mu.area(r) for r in rles], dtype=np.float64)
    keep, sup = [], np.zeros(len(cands), bool)
    for i in range(len(cands)):
        if sup[i]:
            continue
        keep.append(i)
        if len(keep) >= max_keep:
            break
        sup |= _row_iou(rles, b, i, ~sup, areas, thr) > thr
    return [cands[i] for i in keep]


def fm_instances(R, H, W, rel_thr=0.5):
    """Feature-matching baseline: bilinear-upsample R (72x72) to (H,W); components of R > rel_thr*max R."""
    import torch, torch.nn.functional as F
    from scipy import ndimage
    r = R.astype(np.float32)
    if r.max() <= 0:
        return []
    up = F.interpolate(torch.from_numpy(r)[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    lab, n = ndimage.label(up > rel_thr * r.max(), structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < 16:
            continue
        ys, xs = np.nonzero(m)
        rl = mu.encode(np.asfortranarray(m.astype(np.uint8))); rl["counts"] = rl["counts"].decode()
        out.append({"s": float(up[m].mean()), "box": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], "rle": rl})
    return out


GATE = re.compile(r"^d?gate\(([0-9.]+)\)\[(.+)\]\[(.+)\]$")


def text_reliability(draws, pool_rel, seed, cid, support_ids=None):
    """mean support-image text confident recall (GT matched at IoU>=0.5 by text detections with SAM3 score>=0.5)
    over the K support images of the draw (seed, cid); support GT only."""
    ids = support_ids if support_ids is not None else draws[f"{seed}|{cid}"]
    return float(np.mean([pool_rel[f"{i}|{cid}"]["conf_recall"] for i in ids]))


def assemble(rec_c, seed, spec, H, W, max_det=100, cid=None, ctx=None):
    if isinstance(spec, str) and (spec.startswith("gate(") or spec.startswith("dgate(")):
        thr, a, b = GATE.match(spec).groups()
        if spec.startswith("dgate("):   # domain level: mean over all classes' support draws of this seed
            r = float(np.mean([text_reliability(ctx["draws"], ctx["pool_rel"], seed, c) for c in ctx["cats"]]))
        else:
            r = text_reliability(ctx["draws"], ctx["pool_rel"], seed, cid, rec_c["seeds"][seed].get("support_ids"))
        return assemble(rec_c, seed, a if r >= float(thr) else b, H, W, max_det, cid, ctx)
    """Final instances [{s, box, rle}] for one (image, class, seed)."""
    o = parse(spec) if isinstance(spec, str) else spec
    sr = rec_c["seeds"][seed] if seed is not None else None
    if o["kind"] == "fm":
        return fm_instances(sr["Rw"], H, W)[:max_det]
    cands = []
    if o["kind"] == "jvote":
        # joint consensus: per-support decodes of ALL listed sources vote together (each source's weights sum to 1/n_src)
        ps, w = [], []
        for src in o["srcs"]:
            p = sr[src]["per_support"]; K = len(p)
            ww = sr["alpha"] if o["vote"] == "a" else [1.0 / K] * K
            ps += p; w += [x / len(o["srcs"]) for x in ww]
        for v in consensus_vote(ps, w, o["viou"], o.get("mv", "0") == "1"):
            c = v["c"]; A = c["Aw"] if o["aff"] == "w" else c["Au"]
            cands.append({"s": v["score"] + o["lam"] * A, "box": c["box"], "rle": c["rle"]})
        o = {**o, "srcs": o["extra"]}
    if o["kind"] in ("pool", "top1"):
        # ablations of consensus voting: 'pool' = union of all per-support decodes + NMS (no voting);
        # 'top1' = only the decode of the highest-alpha support (single routed exemplar)
        j = int(np.argmax(sr["alpha"]))
        for src in o["srcs"]:
            ps = sr[src]["per_support"]
            for i, p in enumerate(ps):
                if o["kind"] == "top1" and i != j:
                    continue
                for c in p["cands"]:
                    A = c["Aw"] if o["aff"] == "w" else c["Au"]
                    cands.append({"s": c["score"] + o["lam"] * A, "box": c["box"], "rle": c["rle"]})
        return _nms(cands, o["nms"])[:max_det]
    if o["kind"] == "vote":
        for src in o["srcs"]:
            ps = sr[src]["per_support"]
            K = len(ps)
            w = sr["alpha"] if o["vote"] == "a" else [1.0 / K] * K
            for v in consensus_vote(ps, w, o["viou"], o.get("mv", "0") == "1"):
                c = v["c"]; A = c["Aw"] if o["aff"] == "w" else c["Au"]
                cands.append({"s": v["score"] + o["lam"] * A, "box": c["box"], "rle": c["rle"]})
        o = {**o, "srcs": o["extra"]}
    for src in o["srcs"]:
        if src == "T":
            aff = sr["T_aff"] if o["aff"] == "w" else sr["T_affu"]
            for c, A in zip(rec_c["T"]["cands"], aff):
                cands.append({"s": c["score"] + o["lam"] * A, "box": c["box"], "rle": c["rle"]})
        else:
            for c in sr[src]["cands"]:
                A = c["Aw"] if o["aff"] == "w" else c["Au"]
                cands.append({"s": c["score"] + o["lam"] * A, "box": c["box"], "rle": c["rle"]})
    return _nms(cands, o["nms"])[:max_det]


def run_method(results, gt_json, spec, seed, taus=(0.5,)):
    """Returns dict with AP50 box/mask and semantic metrics for each tau (list)."""
    gt = gt_json
    cats = [c["id"] for c in gt["categories"]]
    imgs = {im["id"]: im for im in gt["images"]}
    gtm = {}
    for a in gt["annotations"]:
        gtm.setdefault((a["image_id"], a["category_id"]), []).append(a["segmentation"])
    dets_box, dets_seg = [], []
    sem = {t: {} for t in taus}
    for rec in results:
        iid = rec["image_id"]; im = imgs[iid]; H, W = im["height"], im["width"]
        for cid in cats:
            rc = rec["classes"][cid] if cid in rec["classes"] else rec["classes"][str(cid)]
            inst = assemble(rc, seed, spec, H, W)
            for d in inst:
                x0, y0, x1, y1 = d["box"]
                dets_box.append({"image_id": iid, "category_id": cid, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": d["s"]})
                dets_seg.append({"image_id": iid, "category_id": cid, "segmentation": d["rle"], "score": d["s"]})
            g = gtm.get((iid, cid), [])
            G = mu.merge(g) if g else None
            gA = int(mu.area(G)) if G else 0
            for t in taus:
                ps = [d["rle"] for d in inst if d["s"] >= t]
                Pm = mu.merge(ps) if ps else None
                pA = int(mu.area(Pm)) if Pm else 0
                I = int(mu.area(mu.merge([Pm, G], intersect=True))) if (Pm and G) else 0
                sem[t][(iid, cid)] = (I, pA, gA)
    out = {"ap50_box": coco_ap50(gt, dets_box, "bbox"), "ap50_mask": coco_ap50(gt, dets_seg, "segm"), "sem": {}}
    for t in taus:
        out["sem"][t] = semantic(sem[t], cats)
    return out


def semantic(table, cats):
    if len(cats) == 1:
        ious, dices = [], []
        for (iid, cid), (I, P, G) in table.items():
            U = P + G - I
            ious.append(1.0 if U == 0 else I / U)
            dices.append(1.0 if P + G == 0 else 2 * I / (P + G))
        return {"mIoU": float(np.mean(ious)), "mDice": float(np.mean(dices)), "per_class": {}}
    acc = {c: np.zeros(3) for c in cats}
    for (iid, cid), v in table.items():
        acc[cid] += np.array(v, dtype=np.float64)
    pc = {}
    for c, (I, P, G) in acc.items():
        if G == 0:
            continue
        pc[int(c)] = {"IoU": I / (P + G - I), "Dice": 2 * I / (P + G)}
    return {"mIoU": float(np.mean([v["IoU"] for v in pc.values()])),
            "mDice": float(np.mean([v["Dice"] for v in pc.values()])), "per_class": pc}


def coco_ap50(gt, dets, iou_type):
    if not dets:
        return {"AP50": 0.0, "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = copy.deepcopy(gt); cg.createIndex()
        cd = cg.loadRes(copy.deepcopy(dets))
        E = COCOeval(cg, cd, iou_type)
        E.params.iouThrs = np.array([0.5]); E.params.maxDets = [1, 10, 100]
        E.evaluate(); E.accumulate()
    prec = E.eval["precision"][0, :, :, 0, -1]          # [R, K]
    pc = {}
    for k, cid in enumerate(E.params.catIds):
        p = prec[:, k]
        if (p > -1).any() and any(a["category_id"] == cid for a in gt["annotations"]):
            pc[int(cid)] = float(p[p > -1].mean())
    return {"AP50": float(np.mean(list(pc.values()))) if pc else 0.0, "per_class": pc}
