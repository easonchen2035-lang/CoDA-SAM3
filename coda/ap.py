"""COCO-identical AP@IoU0.5 computed from per-image match records (memory-light, mergeable across shards).

Replicates pycocotools COCOeval (iouThrs=[0.5], maxDets=100, areaRng 'all', no crowd GT):
  per image & category: dets sorted by score (desc, stable mergesort) truncated to 100; GT order as in annotations;
  each det (in order) takes the not-yet-matched GT with the highest IoU >= 0.5 (ties -> later GT, as in COCO's loop
  `if ious[dind,gind] < iou: continue` which keeps strictly-greater only... see match()).
  accumulate: concat over images, stable sort by -score, cumulative TP/FP, 101-point interpolated precision.
Verified equal to pycocotools on Kvasir val (scripts/check_ap.py)."""
import numpy as np
from pycocotools import mask as mu


def match(det_scores, ious, n_gt):
    """ious: [n_det, n_gt] (dets already sorted desc & truncated). Returns tp bool array (COCO logic)."""
    tp = np.zeros(len(det_scores), bool)
    used = np.zeros(n_gt, bool)
    for d in range(len(det_scores)):
        iou = min(0.5, 1 - 1e-10)
        m = -1
        for g in range(n_gt):
            if used[g]:
                continue
            if ious[d, g] < iou:
                continue
            iou = ious[d, g]; m = g
        if m >= 0:
            used[m] = True; tp[d] = True
    return tp


def image_records(dets, gts, kind):
    """dets: list of dict(s, box[x0,y0,x1,y1], rle); gts: list of COCO anns (one image, one category).
    Returns (scores, tp) for this image (top-100 by score)."""
    if not dets:
        return np.zeros(0), np.zeros(0, bool)
    order = np.argsort([-d["s"] for d in dets], kind="mergesort")[:100]
    dets = [dets[i] for i in order]
    sc = np.array([d["s"] for d in dets])
    if not gts:
        return sc, np.zeros(len(dets), bool)
    if kind == "bbox":
        db = [[d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]] for d in dets]
        ious = np.asarray(mu.iou(db, [g["bbox"] for g in gts], [0] * len(gts)))
    else:
        ious = np.asarray(mu.iou([d["rle"] for d in dets], [g["segmentation"] for g in gts], [0] * len(gts)))
    return sc, match(sc, ious.reshape(len(dets), len(gts)), len(gts))


def ap_from_records(scores, tps, n_gt):
    """COCO accumulate for one category (101 recall points)."""
    if n_gt == 0:
        return None
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="mergesort")
    tp = tps[o].astype(np.float64); fp = 1 - tp
    tpc, fpc = np.cumsum(tp), np.cumsum(fp)
    rc = tpc / n_gt
    pr = tpc / (fpc + tpc + np.spacing(1))
    pr = pr.tolist()
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    rs = np.linspace(0, 1, 101)
    inds = np.searchsorted(rc, rs, side="left")
    q = np.zeros(101)
    for ri, pi in enumerate(inds):
        if pi < len(pr):
            q[ri] = pr[pi]
    return float(q.mean())
