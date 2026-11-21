"""Support-side text reliability: run text-only SAM3 on every POOL image for every class it contains and record
R-precision of the text prompt on that image: (#GT instances matched at mask IoU>=0.5 by the top-|GT| text detections)/|GT|.
At inference, a support draw's text reliability for class c = mean of this value over its K support images
(only labelled support images are used; no query/val/test GT)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, json, sys
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mu
from coda.sam3core import FrozenSAM3
from coda import gpu_rle

P = argparse.ArgumentParser(); P.add_argument("--dataset", required=True); a = P.parse_args()
D = f"{PROJECT_ROOT}/data/proc/{a.dataset}"
pool = json.load(open(f"{D}/pool.json"))
S = FrozenSAM3(feat_source="fpn72")
names = {c["id"]: c["name"] for c in pool["categories"]}
by = {}
for x in pool["annotations"]:
    by.setdefault((x["image_id"], x["category_id"]), []).append(x)
out = {}
for im in pool["images"]:
    cids = sorted({c for (i, c) in by if i == im["id"]})
    if not cids:
        continue
    enc = S.encode_image(Image.open(f"{D}/images/{im['file_name']}"))
    H, W = im["height"], im["width"]
    for cid in cids:
        gts = by[(im["id"], cid)]
        o = S.decode(enc, *S.encode_text(names[cid]))
        n = len(gts)
        idx = torch.argsort(o["scores"], descending=True)[:n]
        full = F.interpolate(o["mask_logits"][idx][None], size=(H, W), mode="bilinear", align_corners=False)[0] > 0
        rl = gpu_rle.encode(full)
        iou = np.asarray(mu.iou(rl, [g["segmentation"] for g in gts], [0] * n)).reshape(len(rl), n)
        sc = o["scores"][idx].tolist()
        matched, conf_matched = set(), 0
        for d in range(len(rl)):
            j = int(np.argmax(np.where([k in matched for k in range(n)], -1, iou[d])))
            if iou[d, j] >= 0.5 and j not in matched:
                matched.add(j)
                conf_matched += sc[d] >= 0.5
        # rprec: ranking-only; conf_recall: GT matched by text detections that SAM3 itself scores >= 0.5
        out[f"{im['id']}|{cid}"] = {"rprec": len(matched) / n, "conf_recall": conf_matched / n, "n_gt": n,
                                   "top_scores": sc}
json.dump(out, open(f"{D}/pool_text_reliability.json", "w"))
print("DONE", a.dataset, len(out), "mean rprec", np.mean([v["rprec"] for v in out.values()]))
