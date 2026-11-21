"""Val-split design probe: quality of support->query response maps (no SAM3 decoding).
Variants: feature source {sam3 ViT 1024-d, sam3 FPN 72x72 256-d, DINOv2-base (external frozen)} x
          contrast {proto: mean FG proto - mean BG proto, knn: max_FG patch sim - max_BG patch sim}.
Metrics on query images that contain the class: pixel AUROC on the 72x72 grid, and whether the arg-max cell is inside GT.
Uses val GT only for *measuring* the design choice (val is the selection split)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys, json, argparse
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mu
from sklearn.metrics import roc_auc_score
from coda.sam3core import FrozenSAM3
from coda.support import draw_supports, soft_grid
from transformers import AutoModel

P = argparse.ArgumentParser(); P.add_argument("--dataset"); P.add_argument("--nq", type=int, default=80)
a = P.parse_args()
D = f"{PROJECT_ROOT}/data/proc/{a.dataset}"
pool = json.load(open(f"{D}/pool.json")); val = json.load(open(f"{D}/val.json"))
S = FrozenSAM3()
dino = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@torch.no_grad()
def feats(path):
    img = Image.open(path).convert("RGB")
    e = S.encode_image(img)
    vit = e.feat
    fpn = F.normalize(e.vision.fpn_hidden_states[2][0].float(), dim=0)
    x = torch.from_numpy(np.array(img.resize((1008, 1008), Image.BICUBIC))).permute(2, 0, 1).float() / 255
    x = ((x - MEAN) / STD)[None].cuda().half()
    h = dino(pixel_values=x).last_hidden_state[0, 1:]           # 72x72 patches of 14px
    d = F.normalize(h.float().t().reshape(-1, 72, 72), dim=0)
    return {"vit": vit, "fpn": fpn, "dino": d}


def gtmask(coco, iid, cid):
    rl = [x["segmentation"] for x in coco["annotations"] if x["image_id"] == iid and x["category_id"] == cid]
    return mu.decode(mu.merge(rl)).astype(bool) if rl else None


cats = [c["id"] for c in val["categories"]]
ims = {i["id"]: i for i in val["images"]}
pims = {i["id"]: i for i in pool["images"]}
qcache = {}
out = {}
for cid in cats:
    qids = sorted({x["image_id"] for x in val["annotations"] if x["category_id"] == cid})[: a.nq if len(cats) == 1 else 15]
    for seed in [0, 1, 2]:
        sids = draw_supports(pool, cid, 3, seed, a.dataset)
        sf = {s: feats(f"{D}/images/{pims[s]['file_name']}") for s in sids}
        sw = {s: soft_grid(gtmask(pool, s, cid)) for s in sids}
        for q in qids:
            if q not in qcache:
                qcache[q] = feats(f"{D}/images/{ims[q]['file_name']}")
            gm = soft_grid(gtmask(val, q, cid)).cpu().numpy() > 0.5
            if not gm.any() or gm.all():
                continue
            for src in ["vit", "fpn", "dino"]:
                fq = qcache[q][src].reshape(qcache[q][src].shape[0], -1)
                Rp, Rk = 0, 0
                for s in sids:
                    f = sf[s][src].reshape(fq.shape[0], -1); w = sw[s].reshape(-1)
                    v = F.normalize((f * w).sum(1), dim=0); b = F.normalize((f * (1 - w)).sum(1), dim=0)
                    Rp = Rp + (v @ fq - b @ fq) / 3
                    fgp = f[:, w >= 0.5] if (w >= 0.5).any() else f[:, w.argmax()][:, None]
                    bgp = f[:, w == 0]
                    Rk = Rk + ((fgp.t() @ fq).max(0).values - (bgp.t() @ fq).max(0).values) / 3
                for name, R in [("proto", Rp), ("knn", Rk)]:
                    r = R.cpu().numpy()
                    k = f"{src}-{name}"
                    out.setdefault(k, {"auc": [], "peak": []})
                    out[k]["auc"].append(roc_auc_score(gm.ravel(), r)); out[k]["peak"].append(gm.ravel()[r.argmax()])
for k, v in out.items():
    print(f"{a.dataset} {k:12s} AUROC {np.mean(v['auc']):.3f}  peak-in-GT {np.mean(v['peak']):.3f}  n={len(v['auc'])}")
