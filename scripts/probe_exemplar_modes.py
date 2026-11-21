"""Development probe on Kvasir *support-pool-candidate* images only (first 60 ids in sorted order; these are
NOT used as validation/test queries once manifests are frozen -- see make_manifests.py which excludes them).
Compares geometry-token coordinate modes for cross-image exemplars: top-1 box IoU vs GT."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys, json
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch
from PIL import Image
from torchvision.ops import box_iou
from coda.sam3core import FrozenSAM3, cat_prompts

R = f"{PROJECT_ROOT}/data/raw/kvasir/Kvasir-SEG"
bb = json.load(open(f"{R}/kavsir_bboxes.json"))
ids = sorted(bb)[:60]
S = FrozenSAM3()
encs = {i: S.encode_image(Image.open(f"{R}/images/{i}.jpg")) for i in ids}
def gtb(i): return [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in bb[i]["bbox"]]
tt = S.encode_text("polyp"); vt = S.encode_text("visual")
res = {}
for q_i, q in enumerate(ids[10:60]):
    sup = ids[q_i % 10]
    H, W = encs[q].orig_hw
    G = torch.tensor(gtb(q), dtype=torch.float32)
    cfgs = {"text": [tt]}
    for mode in ["native", "nopos", "center"]:
        ex = S.geometry_tokens(encs[sup], gtb(sup)[:1], coord_mode=mode)
        cfgs[f"text+ex_{mode}"] = [tt, ex]
        cfgs[f"visual+ex_{mode}"] = [vt, ex]
    for name, parts in cfgs.items():
        o = S.decode(encs[q], *cat_prompts(parts))
        j = int(o["scores"].argmax())
        pb = (o["boxes"][j].cpu() * torch.tensor([W, H, W, H])).reshape(1, 4)
        iou = float(box_iou(pb, G).max())
        res.setdefault(name, []).append((iou, float(o["scores"][j])))
for k, v in res.items():
    a = np.array(v); print(f"{k:22s} top1 IoU mean {a[:,0].mean():.3f}  IoU>0.5 {np.mean(a[:,0]>0.5):.2f}  score {a[:,1].mean():.3f}")
