"""PerSAM baseline (training-free; Zhang et al., "Personalize Segment Anything Model with One Shot", ICLR 2024),
re-implemented with HF `SamModel` (facebook/sam-vit-huge), following the official PerSAM inference:
  1. target embedding = masked average of the support SAM image embedding (64x64) under the support GT mask
     (here: average over the same K retrieved supports used by our method, all class-c instances)
  2. query location confidence map = cos(target, query embedding); positive point = argmax, negative point = argmin
  3. target-guided attention (attention_similarity = sigmoid-ed confidence map) + target-semantic prompting
     (target_embedding) in the SAM mask decoder
  4. cascaded post-refinement: (a) re-prompt with the best mask logits + point, (b) re-prompt with the box of (a)
PerSAM outputs ONE instance per image and class (its design); score = SAM predicted IoU of the final mask.
Record format = run_select records with config 'P' (per_support with one entry) so eval_fast can read it."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, json, os, pickle, sys, time
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mu
from transformers import SamModel, SamProcessor
from coda.sam3core import FrozenSAM3
from coda.support import build_memory
from coda import gpu_rle

P = argparse.ArgumentParser()
P.add_argument("--dataset", required=True); P.add_argument("--split", required=True)
P.add_argument("--K", type=int, default=3); P.add_argument("--shard", type=int, default=0)
P.add_argument("--nshards", type=int, default=1); P.add_argument("--limit", type=int, default=None)
P.add_argument("--out", required=True)
a = P.parse_args()
D = f"{PROJECT_ROOT}/data/proc/{a.dataset}"
pool = json.load(open(f"{D}/pool.json")); qco = json.load(open(f"{D}/{a.split}.json"))
cats = [c["id"] for c in qco["categories"]]
queries = (qco["images"][: a.limit] if a.limit else qco["images"])[a.shard::a.nshards]
os.makedirs(a.out, exist_ok=True)
tag = f"{a.split}_K{a.K}_s0_sh{a.shard}of{a.nshards}"
dev = "cuda"
sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()
proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
S3 = FrozenSAM3(feat_source="fpn72")   # only for the SAME retrieval descriptor as our method (fair support set)
pim = {im["id"]: im for im in pool["images"]}
cls_imgs = {c: sorted({x["image_id"] for x in pool["annotations"] if x["category_id"] == c}) for c in cats}


@torch.no_grad()
def sam_embed(img):
    inp = proc(images=img, return_tensors="pt").to(dev)
    emb = sam.get_image_embeddings(inp["pixel_values"])          # [1,256,64,64]
    return emb, inp


def mask_to_grid(m, inp):
    """GT mask (orig res) -> 64x64 weights consistent with SAM's resize-longest-1024 + pad preprocessing."""
    H, W = m.shape
    s = 1024 / max(H, W); h, w = round(H * s), round(W * s)
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(h, w), mode="bilinear")
    t = F.pad(t, (0, 1024 - w, 0, 1024 - h))
    return F.interpolate(t, size=(64, 64), mode="area")[0, 0].to(dev)


# global descriptors for retrieval (identical to run_select 'retr')
gdesc = {}
for iid in sorted(pim):
    gdesc[iid] = S3.encode_image(Image.open(f"{D}/images/{pim[iid]['file_name']}")).gfeat
targets = {}   # (cid, iid) -> target feature [256]


def target_feat(cid, iid):
    if (cid, iid) not in targets:
        img = Image.open(f"{D}/images/{pim[iid]['file_name']}").convert("RGB")
        emb, inp = sam_embed(img)
        m = np.zeros((img.height, img.width), bool)
        for x in pool["annotations"]:
            if x["image_id"] == iid and x["category_id"] == cid:
                m |= mu.decode(x["segmentation"]).astype(bool)
        w = mask_to_grid(m, inp)
        f = emb[0]
        targets[(cid, iid)] = ((f * w).sum((1, 2)) / w.sum().clamp_min(1e-6))
    return targets[(cid, iid)]


results = []
t0 = time.time()
for qi, im in enumerate(queries):
    H, W = im["height"], im["width"]
    qimg = Image.open(f"{D}/images/{im['file_name']}").convert("RGB")
    gq = S3.encode_image(qimg).gfeat
    emb, inp = sam_embed(qimg)
    rec = {"image_id": im["id"], "file_name": im["file_name"], "classes": {}}
    for cid in cats:
        ids = cls_imgs[cid]
        sims = torch.stack([gdesc[i] for i in ids]) @ gq
        sel = [ids[j] for j in torch.argsort(sims, descending=True)[: a.K].tolist()]
        tgt = torch.stack([target_feat(cid, i) for i in sel]).mean(0)          # [256]
        tgt_n = F.normalize(tgt, dim=0)
        fq = F.normalize(emb[0], dim=0)                                      # [256,64,64]
        sim = (tgt_n[:, None, None] * fq).sum(0)                              # [64,64]
        # map 64x64 grid -> original image coords
        s = 1024 / max(H, W)
        sim_up = F.interpolate(sim[None, None], size=(1024, 1024), mode="bilinear")[0, 0][: round(H * s), : round(W * s)]
        sim_o = F.interpolate(sim_up[None, None], size=(H, W), mode="bilinear")[0, 0]
        py, px = np.unravel_index(int(sim_o.argmax()), sim_o.shape); ny, nx = np.unravel_index(int(sim_o.argmin()), sim_o.shape)
        attn = ((sim - sim.mean()) / (sim.std() + 1e-6)).sigmoid().reshape(1, 1, 64 * 64)   # PerSAM: sim normalised, sigmoid
        pts = [[[[float(px), float(py)], [float(nx), float(ny)]]]]
        lab = [[[1, 0]]]
        pi = proc(images=qimg, input_points=pts, input_labels=lab, return_tensors="pt").to(dev)
        o = sam(image_embeddings=emb, input_points=pi["input_points"], input_labels=pi["input_labels"],
                multimask_output=False, attention_similarity=attn, target_embedding=tgt.reshape(1, 1, 1, 256))
        logits = o.pred_masks[0, 0]                                            # [1,256,256]
        # cascaded post-refinement 1: mask-logit prompt + points
        o2 = sam(image_embeddings=emb, input_points=pi["input_points"], input_labels=pi["input_labels"],
                 input_masks=logits[None], multimask_output=True)
        best = int(o2.iou_scores[0, 0].argmax())
        lg2 = o2.pred_masks[0, 0, best]
        up = proc.post_process_masks(o2.pred_masks[:, :, best:best + 1].cpu(), pi["original_sizes"].cpu(), pi["reshaped_input_sizes"].cpu())[0][0, 0]
        ys, xs = np.nonzero(up.numpy())
        if len(xs):
            box = [[[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]]
            pb = proc(images=qimg, input_points=pts, input_labels=lab, input_boxes=box, return_tensors="pt").to(dev)
            o3 = sam(image_embeddings=emb, input_points=pb["input_points"], input_labels=pb["input_labels"],
                     input_boxes=pb["input_boxes"], input_masks=lg2[None, None], multimask_output=True)
            b3 = int(o3.iou_scores[0, 0].argmax())
            fin = proc.post_process_masks(o3.pred_masks[:, :, b3:b3 + 1].cpu(), pb["original_sizes"].cpu(), pb["reshaped_input_sizes"].cpu())[0][0, 0]
            score = float(o3.iou_scores[0, 0, b3])
        else:
            fin, score = up, float(o2.iou_scores[0, 0, best])
        cands = []
        if fin.any():
            r = gpu_rle.encode(fin[None].to(dev).bool())[0]
            ys, xs = np.nonzero(fin.numpy())
            cands = [{"score": score, "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                      "rle": r, "Aw": 0.0, "Au": 0.0, "area": int(fin.sum())}]
        rec["classes"][cid] = {"T": {"cands": []}, "seeds": {0: {"alpha": [1.0], "support_ids": sel, "T_aff": [], "T_affu": [],
                                                                  "P": {"cands": cands}}}}
    results.append(rec)
    if qi % 20 == 0:
        print(f"[{qi+1}/{len(queries)}] elapsed {time.time()-t0:.0f}s", flush=True)
pickle.dump({"args": vars(a), "results": results, "draws": {}}, open(f"{a.out}/raw_{tag}.pkl", "wb"))
print("DONE", len(results), f"{time.time()-t0:.0f}s")
