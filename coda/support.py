"""Support memory (domain context) and query-conditioned routing for CoDA-SAM3 (re-implementation).

Representation per support entry i of class c (all from the *frozen* SAM3 image encoder):
  v_i  FG visual prototype : masked average pooling of the L2-normalised dense feature map f (ViT, 1024-d, 72x72)
                             with the union GT mask of class c (area-downsampled to 72x72 as soft weights)
  b_i  BG visual prototype : same with (1 - FG weight)
  g_i  global descriptor   : GAP of f over the whole support image (used ONLY for routing)
  s_i  structural stats    : per-instance [log area ratio, log aspect ratio] of class-c instances
  e_i  exemplar tokens     : SAM3 geometry-encoder tokens of up to E largest class-c instance boxes,
                             computed on the support image (coord_mode 'nopos', see sam3core)
Text is NOT concatenated with visual vectors: the class phrase is fed to SAM3's own text pathway.
All similarities are computed within one space (visual-visual).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mu


@dataclass
class ClassMemory:
    cat_id: int
    image_ids: list                 # pool image ids (support draw)
    v: torch.Tensor                 # [K, C]
    b: torch.Tensor                 # [K, C]
    g: torch.Tensor                 # [K, C]
    tokens: list                    # K x (tok [1,n,256], mask [1,n])
    struct: np.ndarray              # [n_inst, 2] (log area ratio, log aspect)
    n_inst: list = field(default_factory=list)
    fg_patches: list = field(default_factory=list)   # K x [C, n_fg] normalised support FG cells (knn response)
    bg_patches: list = field(default_factory=list)   # K x [C, n_bg]


def draw_supports(pool_coco, cat_id, K, seed, dataset):
    """K distinct pool images containing >=1 instance of cat_id; deterministic in (dataset, cat, K, seed)."""
    imgs = sorted({a["image_id"] for a in pool_coco["annotations"] if a["category_id"] == cat_id})
    rng = random.Random(f"{dataset}|{cat_id}|{K}|{seed}")
    if len(imgs) <= K:
        return imgs
    return sorted(rng.sample(imgs, K))


def masks_for(coco, image_id, cat_id):
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id and a["category_id"] == cat_id]
    ms = [mu.decode(a["segmentation"]).astype(bool) for a in anns]
    return anns, ms


def soft_grid(mask_bool, grid=72, device="cuda"):
    t = torch.from_numpy(mask_bool.astype(np.float32))[None, None].to(device)
    return F.interpolate(t, size=(grid, grid), mode="area")[0, 0]


def build_memory(S, pool_coco, pool_dir, cat_id, image_ids, enc_cache, E=3, coord_mode="nopos"):
    from PIL import Image
    vs, bs, gs, toks, struct, ninst, fgp, bgp = [], [], [], [], [], [], [], []
    for iid in image_ids:
        im = next(x for x in pool_coco["images"] if x["id"] == iid)
        if iid not in enc_cache:
            enc_cache[iid] = S.encode_image(Image.open(f"{pool_dir}/{im['file_name']}"))
        enc = enc_cache[iid]
        anns, ms = masks_for(pool_coco, iid, cat_id)
        union = np.any(np.stack(ms), 0)
        w = soft_grid(union, enc.feat.shape[-1])
        f = enc.feat
        v = (f * w).sum((1, 2)) / w.sum().clamp_min(1e-6)
        wb = 1 - w
        b = (f * wb).sum((1, 2)) / wb.sum().clamp_min(1e-6)
        vs.append(F.normalize(v, dim=0)); bs.append(F.normalize(b, dim=0)); gs.append(enc.gfeat)
        fl, wl = f.reshape(f.shape[0], -1), w.reshape(-1)
        fgp.append(fl[:, wl >= 0.5] if (wl >= 0.5).any() else fl[:, wl.argmax()][:, None])
        bgp.append(fl[:, wl == 0])
        H, W = im["height"], im["width"]
        order = np.argsort([-a["area"] for a in anns])[:E]
        boxes = []
        for j in order:
            x, y, bw, bh = anns[j]["bbox"]
            boxes.append([x, y, x + max(bw, 1), y + max(bh, 1)])
        toks.append(S.geometry_tokens(enc, boxes, coord_mode=coord_mode))
        for a in anns:
            x, y, bw, bh = a["bbox"]
            struct.append([math.log(max(a["area"], 1) / (H * W)), math.log(max(bw, 1) / max(bh, 1))])
        ninst.append(len(anns))
    return ClassMemory(cat_id, list(image_ids), torch.stack(vs), torch.stack(bs), torch.stack(gs), toks,
                       np.array(struct, dtype=np.float32), ninst, fgp, bgp)


def route(mem: ClassMemory, gq: torch.Tensor, temp=0.05):
    """alpha_i = softmax(cos(g_q, g_i)/T): visual-only routing weights."""
    return torch.softmax((mem.g @ gq) / temp, 0)


def response(mem: ClassMemory, fq: torch.Tensor, alpha: torch.Tensor, mode="proto"):
    """Support-conditioned response on the 72x72 grid (fq: [C,h,w] L2-normalised).
    proto: R(u) = sum_i alpha_i [cos(f(u), v_i) - cos(f(u), b_i)]
    knn  : R(u) = sum_i alpha_i [max_{p in FG_i} cos(f(u), p) - max_{p in BG_i} cos(f(u), p)]
           (FG_i: support cells with FG weight >= 0.5, or the max-weight cell; BG_i: cells with weight 0)"""
    C, h, w = fq.shape
    ff = fq.reshape(C, -1)
    if mode == "proto":
        fg = mem.v @ ff          # [K, hw]
        bg = mem.b @ ff
        return ((alpha[:, None] * (fg - bg)).sum(0)).reshape(h, w)
    R = 0
    for i in range(len(mem.fg_patches)):
        fgs = (mem.fg_patches[i].t() @ ff).max(0).values
        bgs = (mem.bg_patches[i].t() @ ff).max(0).values if mem.bg_patches[i].shape[1] else torch.zeros_like(fgs)
        R = R + alpha[i] * (fgs - bgs)
    return R.reshape(h, w)


def response_boxes(R: torch.Tensor, rel_thr=0.5, max_boxes=3):
    """Correspondence-driven geometric prior: connected components of {R > rel_thr * max R} (only if max R > 0),
    ranked by peak response; returns normalised xyxy boxes (grid-cell extents)."""
    from scipy import ndimage
    r = R.float().cpu().numpy()
    mx = r.max()
    if mx <= 0:
        return []
    lab, n = ndimage.label(r > rel_thr * mx, structure=np.ones((3, 3)))
    comps = []
    h, w = r.shape
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        comps.append((r[ys, xs].max(), [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]))
    comps.sort(key=lambda t: -t[0])
    return [c[1] for c in comps[:max_boxes]]
