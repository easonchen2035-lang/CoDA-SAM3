"""GPU-synchronised per-query latency for each method (single class, batch 1, bf16, H100).

Timed region per query (torch.cuda.synchronize() before start and before stop):
  PIL image already in memory (disk I/O excluded) -> SAM3 preprocessing -> image encoder ->
  [routing + response map + geometric boxes] -> SAM3 decodes for the method's prompt configurations ->
  mask upsampling to eval resolution -> affinity + mask-IoU NMS (CPU, pycocotools) -> final instances.
'cached'   : support memory (K support encodings, prototypes, exemplar tokens) built once beforehand.
'uncached' : support memory is rebuilt inside the timed region for every query (K support image encodings).
Text embeddings are cached in both settings (one phrase per class).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, json, sys, time
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mu
from coda.sam3core import FrozenSAM3, cat_prompts
from coda.support import draw_supports, build_memory, route, response, response_boxes
from coda import gpu_rle
from coda.evaluate import _nms, consensus_vote

P = argparse.ArgumentParser()
P.add_argument("--dataset", default="kvasir"); P.add_argument("--n", type=int, default=50)
P.add_argument("--K", type=int, default=3); P.add_argument("--out", required=True)
P.add_argument("--methods", nargs="+", default=["T", "E", "TE", "TEr", "G", "TG", "TEG", "fm"])
P.add_argument("--fusions", nargs="*", default=[])   # e.g. "T+TEr+TG"
a = P.parse_args()
D = f"{PROJECT_ROOT}/data/proc/{a.dataset}"
pool = json.load(open(f"{D}/pool.json")); test = json.load(open(f"{D}/test.json"))
cid, cname = test["categories"][0]["id"], test["categories"][0]["name"]
S = FrozenSAM3(feat_source="fpn72")
T = S.encode_text(cname); V = S.encode_text("visual")
ids = draw_supports(pool, cid, a.K, 0, a.dataset)
imgs = [Image.open(f"{D}/images/{im['file_name']}").convert("RGB") for im in test["images"][: a.n + 5]]
pool_imgs = {im["id"]: Image.open(f"{D}/images/{im['file_name']}").convert("RGB") for im in pool["images"] if im["id"] in ids}


def mem_build():
    cache = {}
    for iid in ids:                       # encode from in-memory PIL images (I/O excluded)
        cache[iid] = S.encode_image(pool_imgs[iid])
    return build_memory(S, pool, f"{D}/images", cid, ids, cache)


def post(o, H, W, R, lam=1.0, keep=10):
    sc = o["scores"]; idx = torch.argsort(sc, descending=True)[:keep]
    ml = o["mask_logits"][idx]
    full = (F.interpolate(ml[None], size=(H, W), mode="bilinear", align_corners=False)[0] > 0)
    A = torch.zeros(len(idx), device="cuda")
    if R is not None:
        m72 = (F.interpolate(ml[None], size=R.shape, mode="bilinear", align_corners=False)[0] > 0).float()
        A = (m72 * R).sum((1, 2)) / m72.sum((1, 2)).clamp_min(1)
    rles = gpu_rle.encode(full)
    return [{"s": float(sc[j] + lam * A[n]), "rle": rles[n]} for n, j in enumerate(idx)]


def post_cands(o, H, W, R, keep=10):
    """same candidate format as run_infer (score, rle, Aw) for consensus voting"""
    sc = o["scores"]; idx = torch.argsort(sc, descending=True)[:keep]
    ml = o["mask_logits"][idx]
    full = F.interpolate(ml[None], size=(H, W), mode="bilinear", align_corners=False)[0] > 0
    m72 = (F.interpolate(ml[None], size=R.shape, mode="bilinear", align_corners=False)[0] > 0).float()
    A = ((m72 * R).sum((1, 2)) / m72.sum((1, 2)).clamp_min(1)).tolist()
    rles = gpu_rle.encode(full)
    return [{"score": float(sc[j]), "rle": rles[n], "Aw": A[n], "Au": A[n]} for n, j in enumerate(idx.tolist())]


def run(img, method, mem):
    H, W = img.height, img.width
    enc = S.encode_image(img)
    if method.startswith("vote"):           # e.g. vote-E1-mv : K single-support decodes + consensus (+ mask) voting
        _, cfg, *flags = method.split("-")
        alpha = route(mem, enc.gfeat); R = response(mem, enc.feat, alpha, "knn")
        ps = [{"cands": post_cands(S.decode(enc, *cat_prompts([T if cfg == "TE1" else V, tk])), H, W, R)} for tk in mem.tokens]
        v = consensus_vote(ps, alpha.tolist(), 0.5, "mv" in flags)
        return _nms([{"s": x["score"], "rle": x["c"]["rle"], "box": None} for x in v], 0.5)
    srcs = method.split("+")
    if method == "T":
        return post(S.decode(enc, *T), H, W, None)
    alpha = route(mem, enc.gfeat); R = response(mem, enc.feat, alpha, "knn")
    if method == "fm":
        r = F.interpolate(R[None, None].float(), size=(H, W), mode="bilinear")[0, 0]
        return [(r > 0.5 * R.max()).cpu().numpy()]
    gb = response_boxes(R)
    gt = S.geometry_tokens(enc, [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in gb]) if gb else None
    ex = cat_prompts(mem.tokens); exr = mem.tokens[int(alpha.argmax())]
    table = {"T": [T], "E": [V, ex], "TE": [T, ex], "TEr": [T, exr], "G": [V, gt] if gt else None,
             "TG": [T, gt] if gt else None, "TEG": [T, ex, gt] if gt else [T, ex]}
    cands = []
    for s in srcs:
        if table[s] is None:
            continue
        cands += post(S.decode(enc, *cat_prompts(table[s])), H, W, R if s != "T" or len(srcs) > 1 else None)
    return _nms(cands, 0.5)


res = {"dataset": a.dataset, "K": a.K, "n": a.n, "gpu": torch.cuda.get_device_name(), "methods": {}}
mem0 = mem_build()
for m in a.methods + a.fusions:
    for mode in ["cached", "uncached"]:
        if m == "T" and mode == "uncached":
            continue
        ts = []
        for i, img in enumerate(imgs):
            torch.cuda.synchronize(); t = time.perf_counter()
            mem = mem_build() if mode == "uncached" else mem0
            run(img, m, mem)
            torch.cuda.synchronize(); dt = time.perf_counter() - t
            if i >= 5:
                ts.append(dt * 1000)
        res["methods"][f"{m}|{mode}"] = {"median_ms": float(np.median(ts)), "mean_ms": float(np.mean(ts)),
                                         "p90_ms": float(np.percentile(ts, 90))}
        print(m, mode, f"median {np.median(ts):.1f} ms", flush=True)
torch.cuda.synchronize(); t = time.perf_counter(); mem_build(); torch.cuda.synchronize()
res["support_memory_build_ms_K"] = (time.perf_counter() - t) * 1000
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
