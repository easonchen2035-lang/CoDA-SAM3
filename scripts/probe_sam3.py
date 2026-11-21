"""Smoke probe: verifies our staged decode == HF Sam3Model.forward, inspects feature shapes,
and checks text-only / in-image box / cross-image exemplar behaviour on one Kvasir pair."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys, time, json
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np, torch
from PIL import Image
from coda.sam3core import FrozenSAM3, cat_prompts

R = f"{PROJECT_ROOT}/data/raw/kvasir/Kvasir-SEG"
bb = json.load(open(f"{R}/kavsir_bboxes.json"))
ids = sorted(bb)[:2]
S = FrozenSAM3()
print("vit last_hidden_state", tuple(S.model.get_vision_features(
    pixel_values=torch.zeros(1, 3, 1008, 1008, device="cuda", dtype=torch.bfloat16)).last_hidden_state.shape))
imgs = [Image.open(f"{R}/images/{i}.jpg").convert("RGB") for i in ids]
encs = [S.encode_image(im) for im in imgs]
print("feat", encs[0].feat.shape, [tuple(f.shape) for f in encs[0].vision.fpn_hidden_states])

# 1) equivalence with HF forward (text only)
inp = S.processor(images=imgs[1], text="polyp", return_tensors="pt").to("cuda")
inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
with torch.no_grad():
    ref = S.model(**inp)
tt = S.encode_text("polyp")
out = S.decode(encs[1], *tt)
r_scores = (ref.pred_logits.float().sigmoid() * ref.presence_logits.float().sigmoid())[0]
print("max |score diff| vs HF forward:", (r_scores - out["scores"]).abs().max().item())

def gt_box(i):
    b = bb[i]["bbox"][0]; return [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
print("GT query box", gt_box(ids[1]), "size", imgs[1].size)
def top(o, k=3):
    s, idx = o["scores"].topk(k); H, W = encs[1].orig_hw
    return [(round(float(a), 3), [round(float(x), 1) for x in (o["boxes"][j] * torch.tensor([W, H, W, H], device="cuda")).tolist()]) for a, j in zip(s, idx)]
print("text-only top3:", top(out))
# 2) cross-image exemplar tokens from support image 0
gt0 = gt_box(ids[0])
ex = S.geometry_tokens(encs[0], [gt0])
print("text+cross exemplar:", top(S.decode(encs[1], *cat_prompts([tt, ex]))))
vis_tt = S.encode_text("visual")
print("'visual'+cross exemplar:", top(S.decode(encs[1], *cat_prompts([vis_tt, ex]))))
# 3) in-image GT box (oracle, sanity only)
ex_in = S.geometry_tokens(encs[1], [gt_box(ids[1])])
print("text+in-image GT box (oracle sanity):", top(S.decode(encs[1], *cat_prompts([tt, ex_in]))))
# timing
torch.cuda.synchronize(); t = time.time()
for _ in range(10): e = S.encode_image(imgs[1])
torch.cuda.synchronize(); print("encode_image ms", (time.time() - t) * 100)
t = time.time()
for _ in range(10): S.decode(encs[1], *tt)
torch.cuda.synchronize(); print("decode ms", (time.time() - t) * 100)
print("mem GB", torch.cuda.max_memory_allocated() / 1e9)
