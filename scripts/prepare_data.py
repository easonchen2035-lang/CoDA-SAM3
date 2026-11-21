"""Build unified, versioned evaluation data for CoDA-SAM3 re-implementation.

Output per dataset D in data/proc/D/:
  images/<name>.png         images at evaluation resolution (long side <= 1024, aspect kept; area-resampled)
  pool.json / val.json / test.json   COCO-format, instance masks as RLE at evaluation resolution
  manifest.json             split membership, source ids, seed, rules, sha256 of each split json

Splits are disjoint at the level of *source images* (for iSAID crops: at the level of the original P-image).
  pool : labelled support pool (support images are drawn from here only)
  val  : hyper-parameter / threshold selection only
  test : final reporting only (never used for any selection)

Instance definition:
  Kvasir-SEG, ISIC2018 (binary masks): 8-connected components of the GT mask at eval resolution with area >= MIN_CC px.
  NWPU VHR-10, iSAID: annotated polygons rasterised at eval resolution (COCO instances).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse, hashlib, io, json, os, random, sys, zipfile
import numpy as np
from PIL import Image
from scipy import ndimage
from pycocotools import mask as mu

ROOT = f"{PROJECT_ROOT}/data"
MAX_SIDE = 1024
MIN_CC = 50  # px at eval resolution
SEED = 20261102
Image.MAX_IMAGE_PIXELS = None


def resize_eval(img, max_side=MAX_SIDE):
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.BOX if img.mode != "L" else Image.NEAREST)
    return img, s


def rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def cc_instances(mask):
    lab, n = ndimage.label(mask > 0, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= MIN_CC:
            out.append(m)
    return out


class Builder:
    def __init__(self, name, categories):
        self.name = name
        self.dir = f"{ROOT}/proc/{name}"
        os.makedirs(f"{self.dir}/images", exist_ok=True)
        self.cats = categories
        self.splits = {}

    def add(self, split, file_stem, img, insts, source_id):
        """insts: list of (category_id, bool mask at eval res)."""
        d = self.splits.setdefault(split, {"images": [], "annotations": []})
        iid = len(d["images"])
        fn = f"{file_stem}.png"
        p = f"{self.dir}/images/{fn}"
        if not os.path.exists(p):
            img.save(p)
        d["images"].append({"id": iid, "file_name": fn, "width": img.width, "height": img.height, "source_id": source_id})
        for cid, m in insts:
            r = rle(m)
            x, y, w, h = mu.toBbox(r).tolist()
            d["annotations"].append({"id": len(d["annotations"]) + 1, "image_id": iid, "category_id": int(cid),
                                     "segmentation": r, "area": float(m.sum()), "bbox": [x, y, w, h], "iscrowd": 0})

    def finish(self, rules):
        man = {"dataset": self.name, "seed": SEED, "max_side": MAX_SIDE, "min_cc": MIN_CC, "rules": rules,
               "categories": self.cats, "splits": {}}
        for s, d in self.splits.items():
            d["categories"] = self.cats
            js = json.dumps(d)
            open(f"{self.dir}/{s}.json", "w").write(js)
            man["splits"][s] = {"n_images": len(d["images"]), "n_instances": len(d["annotations"]),
                                "sha256": hashlib.sha256(js.encode()).hexdigest(),
                                "source_ids": [im["source_id"] for im in d["images"]]}
            print(self.name, s, len(d["images"]), "images", len(d["annotations"]), "instances", flush=True)
        json.dump(man, open(f"{self.dir}/manifest.json", "w"), indent=1)


def split_ids(ids, n_pool, n_val, seed=SEED):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    return ids[:n_pool], ids[n_pool:n_pool + n_val], ids[n_pool + n_val:]


# ---------------------------------------------------------------- Kvasir-SEG
def kvasir():
    R = f"{ROOT}/raw/kvasir/Kvasir-SEG"
    ids = sorted(f[:-4] for f in os.listdir(f"{R}/images"))
    # The first 100 sorted ids form the support pool (the dev probe probe_exemplar_modes.py only touched ids[:60]);
    # the remaining 900 are shuffled into val 200 / test 700.
    pool = ids[:100]
    rest = ids[100:]
    random.Random(SEED).shuffle(rest)
    val, test = rest[:200], rest[200:]
    B = Builder("kvasir", [{"id": 1, "name": "polyp"}])
    for split, L in [("pool", pool), ("val", val), ("test", test)]:
        for i in L:
            img, _ = resize_eval(Image.open(f"{R}/images/{i}.jpg").convert("RGB"))
            m = Image.open(f"{R}/masks/{i}.jpg").convert("L").resize(img.size, Image.BILINEAR)
            m = np.array(m) > 127
            B.add(split, i, img, [(1, x) for x in cc_instances(m)], i)
    B.finish({"pool": "first 100 sorted image ids", "val/test": f"remaining 900 shuffled with seed {SEED}: 200/700",
              "instances": f"8-connected components of GT mask (thr 127 after bilinear resize), area>={MIN_CC}px"})


# ---------------------------------------------------------------- ISIC2018 Task 1
def isic(pool_n=200):
    R = f"{ROOT}/raw/isic2018"
    B = Builder("isic2018", [{"id": 1, "name": "skin lesion"}])

    def read_split(inp_zip, gt_zip, ids_subset=None):
        zi, zg = zipfile.ZipFile(f"{R}/{inp_zip}"), zipfile.ZipFile(f"{R}/{gt_zip}")
        imgs = {os.path.basename(n)[:-4]: n for n in zi.namelist() if n.lower().endswith(".jpg")}
        gts = {os.path.basename(n).replace("_segmentation.png", ""): n for n in zg.namelist() if n.endswith("_segmentation.png")}
        keys = sorted(set(imgs) & set(gts))
        return zi, zg, imgs, gts, keys

    # pool: random subset of the official training split
    zi, zg, imgs, gts, keys = read_split("ISIC2018_Task1-2_Training_Input.zip", "ISIC2018_Task1_Training_GroundTruth.zip")
    pool = sorted(random.Random(SEED).sample(keys, pool_n))
    todo = [("pool", zi, zg, imgs, gts, pool)]
    zi2, zg2, imgs2, gts2, keys2 = read_split("ISIC2018_Task1-2_Validation_Input.zip", "ISIC2018_Task1_Validation_GroundTruth.zip")
    todo.append(("val", zi2, zg2, imgs2, gts2, keys2))
    zi3, zg3, imgs3, gts3, keys3 = read_split("ISIC2018_Task1-2_Test_Input.zip", "ISIC2018_Task1_Test_GroundTruth.zip")
    todo.append(("test", zi3, zg3, imgs3, gts3, keys3))
    for split, zi, zg, imgs, gts, L in todo:
        for k in L:
            img, _ = resize_eval(Image.open(io.BytesIO(zi.read(imgs[k]))).convert("RGB"))
            m = Image.open(io.BytesIO(zg.read(gts[k]))).convert("L").resize(img.size, Image.BILINEAR)
            m = np.array(m) > 127
            B.add(split, k, img, [(1, x) for x in cc_instances(m)], k)
    B.finish({"pool": f"{pool_n} images sampled (seed {SEED}) from the official 2594-image Task1 training split",
              "val": "official Task1 validation split (100)", "test": "official Task1 test split (1000)",
              "instances": f"8-connected components of GT mask, area>={MIN_CC}px"})


# ---------------------------------------------------------------- NWPU VHR-10 (COCO masks)
NWPU_NAMES = {1: "airplane", 2: "ship", 3: "storage tank", 4: "baseball diamond", 5: "tennis court",
              6: "basketball court", 7: "ground track field", 8: "harbor", 9: "bridge", 10: "vehicle"}


def nwpu():
    R = f"{ROOT}/raw/nwpu"
    ann = json.load(open(f"{R}/repo/NWPU VHR-10_dataset_coco/annotations.json"))
    by_img = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    ims = {im["id"]: im for im in ann["images"]}
    pool, val, test = split_ids(sorted(ims), 130, 130)
    B = Builder("nwpu", [{"id": k, "name": v} for k, v in NWPU_NAMES.items()])
    for split, L in [("pool", pool), ("val", val), ("test", test)]:
        for iid in L:
            im = ims[iid]
            img0 = Image.open(f"{R}/x/NWPU VHR-10 dataset/positive image set/{im['file_name']}").convert("RGB")
            assert img0.size == (im["width"], im["height"]), im
            img, s = resize_eval(img0)
            insts = []
            for a in by_img.get(iid, []):
                seg = a["segmentation"]
                if isinstance(seg, list):
                    seg = [[c * s for c in poly] for poly in seg if len(poly) >= 6]
                    if not seg:
                        continue
                    m = mu.decode(mu.merge(mu.frPyObjects(seg, img.height, img.width))).astype(bool)
                else:
                    raise ValueError("unexpected seg")
                if m.sum() > 0:
                    insts.append((a["category_id"], m))
            B.add(split, im["file_name"][:-4], img, insts, im["file_name"])
    B.finish({"source": "650 positive images, COCO masks from chaozhong2010/VHR-10_dataset_coco",
              "split": f"image-level shuffle seed {SEED}: pool 130 / val 130 / test 390",
              "negative_set": "the 150 negative images are not used"})


# ---------------------------------------------------------------- iSAID (val split, 896px crops)
ISAID_NAMES = {1: "ship", 2: "storage tank", 3: "baseball diamond", 4: "tennis court", 5: "basketball court",
               6: "ground track field", 7: "bridge", 8: "large vehicle", 9: "small vehicle", 10: "helicopter",
               11: "swimming pool", 12: "roundabout", 13: "soccer ball field", 14: "plane", 15: "harbor"}
CROP = 896


def isaid(max_crops={"pool": 400, "val": 200, "test": 600}):
    R = f"{ROOT}/raw/isaid/val"
    ann = json.load(open(f"{R}/Annotations/iSAID_val.json"))
    by_img = {}
    for a in ann["annotations"]:
        by_img.setdefault(int(a["image_id"]), []).append(a)
    ims = {int(im["id"]): im for im in ann["images"]}
    pool, val, test = split_ids(sorted(ims), 92, 92)
    B = Builder("isaid", [{"id": k, "name": v} for k, v in ISAID_NAMES.items()])
    rng = random.Random(SEED)
    for split, L in [("pool", pool), ("val", val), ("test", test)]:
        cand = []
        for iid in L:
            im = ims[iid]
            W, H = im["width"], im["height"]
            xs = sorted(set(list(range(0, max(W - CROP, 0) + 1, CROP)) + [max(W - CROP, 0)]))
            ys = sorted(set(list(range(0, max(H - CROP, 0) + 1, CROP)) + [max(H - CROP, 0)]))
            for x0 in xs:
                for y0 in ys:
                    cand.append((iid, x0, y0))
        rng.shuffle(cand)
        kept = 0
        cache = {}
        for iid, x0, y0 in cand:
            if kept >= max_crops[split]:
                break
            im = ims[iid]
            W, H = im["width"], im["height"]
            cw, ch = min(CROP, W), min(CROP, H)
            insts = []
            for a in by_img.get(iid, []):
                bx, by_, bw, bh = json.loads(a["bbox"]) if isinstance(a["bbox"], str) else a["bbox"]
                if bx > x0 + cw or by_ > y0 + ch or bx + bw < x0 or by_ + bh < y0:
                    continue
                seg = json.loads(a["segmentation"]) if isinstance(a["segmentation"], str) else a["segmentation"]
                seg = [[(c - x0) if j % 2 == 0 else (c - y0) for j, c in enumerate(p)] for p in seg if len(p) >= 6]
                if not seg:
                    continue
                m = mu.decode(mu.merge(mu.frPyObjects(seg, ch, cw))).astype(bool)
                full_area = float(a["area"]) if not isinstance(a["area"], str) else float(a["area"])
                # keep instances with >= 50% of their area inside the crop and >= 16 px
                if m.sum() >= 16 and m.sum() >= 0.5 * full_area:
                    insts.append((int(a["category_id"]), m))
            if not insts:
                continue
            if iid not in cache:
                cache.clear()
                cache[iid] = Image.open(f"{R}/images/{im['file_name']}").convert("RGB")
            img = cache[iid].crop((x0, y0, x0 + cw, y0 + ch))
            B.add(split, f"{im['file_name'][:-4]}_{x0}_{y0}", img, insts, f"{im['file_name']}@{x0},{y0}")
            kept += 1
    B.finish({"source": "iSAID val split (458 DOTA-v1 val images, official iSAID annotations)",
              "split": f"source-image shuffle seed {SEED}: pool 92 / val 92 / test 274 source images",
              "crops": f"{CROP}x{CROP} grid crops (edge-aligned), crops with >=1 kept instance; random subset "
                       f"{max_crops} per split (seed {SEED})",
              "instance_keep": ">=50% of instance area inside crop and >=16px",
              "note": "NOT the iSAID-5i protocol (256px patches, 3 class folds); all 15 classes evaluated"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    a = ap.parse_args()
    for d in a.datasets:
        globals()[d]()
