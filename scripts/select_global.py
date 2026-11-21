"""Freeze the CoDA-SAM3 configuration from VAL ONLY.
Rule: among the CoDA candidate specs (any 'vote:' or 'fuse:' spec), choose the single spec maximising the mean over the
4 datasets of the val selection score (AP50_box + mIoU@tau*)/2 (draw-mean). tau* stays per dataset (val-selected).
Baselines are fixed specs (no choice), each with its own val-selected tau per dataset.
Writes configs/frozen_selection.json."""
import json
import numpy as np

DS = ["kvasir", "isic2018", "nwpu", "isaid"]
BASELINES = {"SAM3 text-only": "src:T", "Direct exemplar (cross-image tokens, all K)": "src:E",
             "Feature matching (FM)": "fm", "Geometric prompt (response boxes)": "src:G",
             "Text + geometric prompt": "src:TG"}
S = {d: json.load(open(f"results/val_{d}_summary.json"))["specs"] for d in DS}
common = set.intersection(*[set(v) for v in S.values()])
cands = [s for s in common if s.startswith("vote:") or s.startswith("fuse:")]
score = {s: float(np.mean([S[d][s]["sel"] for d in DS])) for s in cands}
order = sorted(cands, key=lambda s: -score[s])
print("top-10 CoDA candidate specs by mean val selection score:")
for s in order[:10]:
    print(f"  {100*score[s]:.2f}  {s}   " + "  ".join(f"{d}:{100*S[d][s]['sel']:.1f}" for d in DS))
best = order[0]
frozen = {"rule": __doc__, "coda_spec": best, "mean_val_sel": score[best],
          "tau": {d: S[d][best]["tau"] for d in DS},
          "baselines": {k: {"spec": v, "tau": {d: S[d][v]["tau"] for d in DS}} for k, v in BASELINES.items()},
          "top10": [(s, score[s]) for s in order[:10]]}
json.dump(frozen, open("configs/frozen_selection.json", "w"), indent=1)
print("FROZEN:", best, frozen["tau"])
