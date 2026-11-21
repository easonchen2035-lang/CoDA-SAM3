"""Summarise eval_specs JSON: mean ± std over support draws.
On a VAL file: tau* per spec = argmax over tau of the draw-mean mIoU; selection score = (AP50_box + mIoU@tau*) / 2.
On a TEST file: pass --taus-from <val summary json> so each spec uses its val-selected tau (never test-tuned)."""
import argparse, json
import numpy as np

P = argparse.ArgumentParser()
P.add_argument("file"); P.add_argument("--taus-from"); P.add_argument("--out"); P.add_argument("--top", type=int, default=200)
P.add_argument("--filter", default="")
a = P.parse_args()
d = json.load(open(a.file))
by = {}
for r in d["runs"]:
    by.setdefault(r["spec"], []).append(r)
fixed = json.load(open(a.taus_from))["specs"] if a.taus_from else None
rows = {}
for spec, rs in by.items():
    if a.filter and a.filter not in spec:
        continue
    rs = sorted(rs, key=lambda r: r["seed"])
    taus = list(rs[0]["sem"].keys())
    if fixed is not None:
        if spec not in fixed:
            continue
        tau = fixed[spec]["tau"]
    else:
        tau = max(taus, key=lambda t: np.mean([r["sem"][t]["mIoU"] for r in rs]))
    f = lambda key: [r[key]["AP50"] for r in rs]
    miou = [r["sem"][tau]["mIoU"] for r in rs]; dice = [r["sem"][tau]["mDice"] for r in rs]
    rows[spec] = {"tau": tau, "seeds": [r["seed"] for r in rs],
                  "ap50_box": f("ap50_box"), "ap50_mask": f("ap50_mask"), "mIoU": miou, "mDice": dice,
                  "sel": float((np.mean(f("ap50_box")) + np.mean(miou)) / 2)}
order = sorted(rows, key=lambda s: -rows[s]["sel"])
ms = lambda v: f"{100*np.mean(v):5.1f}±{100*np.std(v):4.1f}"
print(f"{d['dataset']} {d['split']}  (mean±std over draws, %)")
print(f"{'spec':48s} {'tau':>6s} {'AP50box':>11s} {'AP50mask':>11s} {'mIoU':>11s} {'mDice':>11s}  sel")
for s in order[: a.top]:
    r = rows[s]
    print(f"{s:48s} {r['tau']:>6s} {ms(r['ap50_box'])} {ms(r['ap50_mask'])} {ms(r['mIoU'])} {ms(r['mDice'])}  {100*r['sel']:.1f}")
if a.out:
    json.dump({"dataset": d["dataset"], "split": d["split"], "specs": rows}, open(a.out, "w"), indent=1)
