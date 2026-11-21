"""Shared figure style + overlay helpers (vector PDF output, consistent palette)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools import mask as mu

# colour-blind-safe palette (Okabe-Ito based)
PAL = {"ours": "#D55E00", "text": "#0072B2", "exemplar": "#009E73", "mosaic": "#CC79A7", "fm": "#999999",
       "geo": "#E69F00", "gt": "#F0E442", "aux": "#56B4E9"}
OVERLAY = {"gt": (0.10, 0.85, 0.25), "pred": (1.00, 0.35, 0.05), "fp": (0.95, 0.10, 0.10)}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
                         "axes.spines.right": False})


def overlay(img, masks, color, alpha=0.45, contour=True):
    """img uint8 HxWx3; masks list of RLE or bool arrays."""
    out = img.astype(np.float32).copy()
    acc = np.zeros(img.shape[:2], bool)
    for m in masks:
        m = mu.decode(m).astype(bool) if isinstance(m, dict) else m
        acc |= m
    c = np.array(color) * 255
    out[acc] = (1 - alpha) * out[acc] + alpha * c
    if contour and acc.any():
        from scipy import ndimage
        edge = acc ^ ndimage.binary_erosion(acc, iterations=2)
        out[edge] = c
    return out.clip(0, 255).astype(np.uint8)


def boxes(ax, bxs, color, lw=1.0):
    import matplotlib.patches as pa
    for b in bxs:
        ax.add_patch(pa.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=lw))
