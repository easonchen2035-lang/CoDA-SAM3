"""Lossless COCO RLE encoding of a batch of binary masks computed on the GPU.
Only run-change positions are transferred to the CPU (instead of n*H*W bytes)."""
import numpy as np
import torch
from pycocotools import mask as mu


def encode(masks: torch.Tensor):
    """masks: bool [n,H,W] on GPU -> list of compressed COCO RLE dicts (counts as str), identical to
    pycocotools.mask.encode of the same arrays."""
    n, H, W = masks.shape
    if n == 0:
        return []
    m = masks.transpose(1, 2).reshape(n, -1)                      # column-major (Fortran) order
    L = m.shape[1]
    ch = (m[:, 1:] != m[:, :-1]).nonzero()                       # [k,2] (row, pos)
    first = m[:, 0].cpu().numpy()
    ch = ch.cpu().numpy()
    rows, pos = ch[:, 0], ch[:, 1] + 1
    split = np.searchsorted(rows, np.arange(1, n))
    out = []
    for i, p in enumerate(np.split(pos, split)):
        b = np.concatenate([[0], p, [L]])
        counts = np.diff(b).tolist()
        if first[i]:
            counts = [0] + counts
        r = mu.frPyObjects({"counts": counts, "size": [H, W]}, H, W)
        r["counts"] = r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]
        out.append(r)
    return out
