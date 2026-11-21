"""Download iSAID val annotations (official JSON, mirrored by isaaccorley/isaid)
and DOTA-v1.0 val images (torchgeo/dota). Extraction done separately."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from huggingface_hub import hf_hub_download
out = f"{PROJECT_ROOT}/data/raw/isaid"
for repo, fn in [("isaaccorley/isaid", "isaid_annotations_val.tar.gz"),
                 ("torchgeo/dota", "dotav1.0_images_val.tar.gz")]:
    p = hf_hub_download(repo, fn, repo_type="dataset", local_dir=out)
    print("done", p, flush=True)
print("ALLDONE")
