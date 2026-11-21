# CoDA-SAM3

Research code for **CoDA-SAM3: Training-Free Support-Conditioned Adaptation for Object Detection and Segmentation**. The method adapts frozen SAM3 using a small set of annotated support images.

## Environment

The study used Python 3.12.3, PyTorch 2.11.0 with CUDA 12.8, Transformers 5.22.1 and an NVIDIA H100 80 GB. The main inference dependencies include Pillow, NumPy, SciPy and pycocotools. Obtain access to `facebook/sam3` and use checkpoint revision `3c879f39` to match the paper. The original loader uses the locally available Hugging Face revision.

## Running the experiments

Run the commands below from the repository root after installing dependencies and obtaining the datasets. `scripts/prepare_data.py` documents the expected raw data layouts and builds the pool, validation and test splits. An example for Kvasir-SEG follows.

```sh
mkdir -p runs/kvasir
python scripts/prepare_data.py kvasir
python scripts/run_infer.py --dataset kvasir --split val --K 3 --seeds 0 1 2 --out runs/kvasir
```

See `scripts/run_infer.py` for token transfer and `scripts/run_select.py` for native-canvas context experiments. `scripts/eval_specs.py` evaluates saved predictions. `configs/` contains experiment definitions. `scripts/*.sbatch` are optional SLURM examples; create `logs/slurm` first and adapt the partition and environment to your cluster. Download the images and model weights separately from their respective sources.

## Version and scope

This source snapshot was exported from the experiment server on 21 November 2026. The SAM3 wrapper, support logic, AP code and GPU RLE implementation match the manuscript v4 archive byte for byte. The current evaluator also contains later joint-voting and pooling branches. The configuration named `frozen_selection.json` belongs to a later global-selection experiment. For the v4 results use the splits, support budgets, methods and thresholds specified in each manuscript table. Machine-specific project paths in the distributed scripts have been replaced with repository-relative paths.
