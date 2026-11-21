#!/bin/bash
PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
# Downloads medical datasets from official sources (resumable).
set -u
ROOT="$PROJECT_ROOT/data/raw"
cd $ROOT/kvasir && curl -sSL -C - -o kvasir-seg.zip https://datasets.simula.no/downloads/kvasir-seg.zip; echo "kvasir rc=$?"
cd $ROOT/isic2018
for f in ISIC2018_Task1-2_Validation_Input.zip ISIC2018_Task1_Validation_GroundTruth.zip ISIC2018_Task1_Test_GroundTruth.zip ISIC2018_Task1-2_Test_Input.zip; do
  curl -sSL -C - -o $f https://isic-challenge-data.s3.amazonaws.com/2018/$f; echo "$f rc=$?"
done
echo ALLDONE
