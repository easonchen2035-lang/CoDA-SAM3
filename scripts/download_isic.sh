#!/bin/bash
PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
# ISIC2018 Task1 official files (moved to isic-archive S3 bucket). Retries, skips completed files.
cd "$PROJECT_ROOT/data/raw/isic2018"
for f in ISIC2018_Task1-2_Validation_Input.zip ISIC2018_Task1_Validation_GroundTruth.zip ISIC2018_Task1_Test_GroundTruth.zip ISIC2018_Task1-2_Test_Input.zip ISIC2018_Task1_Training_GroundTruth.zip ISIC2018_Task1-2_Training_Input.zip; do
  [ -f "$f" ] && { echo "$f exists"; continue; }
  for i in $(seq 1 20); do
    curl -sSfL --retry 5 --retry-all-errors -C - -o "$f.tmp" "https://isic-archive.s3.amazonaws.com/challenges/2018/$f" && { mv "$f.tmp" "$f"; echo "$f ok"; break; }
    echo "$f attempt $i failed"; sleep 10
  done
done
echo ALLDONE
