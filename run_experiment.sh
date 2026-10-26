#!/bin/bash
set -e

echo "1. Preprocessing"
py src/preprocess.py

echo "2. Training"
py src/train.py

echo "3. Prediction"
py src/predict.py

echo "Done!"
