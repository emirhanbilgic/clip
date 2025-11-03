# CLIP CUB Pointing Evaluation

Evaluate pointing-game accuracy on the CUB-200-2011 dataset using CLIP concept saliency in the joint space.

## Quick start (Kaggle Notebook)

1. Prepare CUB-200-2011 as a Kaggle Dataset (or add it to the notebook as an input): it should contain the folder `CUB_200_2011` with the standard structure, including `images/`, `images.txt`, `bounding_boxes.txt`, `train_test_split.txt`, etc.
2. Clone this repo and install dependencies (Kaggle already has PyTorch):

```bash
!git clone https://github.com/REPLACE_WITH_YOUR_REPO/clip-cub-pointing.git
%cd clip-cub-pointing
!pip install -q -r requirements.txt
```

3. Run the evaluation (adjust the CUB root to your Kaggle input path):

```bash
python evaluate_cub_pointing.py \
  --cub-root /kaggle/input/cub-200-2011/CUB_200_2011 \
  --model-name openai/clip-vit-base-patch32 \
  --concept "bird" \
  --split test \
  --batch-size 16 \
  --max-images 2000 \
  --out-json /kaggle/working/cub_clip_pointing.json
```

The script prints a JSON-like dictionary and writes metrics to `--out-json` when provided. The key `pointing_game_accuracy` is the final metric.

## Local run (conda/pip)

```bash
pip install -r requirements.txt
# If you don't have PyTorch yet, install it from https://pytorch.org/ (or, CPU-only example):
# pip install torch --index-url https://download.pytorch.org/whl/cpu

python evaluate_cub_pointing.py \
  --cub-root /path/to/CUB_200_2011 \
  --model-name openai/clip-vit-base-patch32 \
  --concept "bird" \
  --split test \
  --batch-size 16 \
  --max-images 2000 \
  --out-json /tmp/cub_clip_pointing.json
```

## Notes

- The evaluation follows the CLIP preprocessing pipeline (resize shortest side to 224, center-crop to 224×224), then maps the predicted most-salient patch center back to the 224×224 crop to check if it falls inside the transformed ground-truth bounding box.
- Model and processor are loaded from Hugging Face Hub (e.g., `openai/clip-vit-base-patch32`). Ensure internet access is enabled in your environment to download weights the first time.


