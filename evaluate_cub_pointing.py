import argparse
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import numpy as np

# Reuse saliency from the existing implementation
from clip_concept_attention import compute_patch_saliency_joint


def read_cub_index_files(cub_root: str) -> Dict[int, Dict]:
    images_txt = os.path.join(cub_root, "images.txt")
    boxes_txt = os.path.join(cub_root, "bounding_boxes.txt")
    split_txt = os.path.join(cub_root, "train_test_split.txt")

    image_id_to_relpath: Dict[int, str] = {}
    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image_id = int(parts[0])
            rel_path = parts[1]
            image_id_to_relpath[image_id] = rel_path

    image_id_to_box: Dict[int, Tuple[float, float, float, float]] = {}
    with open(boxes_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            image_id_to_box[image_id] = (x, y, w, h)

    image_id_is_train: Dict[int, bool] = {}
    with open(split_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            image_id = int(parts[0])
            is_train = parts[1] == "1"
            image_id_is_train[image_id] = is_train

    merged: Dict[int, Dict] = {}
    for image_id, rel_path in image_id_to_relpath.items():
        if image_id not in image_id_to_box or image_id not in image_id_is_train:
            continue
        merged[image_id] = {
            "rel_path": rel_path,
            "bbox": image_id_to_box[image_id],
            "is_train": image_id_is_train[image_id],
        }
    return merged


def center_crop_params(new_w: int, new_h: int, target: int = 224) -> Tuple[int, int]:
    off_x = max((new_w - target) // 2, 0)
    off_y = max((new_h - target) // 2, 0)
    return off_x, off_y


def transform_bbox_to_224(
    bbox: Tuple[float, float, float, float], orig_w: int, orig_h: int, target: int = 224
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    if orig_w <= 0 or orig_h <= 0:
        return 0.0, 0.0, 0.0, 0.0

    scale = target / float(min(orig_w, orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    x_s = x * scale
    y_s = y * scale
    w_s = w * scale
    h_s = h * scale

    off_x, off_y = center_crop_params(new_w, new_h, target)
    x_c = x_s - off_x
    y_c = y_s - off_y

    x_c2 = x_c + w_s
    y_c2 = y_c + h_s

    x_c = max(0.0, min(float(target), x_c))
    y_c = max(0.0, min(float(target), y_c))
    x_c2 = max(0.0, min(float(target), x_c2))
    y_c2 = max(0.0, min(float(target), y_c2))

    w_c = max(0.0, x_c2 - x_c)
    h_c = max(0.0, y_c2 - y_c)
    return x_c, y_c, w_c, h_c


def patch_index_to_center_xy(
    patch_index: int, grid_h: int, grid_w: int, target: int = 224
) -> Tuple[float, float]:
    row = patch_index // grid_w
    col = patch_index % grid_w
    cell_h = target / float(grid_h)
    cell_w = target / float(grid_w)
    cx = (col + 0.5) * cell_w
    cy = (row + 0.5) * cell_h
    return cx, cy


def point_in_box(x: float, y: float, box: Tuple[float, float, float, float]) -> bool:
    bx, by, bw, bh = box
    return (x >= bx) and (x <= bx + bw) and (y >= by) and (y <= by + bh)


def infer_patch_grid(model: CLIPModel) -> Tuple[int, int, int]:
    image_size = getattr(model.vision_model.config, "image_size", 224)
    patch_size = getattr(model.vision_model.config, "patch_size", 32)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    grid_h = image_size // patch_size
    grid_w = image_size // patch_size
    return grid_h, grid_w, image_size


def evaluate_pointing(
    cub_root: str,
    model_name: str,
    concept_text: str,
    split: str,
    batch_size: int,
    device: torch.device,
    max_images: int = None,
) -> Dict[str, float]:
    index = read_cub_index_files(cub_root)
    image_ids: List[int] = [
        i for i, rec in index.items() if (rec["is_train"] if split == "train" else not rec["is_train"])
    ]
    image_ids.sort()
    if max_images is not None:
        image_ids = image_ids[:max_images]

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    grid_h, grid_w, target = infer_patch_grid(model)

    with torch.no_grad():
        text_inputs = processor(text=concept_text, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = model.get_text_features(**text_inputs).detach()

    total = 0
    success = 0

    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i : i + batch_size]
        batch_imgs: List[Image.Image] = []
        batch_paths: List[str] = []
        batch_boxes: List[Tuple[float, float, float, float]] = []
        batch_sizes: List[Tuple[int, int]] = []

        for image_id in batch_ids:
            rel_path = index[image_id]["rel_path"]
            abs_path = os.path.join(cub_root, "images", rel_path)
            bbox = index[image_id]["bbox"]
            img = Image.open(abs_path).convert("RGB")
            w0, h0 = img.size
            batch_imgs.append(img)
            batch_paths.append(abs_path)
            batch_boxes.append(bbox)
            batch_sizes.append((w0, h0))

        inputs = processor(images=batch_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vis_outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
            o_x = vis_outputs.last_hidden_state
            if o_x.shape[1] > 1:
                o_x_for_sal = o_x[:, 1:, :]
            else:
                o_x_for_sal = o_x

            sal = compute_patch_saliency_joint(model, o_x_for_sal, text_features)

        for b in range(len(batch_ids)):
            sal_vec = sal[b]
            if sal_vec.size == 0:
                continue
            argmax_idx = int(np.argmax(sal_vec))
            cx, cy = patch_index_to_center_xy(argmax_idx, grid_h, grid_w, target)

            orig_w, orig_h = batch_sizes[b]
            tr_box = transform_bbox_to_224(batch_boxes[b], orig_w, orig_h, target)

            total += 1
            if point_in_box(cx, cy, tr_box):
                success += 1

    acc = (success / total) if total > 0 else 0.0
    return {"pointing_game_accuracy": acc, "num_images": float(total)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pointing-game accuracy on CUB using CLIP concept saliency"
    )
    parser.add_argument("--cub-root", type=str, required=True, help="Path to CUB_200_2011 root")
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="bird",
        help="Text concept to localize (default: 'bird')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out-json", type=str, default=None, help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Evaluating CUB pointing-game ...")
    print(f"Model: {args.model_name} | Concept: '{args.concept}' | Split: {args.split}")

    metrics = evaluate_pointing(
        cub_root=args.cub_root,
        model_name=args.model_name,
        concept_text=args.concept,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
        max_images=args.max_images,
    )
    result = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    print(result)
    if args.out_json is not None:
        import json
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved metrics to {args.out_json}")


if __name__ == "__main__":
    main()


