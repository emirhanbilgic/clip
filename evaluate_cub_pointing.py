import argparse
import os
import math
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import numpy as np

# Reuse saliency from the existing implementation
from clip_concept_attention import compute_patch_saliency_joint, visualize_saliency_on_image


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
        base_text_features = model.get_text_features(**text_inputs).detach()  # (1, D)

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

            # Expand text features to match batch
            bsz = o_x_for_sal.shape[0]
            text_features = base_text_features.repeat(bsz, 1)
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


# ---------------------------- Part-level utilities ----------------------------

def transform_point_to_224(x: float, y: float, orig_w: int, orig_h: int, target: int = 224) -> Tuple[float, float]:
    if orig_w <= 0 or orig_h <= 0:
        return 0.0, 0.0
    scale = target / float(min(orig_w, orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    x_s = x * scale
    y_s = y * scale
    off_x, off_y = center_crop_params(new_w, new_h, target)
    x_c = x_s - off_x
    y_c = y_s - off_y
    x_c = max(0.0, min(float(target), x_c))
    y_c = max(0.0, min(float(target), y_c))
    return x_c, y_c


def make_224_center_crop(img: Image.Image, target: int = 224) -> Image.Image:
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return img
    scale = target / float(min(orig_w, orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    img_r = img.resize((new_w, new_h), resample=Image.BICUBIC)
    off_x, off_y = center_crop_params(new_w, new_h, target)
    return img_r.crop((off_x, off_y, off_x + target, off_y + target))


def read_cub_parts(cub_root: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    parts_txt = os.path.join(cub_root, "parts", "parts.txt")
    part_id_to_name: Dict[int, str] = {}
    name_to_part_id: Dict[str, int] = {}
    with open(parts_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: part_id part_name
            pid_str, *name_tokens = line.split()
            pid = int(pid_str)
            name = " ".join(name_tokens)
            name_l = name.strip().lower()
            part_id_to_name[pid] = name_l
            name_to_part_id[name_l] = pid
    return part_id_to_name, name_to_part_id


def read_cub_part_locs(cub_root: str) -> Dict[int, List[Tuple[int, float, float, int]]]:
    locs_txt = os.path.join(cub_root, "parts", "part_locs.txt")
    image_to_parts: Dict[int, List[Tuple[int, float, float, int]]] = {}
    with open(locs_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: image_id part_id x y visible
            img_id_str, part_id_str, x_str, y_str, vis_str = line.split()
            img_id = int(img_id_str)
            part_id = int(part_id_str)
            x = float(x_str)
            y = float(y_str)
            visible = int(vis_str)
            image_to_parts.setdefault(img_id, []).append((part_id, x, y, visible))
    return image_to_parts


def build_default_concept_to_parts(part_id_to_name: Dict[int, str]) -> Dict[str, List[int]]:
    # Known CUB parts often include: back, beak, belly, breast, crown, forehead,
    # left eye, right eye, left leg, right leg, left wing, right wing, nape, tail, throat
    name_map = {pid: name.lower() for pid, name in part_id_to_name.items()}

    def ids_with(tokens: List[str]) -> List[int]:
        out: List[int] = []
        for pid, nm in name_map.items():
            for t in tokens:
                if t in nm:
                    out.append(pid)
                    break
        return out

    def exact(names: List[str]) -> List[int]:
        targets = set([n.lower() for n in names])
        return [pid for pid, nm in name_map.items() if nm in targets]

    concept_to_parts: Dict[str, List[int]] = {}
    concept_to_parts["beak"] = ids_with(["beak", "bill"]) or exact(["beak"])  # robustness
    concept_to_parts["wing"] = ids_with(["wing"])  # left/right wing
    concept_to_parts["eye"] = ids_with(["eye"])    # left/right eye
    concept_to_parts["leg"] = ids_with(["leg"])    # left/right leg
    concept_to_parts["tail"] = ids_with(["tail"])  # tail
    # Head composite: crown, forehead, nape, throat, beak, eyes
    head_tokens = ["crown", "forehead", "nape", "throat", "beak", "bill", "eye", "head"]
    concept_to_parts["head"] = ids_with(head_tokens)
    # Optional torso concepts
    concept_to_parts["breast"] = ids_with(["breast"]) or exact(["breast"])
    concept_to_parts["belly"] = ids_with(["belly"]) or exact(["belly"])
    concept_to_parts["back"] = ids_with(["back"]) or exact(["back"])

    # Remove empty entries
    concept_to_parts = {k: v for k, v in concept_to_parts.items() if len(v) > 0}
    return concept_to_parts


def compute_peak_xy(
    sal_vec: np.ndarray,
    grid_h: int,
    grid_w: int,
    target: int,
    method: str = "argmax",
) -> Tuple[float, float]:
    if sal_vec.size == 0:
        return 0.0, 0.0
    if method == "argmax":
        argmax_idx = int(np.argmax(sal_vec))
        return patch_index_to_center_xy(argmax_idx, grid_h, grid_w, target)
    # center of mass
    cell_h = target / float(grid_h)
    cell_w = target / float(grid_w)
    idxs = np.arange(sal_vec.size)
    rows = (idxs // grid_w).astype(np.float32)
    cols = (idxs % grid_w).astype(np.float32)
    cx = np.sum(sal_vec * ((cols + 0.5) * cell_w))
    cy = np.sum(sal_vec * ((rows + 0.5) * cell_h))
    return float(cx), float(cy)


def compute_radius(
    tr_box: Tuple[float, float, float, float],
    target: int,
    radius_mode: str,
    radius_ratio: float,
    radius_px: Optional[float],
) -> float:
    if radius_mode == "abs" and radius_px is not None:
        return float(radius_px)
    bx, by, bw, bh = tr_box
    if radius_mode == "bbox":
        base = math.sqrt(max(bw * bh, 1e-6))
        return float(radius_ratio * base)
    # minHW over the 224 crop (i.e., 224)
    base = float(min(target, target))
    return float(radius_ratio * base)


def normalize_distance(
    dist: float,
    tr_box: Tuple[float, float, float, float],
    target: int,
    norm_mode: str,
) -> float:
    if norm_mode == "bbox":
        _, _, bw, bh = tr_box
        denom = max(math.sqrt(max(bw * bh, 1e-6)), 1e-6)
        return float(dist / denom)
    denom = float(min(target, target))
    return float(dist / max(denom, 1e-6))


def evaluate_part_pointing(
    cub_root: str,
    model_name: str,
    concepts: List[str],
    prompt_template: str,
    split: str,
    batch_size: int,
    device: torch.device,
    peak_method: str = "argmax",
    radius_mode: str = "bbox",
    radius_ratio: float = 0.1,
    radius_px: Optional[float] = None,
    norm_mode: str = "bbox",
    max_images: int = None,
    save_examples_dir: Optional[str] = None,
    examples_per_concept: int = 0,
) -> Dict[str, object]:
    index = read_cub_index_files(cub_root)
    image_ids: List[int] = [
        i for i, rec in index.items() if (rec["is_train"] if split == "train" else not rec["is_train"])
    ]
    image_ids.sort()
    if max_images is not None:
        image_ids = image_ids[:max_images]

    part_id_to_name, _ = read_cub_parts(cub_root)
    concept_to_parts = build_default_concept_to_parts(part_id_to_name)
    # Filter requested concepts by availability in the dataset
    eval_concepts = [c for c in concepts if c in concept_to_parts]
    if len(eval_concepts) == 0:
        return {"error": "No requested concepts available in CUB parts."}

    image_to_parts = read_cub_part_locs(cub_root)

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    grid_h, grid_w, target = infer_patch_grid(model)

    per_concept_stats: Dict[str, Dict[str, float]] = {
        c: {"success": 0.0, "total": 0.0, "dists": []} for c in eval_concepts
    }

    # example saving bookkeeping
    examples_saved: Dict[str, int] = {c: 0 for c in eval_concepts}

    for i in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[i : i + batch_size]
        batch_imgs: List[Image.Image] = []
        batch_boxes: List[Tuple[float, float, float, float]] = []
        batch_sizes: List[Tuple[int, int]] = []

        for image_id in batch_ids:
            rel_path = index[image_id]["rel_path"]
            abs_path = os.path.join(cub_root, "images", rel_path)
            bbox = index[image_id]["bbox"]
            img = Image.open(abs_path).convert("RGB")
            w0, h0 = img.size
            batch_imgs.append(img)
            batch_boxes.append(bbox)
            batch_sizes.append((w0, h0))

        inputs = processor(images=batch_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vis_outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
            o_x = vis_outputs.last_hidden_state
            o_x_for_sal = o_x[:, 1:, :] if o_x.shape[1] > 1 else o_x

        # Evaluate each concept separately
        for concept in eval_concepts:
            prompt = prompt_template.format(concept=concept)
            with torch.no_grad():
                text_inputs = processor(text=prompt, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                base_text_features = model.get_text_features(**text_inputs).detach()  # (1, D)
                bsz = o_x_for_sal.shape[0]
                text_features = base_text_features.repeat(bsz, 1)

                sal = compute_patch_saliency_joint(model, o_x_for_sal, text_features)

            mapped_part_ids = set(concept_to_parts[concept])

            for b, image_id in enumerate(batch_ids):
                # Ground truth visible keypoints for this concept
                gt_points: List[Tuple[float, float]] = []
                for (pid, x, y, vis) in image_to_parts.get(image_id, []):
                    if vis != 1 or pid not in mapped_part_ids:
                        continue
                    orig_w, orig_h = batch_sizes[b]
                    px, py = transform_point_to_224(x, y, orig_w, orig_h, target)
                    gt_points.append((px, py))
                if len(gt_points) == 0:
                    continue  # skip if part not visible

                tr_box = transform_bbox_to_224(batch_boxes[b], batch_sizes[b][0], batch_sizes[b][1], target)
                radius = compute_radius(tr_box, target, radius_mode, radius_ratio, radius_px)

                sal_vec = sal[b]
                px_pred, py_pred = compute_peak_xy(sal_vec, grid_h, grid_w, target, peak_method)

                # Distance to nearest visible keypoint
                dists = [math.hypot(px_pred - gx, py_pred - gy) for (gx, gy) in gt_points]
                d_min = min(dists)
                d_norm = normalize_distance(d_min, tr_box, target, norm_mode)

                per_concept_stats[concept]["total"] += 1.0
                per_concept_stats[concept]["dists"].append(d_norm)
                if d_min <= radius:
                    per_concept_stats[concept]["success"] += 1.0

                # Save examples if requested
                if save_examples_dir is not None and examples_per_concept > 0:
                    if examples_saved[concept] < examples_per_concept:
                        try:
                            os.makedirs(os.path.join(save_examples_dir, concept), exist_ok=True)
                            img_crop = make_224_center_crop(batch_imgs[b], target)
                            title = f"{concept} | d={d_min:.1f} r={radius:.1f} {'✓' if d_min <= radius else '✗'}"
                            out_path = os.path.join(
                                save_examples_dir,
                                concept,
                                f"img{image_id}_ex{examples_saved[concept]+1}.png",
                            )
                            visualize_saliency_on_image(
                                img_crop,
                                sal_vec,
                                (grid_h, grid_w),
                                title=title,
                                save_path=out_path,
                                show=False,
                            )
                            examples_saved[concept] += 1
                        except Exception as e:
                            # Non-fatal: continue evaluation even if saving fails
                            pass

    # Aggregate metrics
    per_concept_metrics: Dict[str, Dict[str, float]] = {}
    successes = 0.0
    totals = 0.0
    for c, stats in per_concept_stats.items():
        tot = stats["total"]
        suc = stats["success"]
        acc = float(suc / tot) if tot > 0 else 0.0
        dists = stats["dists"]
        mean_d = float(np.mean(dists)) if len(dists) > 0 else 0.0
        med_d = float(np.median(dists)) if len(dists) > 0 else 0.0
        per_concept_metrics[c] = {
            "pointing_accuracy": acc,
            "num_images": float(tot),
            "mean_norm_dist": mean_d,
            "median_norm_dist": med_d,
        }
        successes += suc
        totals += tot

    micro = float(successes / totals) if totals > 0 else 0.0
    macro = float(np.mean([m["pointing_accuracy"] for m in per_concept_metrics.values()])) if len(per_concept_metrics) > 0 else 0.0

    return {
        "per_concept": per_concept_metrics,
        "micro_avg_pointing_accuracy": micro,
        "macro_avg_pointing_accuracy": macro,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate object/part pointing-game on CUB using CLIP concept saliency"
    )
    parser.add_argument("--cub-root", type=str, required=True, help="Path to CUB_200_2011 root")
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="object",
        choices=["object", "part"],
        help="Evaluation type: object bounding-box pointing or part keypoint pointing",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="bird",
        help="Text concept for object-level eval (default: 'bird')",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="beak,wing,tail,eye,leg,head",
        help="Comma-separated concept list for part-level eval",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="a photo of a bird {concept}",
        help="Prompt template for part concepts (use {concept})",
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
    parser.add_argument("--peak", type=str, default="argmax", choices=["argmax", "com"], help="Peak selection")
    parser.add_argument("--radius-mode", type=str, default="bbox", choices=["bbox", "minHW", "abs"], help="Part pointing radius mode")
    parser.add_argument("--radius-ratio", type=float, default=0.1, help="Radius ratio for bbox/minHW modes")
    parser.add_argument("--radius-px", type=float, default=None, help="Absolute radius in pixels when radius-mode=abs")
    parser.add_argument("--norm-mode", type=str, default="bbox", choices=["bbox", "minHW"], help="Normalization for distance metrics")
    parser.add_argument("--save-examples-dir", type=str, default=None, help="Directory to save part-level heatmap examples")
    parser.add_argument("--examples-per-concept", type=int, default=4, help="Number of examples to save per concept")
    parser.add_argument("--out-json", type=str, default=None, help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Evaluating CUB ...")
    if args.eval == "object":
        print(f"Object-level pointing | Model: {args.model_name} | Concept: '{args.concept}' | Split: {args.split}")
        metrics = evaluate_pointing(
            cub_root=args.cub_root,
            model_name=args.model_name,
            concept_text=args.concept,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
            max_images=args.max_images,
        )
    else:
        concepts = [c.strip().lower() for c in args.concepts.split(",") if c.strip()]
        print(
            f"Part-level pointing | Model: {args.model_name} | Concepts: {concepts} | Split: {args.split} | Peak={args.peak}"
        )
        metrics = evaluate_part_pointing(
            cub_root=args.cub_root,
            model_name=args.model_name,
            concepts=concepts,
            prompt_template=args.prompt_template,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
            peak_method=args.peak,
            radius_mode=args.radius_mode,
            radius_ratio=args.radius_ratio,
            radius_px=args.radius_px,
            norm_mode=args.norm_mode,
            max_images=args.max_images,
            save_examples_dir=args.save_examples_dir,
            examples_per_concept=args.examples_per_concept,
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


