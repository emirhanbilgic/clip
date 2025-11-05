import argparse
import os
import math
import glob
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import unicodedata

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import average_precision_score, roc_auc_score

from clip_concept_attention import compute_patch_saliency_joint


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    # remove diacritics
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    return " ".join(s.split())


STANDARD_CONCEPTS = {
    "lobed arch",
    "porthole",
    "broken pediment",
    "solomonic column",
    "pointed arch",
    "serliana",
    "trefoil arch",
    "rounded arch",
    "ogee arch",
}

# Spanish/English aliases to standard concept names
CONCEPT_ALIASES: Dict[str, str] = {}
_concept_pairs = [
    ("lobed arch", ["lobed arch", "arco lobulado", "arco trilobulado", "multilobed arch", "lobulado"]),
    ("porthole", ["porthole", "ojo de buey", "oculus"]),
    ("broken pediment", ["broken pediment", "fronton partido", "fronton roto"]),
    ("solomonic column", ["solomonic column", "columna salomonica", "salomonica"]),
    ("pointed arch", ["pointed arch", "arco apuntado", "arco ojival", "ogival arch", "ojival"]),
    ("serliana", ["serliana", "serliana arch", "arquitectura serliana"]),
    ("trefoil arch", ["trefoil arch", "arco trefoil", "arco trilobulado", "trifolio", "arco trebolado"]),
    ("rounded arch", ["rounded arch", "round arch", "semi circular arch", "semicircular arch", "arco de medio punto", "medio punto"]),
    ("ogee arch", ["ogee arch", "arco conopial", "conopial"]),
]
for std, aliases in _concept_pairs:
    for a in aliases:
        CONCEPT_ALIASES[normalize_text(a)] = std

# Class aliases to standard class names used in scenarios
CLASS_ALIASES: Dict[str, str] = {}
_class_pairs = [
    ("baroque", ["baroque", "barroco"]),
    ("gothic", ["gothic", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico", "gotico"]),
    ("renaissance", ["renaissance", "renacentista", "renacimiento"]),
    ("hispanic-muslim", ["hispanic muslim", "hispanic-muslim", "hispanomusulman", "hispanomusulman", "hispanomusulman", "hispanomusulman"]),
]
for std, aliases in _class_pairs:
    for a in aliases:
        CLASS_ALIASES[normalize_text(a)] = std


SCENARIOS = [
    ("hispanic-muslim", "baroque", "lobed arch"),
    ("baroque", "renaissance", "porthole"),
    ("baroque", "gothic", "broken pediment"),
    ("baroque", "renaissance", "solomonic column"),
    ("gothic", "hispanic-muslim", "pointed arch"),
    ("renaissance", "baroque", "serliana"),
    ("gothic", "baroque", "trefoil arch"),
    ("baroque", "renaissance", "rounded arch"),
    ("gothic", "renaissance", "ogee arch"),
]


def map_class_name(name: str) -> Optional[str]:
    n = normalize_text(name)
    if n in CLASS_ALIASES:
        return CLASS_ALIASES[n]
    # try direct if already standard
    if n in {"baroque", "gothic", "renaissance", "hispanic muslim", "hispanic-muslim"}:
        return "hispanic-muslim" if n in {"hispanic muslim", "hispanic-muslim"} else n
    return None


def infer_class_from_path(path: str) -> Optional[str]:
    lower = path.lower()
    for cls in ["baroque", "gothic", "renaissance", "hispanic-muslim", "hispanic muslim"]:
        if f"/{cls}/" in lower or lower.endswith(f"/{cls}"):
            return map_class_name(cls)
    # fallback: use immediate parent dir name
    base = os.path.basename(os.path.dirname(path))
    mapped = map_class_name(base)
    return mapped


def find_xml_for_image(img_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    d = os.path.dirname(img_path)
    candidates = [
        os.path.join(d, f"{stem}.xml"),
        os.path.join(d, "xml", f"{stem}.xml"),
        os.path.join(os.path.dirname(d), "xml", f"{stem}.xml"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def parse_monumai_xml(xml_path: str) -> Tuple[List[Tuple[str, Tuple[int, int, int, int]]], Optional[str], Tuple[int, int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    folder = root.findtext("folder")
    width = root.findtext("size/width")
    height = root.findtext("size/height")
    try:
        W = int(width) if width is not None else None
        H = int(height) if height is not None else None
    except Exception:
        W, H = None, None

    objects: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for obj in root.findall("object"):
        name_tag = obj.findtext("name") or ""
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = int(float(bnd.findtext("xmin")))
            ymin = int(float(bnd.findtext("ymin")))
            xmax = int(float(bnd.findtext("xmax")))
            ymax = int(float(bnd.findtext("ymax")))
        except Exception:
            continue
        c_name = normalize_text(name_tag)
        mapped = CONCEPT_ALIASES.get(c_name, c_name)
        objects.append((mapped, (xmin, ymin, xmax, ymax)))

    clabel = map_class_name(folder) if folder else None
    return objects, clabel, (W or -1, H or -1)


def transform_boxes_to_clip_coords(boxes: List[Tuple[int, int, int, int]], orig_size: Tuple[int, int], target_size: int = 224) -> List[Tuple[int, int, int, int]]:
    W, H = orig_size
    if W <= 0 or H <= 0:
        return []
    short = min(W, H)
    scale = target_size / short
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    # center crop to target_size x target_size
    left = max(0, (new_w - target_size) // 2)
    top = max(0, (new_h - target_size) // 2)

    out: List[Tuple[int, int, int, int]] = []
    for (xmin, ymin, xmax, ymax) in boxes:
        x0 = int(round(xmin * scale)) - left
        y0 = int(round(ymin * scale)) - top
        x1 = int(round(xmax * scale)) - left
        y1 = int(round(ymax * scale)) - top
        x0 = max(0, min(target_size, x0))
        y0 = max(0, min(target_size, y0))
        x1 = max(0, min(target_size, x1))
        y1 = max(0, min(target_size, y1))
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out


def boxes_to_mask(boxes: List[Tuple[int, int, int, int]], size: Tuple[int, int]) -> np.ndarray:
    W, H = size
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        mask[y0:y1, x0:x1] = 1
    return mask


def upsample_saliency_to_pixels(sal: np.ndarray, image_size: int) -> np.ndarray:
    # sal is (H_p, W_p) or (N,) flattened
    if sal.ndim == 1:
        s = int(math.sqrt(int(sal.size)))
        sal = sal.reshape(s, s)
    h_p, w_p = sal.shape
    scale_y = image_size // h_p
    scale_x = image_size // w_p
    sal_up = np.kron(sal, np.ones((scale_y, scale_x), dtype=sal.dtype))
    # ensure exact size
    sal_up = sal_up[:image_size, :image_size]
    return sal_up


def compute_patch_grid_sizes(model: CLIPModel) -> Tuple[int, int]:
    image_size = getattr(model.vision_model.config, "image_size", 224)
    patch_size = getattr(model.vision_model.config, "patch_size", 32)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    h_p = image_size // patch_size
    w_p = image_size // patch_size
    return h_p, w_p


def transform_image_to_clip_224(img: Image.Image, target_size: int = 224) -> Image.Image:
    W, H = img.size
    if W <= 0 or H <= 0:
        return img.convert("RGB").resize((target_size, target_size), Image.BICUBIC)
    short = min(W, H)
    scale = target_size / short
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    img_scaled = img.resize((new_w, new_h), Image.BICUBIC)
    left = max(0, (new_w - target_size) // 2)
    top = max(0, (new_h - target_size) // 2)
    img_224 = img_scaled.crop((left, top, left + target_size, top + target_size))
    return img_224


def draw_boxes(image_224: Image.Image, boxes_224: List[Tuple[int, int, int, int]], color: Tuple[int, int, int] = (0, 255, 0), width: int = 2) -> Image.Image:
    out = image_224.copy()
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1) in boxes_224:
        for w in range(width):
            draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=color)
    return out


def save_example_visualization(save_path: str, img_224: Image.Image, saliency_224: np.ndarray, boxes_224: List[Tuple[int, int, int, int]], title_left: str, title_right: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # Left: GT boxes
    img_gt = draw_boxes(img_224, boxes_224, color=(0, 255, 0), width=2)
    ax[0].imshow(img_gt)
    ax[0].axis("off")
    ax[0].set_title(title_left)

    # Right: saliency heatmap overlay + boxes
    ax[1].imshow(img_224)
    ax[1].imshow(saliency_224, cmap="gray", alpha=0.45, norm=Normalize(vmin=float(saliency_224.min()), vmax=float(saliency_224.max())))
    # overlay boxes as green rectangles
    for (x0, y0, x1, y1) in boxes_224:
        ax[1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="lime", linewidth=1.5)
    ax[1].axis("off")
    ax[1].set_title(title_right)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def save_mask_comparison(save_path: str, img_224: Image.Image, gt_mask_224: np.ndarray, pred_mask_224: np.ndarray, title_left: str, title_right: str):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # Left: GT mask overlay
    ax[0].imshow(img_224)
    ax[0].imshow(gt_mask_224, cmap="Reds", alpha=0.45)
    ax[0].axis("off")
    ax[0].set_title(title_left)

    # Right: Predicted mask overlay
    ax[1].imshow(img_224)
    ax[1].imshow(pred_mask_224, cmap="Greens", alpha=0.45)
    ax[1].axis("off")
    ax[1].set_title(title_right)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def compute_image_text_score(model: CLIPModel, processor: CLIPProcessor, img: Image.Image, concept_text: str, device: torch.device) -> float:
    with torch.no_grad():
        text_inputs = processor(text=[concept_text], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        img_features = model.get_image_features(pixel_values=pixel_values)
        img_features = F.normalize(img_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        score = torch.einsum("bd,bd->b", img_features, text_features).item()
    return float(score)


def compute_saliency_for_image(model: CLIPModel, processor: CLIPProcessor, img: Image.Image, concept_text: str, device: torch.device) -> Tuple[np.ndarray, int]:
    # returns (saliency_vector length N, image_size)
    image_size = getattr(model.vision_model.config, "image_size", 224)
    with torch.no_grad():
        text_inputs = processor(text=[concept_text], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        vis_outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        o_x = vis_outputs.last_hidden_state
        if o_x.shape[1] > 1:
            o_x_for_sal = o_x[:, 1:, :]
        else:
            o_x_for_sal = o_x
        sal = compute_patch_saliency_joint(model, o_x_for_sal, text_features)
        sal_vec = sal[0]  # (N,)
    return sal_vec, image_size


def evaluate_localization(pairs: List[Tuple[str, str, List[Tuple[int, int, int, int]]]], model_name: str, device: torch.device,
                         save_examples_dir: Optional[str] = None, examples_per_class: int = 5, topk_percent: Optional[float] = None) -> Dict[str, float]:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    ious: List[float] = []
    accs: List[float] = []
    aps: List[float] = []
    per_class_saved: Dict[str, int] = {}

    for img_path, concept_text, boxes in pairs:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        sal_vec, image_size = compute_saliency_for_image(model, processor, img, concept_text, device)
        sal_grid = upsample_saliency_to_pixels(sal_vec, image_size)  # (H, W)

        orig_W, orig_H = img.size
        boxes_224 = transform_boxes_to_clip_coords(boxes, (orig_W, orig_H), target_size=image_size)
        if not boxes_224:
            # no positives; skip for localization
            continue
        gt_mask = boxes_to_mask(boxes_224, (image_size, image_size))  # (H, W)

        # Binarize saliency: keep top P% if specified, else mean threshold
        if topk_percent is not None and 0.0 < float(topk_percent) < 100.0:
            q = 1.0 - (float(topk_percent) / 100.0)
            thr = float(np.quantile(sal_grid, q))
            pred_bin = (sal_grid >= thr).astype(np.uint8)
        else:
            thr = float(sal_grid.mean())
            pred_bin = (sal_grid > thr).astype(np.uint8)

        # IoU
        inter = np.logical_and(pred_bin == 1, gt_mask == 1).sum()
        union = np.logical_or(pred_bin == 1, gt_mask == 1).sum()
        iou = float(inter) / float(union) if union > 0 else 0.0
        ious.append(iou)

        # Pixel Accuracy
        acc = float((pred_bin == gt_mask).sum()) / float(gt_mask.size)
        accs.append(acc)

        # Pixel-wise AP
        y_true = gt_mask.flatten().astype(np.uint8)
        y_score = sal_grid.flatten().astype(np.float32)
        if y_true.max() > 0:  # only compute AP if positives exist
            ap = float(average_precision_score(y_true, y_score))
            aps.append(ap)

        # Optionally save example visualization
        if save_examples_dir:
            cls_label = infer_class_from_path(img_path) or "unknown"
            cls_label = normalize_text(cls_label)
            count = per_class_saved.get(cls_label, 0)
            if count < examples_per_class:
                img_224 = transform_image_to_clip_224(img, target_size=image_size)
                # Ensure saliency grid is exactly 224x224
                sal_224 = sal_grid.astype(np.float32)
                # Compose filename
                stem = os.path.splitext(os.path.basename(img_path))[0]
                subdir = os.path.join(save_examples_dir, cls_label)
                out_path = os.path.join(subdir, f"{stem}_{concept_text.replace(' ', '_')}.png")
                title_l = f"GT: {cls_label} / {concept_text}"
                title_r = "Saliency + GT"
                try:
                    save_example_visualization(out_path, img_224, sal_224, boxes_224, title_l, title_r)
                    # Also save GT vs Pred mask overlays
                    top_label = f"top{int(topk_percent)}%" if topk_percent is not None else "mean"
                    out_mask = os.path.join(subdir, f"{stem}_{concept_text.replace(' ', '_')}_{top_label}_masks.png")
                    save_mask_comparison(
                        out_mask,
                        img_224,
                        gt_mask.astype(np.uint8),
                        pred_bin.astype(np.uint8),
                        title_left="GT mask",
                        title_right=f"Pred mask ({top_label})",
                    )
                    per_class_saved[cls_label] = count + 1
                except Exception:
                    pass

    metrics = {
        "PixelAcc": float(np.mean(accs)) if accs else 0.0,
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }
    return metrics


def evaluate_detection_scenarios(
    samples: List[Tuple[str, str, str, List[str]]],
    model_name: str,
    device: torch.device,
    trials: int = 10,
    debug: bool = False,
    save_examples_dir: Optional[str] = None,
    examples_per_class: int = 5,
) -> Dict[str, float]:
    """
    Paper protocol: for each scenario (c1, c2, k), compute AUROC between
    positives = images of class c1 where k is present and
    negatives = images of class c1 where k is absent.
    Perform 10 random balanced subsampling runs per scenario and average.
    """
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    aucs: List[float] = []
    scenarios_used = 0
    skipped_scenarios: List[str] = []

    detection_runs = max(1, int(trials))
    rng = np.random.RandomState(42)

    # per-class saved counters split by polarity
    per_class_pos_saved: Dict[str, int] = {}
    per_class_neg_saved: Dict[str, int] = {}

    for (c1, c2, k_raw) in SCENARIOS:
        k = normalize_text(k_raw)

        # gather subsets
        scores_present: List[float] = []  # c1 & k present
        scores_absent_c1: List[float] = []  # c1 & k absent

        for img_path, cls_label, _, present_list in samples:
            if cls_label != c1:
                continue  # only class c1 is considered in paper evaluation
            is_present = any(normalize_text(x) == k for x in present_list)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            s = compute_image_text_score(model, processor, img, k, device)

            if is_present:
                scores_present.append(s)
            else:
                scores_absent_c1.append(s)

            # Optionally save visualization for positives and negatives
            if save_examples_dir:
                try:
                    cls_norm = normalize_text(cls_label)
                    bucket = "pos" if is_present else "neg"
                    if is_present:
                        count = per_class_pos_saved.get(cls_norm, 0)
                    else:
                        count = per_class_neg_saved.get(cls_norm, 0)
                    if count < examples_per_class:
                        # compute saliency heatmap for (img, k)
                        sal_vec, image_size = compute_saliency_for_image(model, processor, img, k, device)
                        sal_grid = upsample_saliency_to_pixels(sal_vec, image_size)

                        # get GT boxes for k if present (for overlay)
                        boxes_224: List[Tuple[int, int, int, int]] = []
                        xml_path = find_xml_for_image(img_path)
                        if xml_path:
                            try:
                                objects, _, (W, H) = parse_monumai_xml(xml_path)
                                # filter boxes for current concept k
                                k_boxes = [b for (cname, b) in objects if normalize_text(cname) == k]
                                if k_boxes and W > 0 and H > 0:
                                    boxes_224 = transform_boxes_to_clip_coords(k_boxes, (W, H), target_size=image_size)
                            except Exception:
                                boxes_224 = []

                        # prepare image at 224
                        img_224 = transform_image_to_clip_224(img, target_size=image_size)

                        # paths
                        subdir = os.path.join(save_examples_dir, cls_norm, bucket)
                        os.makedirs(subdir, exist_ok=True)
                        stem = os.path.splitext(os.path.basename(img_path))[0]
                        out_path = os.path.join(subdir, f"{stem}_{k.replace(' ', '_')}.png")
                        title_l = f"GT: {cls_norm} / {k}"
                        title_r = f"Saliency + {'GT' if boxes_224 else 'NoGT'}"
                        save_example_visualization(out_path, img_224, sal_grid.astype(np.float32), boxes_224, title_l, title_r)

                        if is_present:
                            per_class_pos_saved[cls_norm] = count + 1
                        else:
                            per_class_neg_saved[cls_norm] = count + 1
                except Exception:
                    # best-effort saving; ignore failures
                    pass

        # skip scenarios without both subsets
        if not scores_present or not scores_absent_c1:
            if debug:
                print(f"[SKIP] scenario (c1={c1}, c2={c2}, k={k}): present={len(scores_present)}, absent={len(scores_absent_c1)}")
            skipped_scenarios.append(f"(c1={c1}, c2={c2}, k={k})")
            continue

        # Balanced subsampling runs
        n = min(len(scores_present), len(scores_absent_c1))
        if n <= 0:
            continue
        scenarios_used += 1
        for run_idx in range(detection_runs):
            rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
            pos_idx = rs.choice(len(scores_present), size=n, replace=False)
            neg_idx = rs.choice(len(scores_absent_c1), size=n, replace=False)

            y_true: List[int] = [1] * n + [0] * n
            y_score: List[float] = [scores_present[i] for i in pos_idx] + [scores_absent_c1[j] for j in neg_idx]

            try:
                auc = float(roc_auc_score(y_true, y_score))
                aucs.append(auc)
            except Exception:
                continue
        if debug:
            print(f"[OK] scenario (c1={c1}, c2={c2}, k={k}): present={len(scores_present)}, absent={len(scores_absent_c1)}, trials={detection_runs}")

    if debug and skipped_scenarios:
        print("Skipped scenarios:")
        for sc in skipped_scenarios:
            print("  -", sc)

    return {
        "AUROC_mean": float(np.mean(aucs)) if aucs else 0.0,
        "AUROC_count": len(aucs),
        "Scenarios_used": scenarios_used,
    }


def collect_dataset_pairs(dataset_root: str) -> Tuple[List[Tuple[str, str, List[Tuple[int, int, int, int]]]], List[Tuple[str, str, str, List[str]]]:
    """
    Returns:
      - localization_pairs: list of (img_path, concept_text, boxes)
      - detection_samples: list of (img_path, cls_label, concept_text_placeholder, present_concepts)
    concept_text_placeholder is unused in detection but kept for clarity.
    """
    image_globs = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.JPG", "**/*.PNG"]
    image_paths: List[str] = []
    for g in image_globs:
        image_paths.extend(glob.glob(os.path.join(dataset_root, g), recursive=True))

    localization_pairs: List[Tuple[str, str, List[Tuple[int, int, int, int]]]] = []
    detection_samples: List[Tuple[str, str, str, List[str]]] = []

    for img_path in image_paths:
        xml_path = find_xml_for_image(img_path)
        if not xml_path:
            continue
        try:
            objects, xml_folder, (W, H) = parse_monumai_xml(xml_path)
        except Exception:
            continue

        cls_raw = infer_class_from_path(img_path) or (xml_folder if xml_folder else "")
        cls_label = map_class_name(cls_raw) or normalize_text(cls_raw)
        present_concepts: List[str] = []

        # group by concept
        concept_to_boxes: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for cname, box in objects:
            cname_norm = normalize_text(cname)
            present_concepts.append(cname_norm)
            concept_to_boxes.setdefault(cname_norm, []).append(box)

        # localization for images where concept is present
        for cname_norm, boxes in concept_to_boxes.items():
            concept_text = CONCEPT_ALIASES.get(cname_norm, cname_norm)
            localization_pairs.append((img_path, concept_text, boxes))

        # detection sample entry
        detection_samples.append((img_path, cls_label, "", present_concepts))

    return localization_pairs, detection_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate Concept Attention (patch saliency) on MonuMAI")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to MonuMAI dataset root")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-detection", action="store_true", help="Skip detection AUROC computation")
    parser.add_argument("--no-localization", action="store_true", help="Skip localization metrics")
    parser.add_argument("--save-examples-dir", type=str, default=None, help="Directory to save example visualizations")
    parser.add_argument("--examples-per-class", type=int, default=5, help="Max examples to save per class")
    parser.add_argument("--trials", type=int, default=10, help="Number of balanced subsampling runs per scenario (paper default: 10)")
    parser.add_argument("--topk-percent", type=float, default=None, help="Top-P percent of pixels to set as positive (0<P<100). If omitted, uses mean threshold.")
    parser.add_argument("--debug-detection", action="store_true", help="Print per-scenario counts and skip reasons for detection")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Device:", device)
    print("Collecting dataset entries ...")
    localization_pairs, detection_samples = collect_dataset_pairs(args.dataset_root)
    print(f"Found {len(localization_pairs)} localization pairs and {len(detection_samples)} images with XML.")

    if not args.no_localization:
        print("Running localization evaluation (PixelAcc, mIoU, mAP) ...")
        loc_metrics = evaluate_localization(localization_pairs, args.model_name, device,
                                           save_examples_dir=args.save_examples_dir,
                                           examples_per_class=args.examples_per_class,
                                           topk_percent=args.topk_percent)
        print("Localization:", loc_metrics)

    if not args.no_detection:
        print("Running scenario-based detection AUROC (paper protocol: c1-present vs c1-absent, %d runs) ..." % args.trials)
        det_metrics = evaluate_detection_scenarios(
            detection_samples,
            args.model_name,
            device,
            trials=args.trials,
            debug=args.debug_detection,
            save_examples_dir=args.save_examples_dir,
            examples_per_class=args.examples_per_class,
        )
        print("Detection:", det_metrics)


if __name__ == "__main__":
    main()


