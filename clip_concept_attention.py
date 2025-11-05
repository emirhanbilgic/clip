import argparse
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import matplotlib.pyplot as plt


def get_module_attr_safe(module, attr_candidates):
    """
    Return first existing attr value from candidates (list of str).
    Raises AttributeError if none exist.
    """
    for a in attr_candidates:
        if hasattr(module, a):
            return getattr(module, a)
    raise AttributeError(f"None of attributes {attr_candidates} found in module {module}.")

def compute_patch_saliency_joint(model: CLIPModel, o_x: torch.Tensor, text_features: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Compute patch saliency by similarity in CLIP joint space:
    - Project per-patch vision features with visual/vision projection
    - L2-normalize and compute dot with normalized text_features
    - Softmax over patches
    Returns saliency per item as numpy array shape (N,)
    """
    # Resolve visual projection as weight matrix M with shape (hidden, joint)
    vision_proj_any = get_module_attr_safe(model, ["vision_projection", "visual_projection"])  # Linear or Tensor/Parameter
    if hasattr(vision_proj_any, "weight"):
        # (out=joint, in=hidden) -> transpose to (hidden, joint)
        M = vision_proj_any.weight.transpose(0, 1)
    else:
        M = vision_proj_any
        if not isinstance(M, torch.Tensor):
            M = torch.as_tensor(M)
        # Ensure orientation (hidden, joint)
        if M.dim() == 2 and M.shape[0] != o_x.shape[-1] and M.shape[1] == o_x.shape[-1]:
            M = M.transpose(0, 1)

    # Map per-patch features to joint space
    M = M.to(o_x.device)
    patch_joint = torch.einsum("bnd,dj->bnj", o_x, M)
    # Normalize
    patch_joint = F.normalize(patch_joint, dim=-1)
    text_n = F.normalize(text_features, dim=-1)
    # Similarity per patch
    sims = torch.einsum("bnj,bj->bn", patch_joint, text_n)
    if temperature != 1.0:
        sims = sims / temperature
    saliency = F.softmax(sims, dim=-1)
    return saliency.detach().cpu().numpy()


def visualize_saliency_on_image(img: Image.Image, saliency: np.ndarray, grid_size: Tuple[int, int], title: str = None, save_path: str = None, show: bool = True):
    """
    Visualize saliency (vector of length N patches) overlaid on image.
    grid_size: (h_patches, w_patches)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    img_arr = np.array(img).astype(np.float32) / 255.0
    h_p, w_p = grid_size
    N = h_p * w_p
    if saliency.size != N:
        raise ValueError(f"saliency length {saliency.size} != grid size product {N}")

    # reshape saliency to grid
    sal_map = saliency.reshape(h_p, w_p)
    # upsample to image size for overlay
    sal_map_up = np.kron(sal_map, np.ones((img_arr.shape[0] // h_p, img_arr.shape[1] // w_p)))

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax[0].imshow(img_arr)
    ax[0].axis("off")
    ax[0].set_title("Original image")

    # Heatmap overlay
    ax[1].imshow(img_arr)
    ax[1].imshow(sal_map_up, cmap="jet", alpha=0.45, norm=Normalize(vmin=sal_map_up.min(), vmax=sal_map_up.max()))
    ax[1].axis("off")
    ax[1].set_title(title or "Concept saliency overlay")
    
    # Grid with numerical scores
    ax[2].imshow(img_arr)
    ax[2].imshow(sal_map_up, cmap="jet", alpha=0.25, norm=Normalize(vmin=sal_map_up.min(), vmax=sal_map_up.max()))
    ax[2].axis("off")
    ax[2].set_title("Saliency scores (higher = more relevant)")
    
    # Overlay text scores on grid
    patch_h = img_arr.shape[0] / h_p
    patch_w = img_arr.shape[1] / w_p
    for i in range(h_p):
        for j in range(w_p):
            score = sal_map[i, j]
            # Position text at center of each patch
            y_center = (i + 0.5) * patch_h
            x_center = (j + 0.5) * patch_w
            # Use contrasting color based on score
            text_color = 'white' if score < (sal_map.max() + sal_map.min()) / 2 else 'black'
            ax[2].text(x_center, y_center, f'{score:.3f}', 
                      ha='center', va='center', fontsize=8, 
                      color=text_color, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='gray', alpha=0.5))
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\nSaliency Statistics:")
    print(f"  Min score: {sal_map.min():.4f}")
    print(f"  Max score: {sal_map.max():.4f}")
    print(f"  Mean score: {sal_map.mean():.4f}")
    print(f"  Std dev: {sal_map.std():.4f}")
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved visualization to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load CLIP model and processor
    model_name = args.model_name
    print(f"Loading CLIP model '{model_name}' ...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load image
    img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Text features in joint space
    text_inputs = processor(text=args.concept, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)  # (1, joint_dim)
    text_features = text_features.detach()

    # Per-patch features from final vision layer
    with torch.no_grad():
        vis_outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
    o_x = vis_outputs.last_hidden_state  # (1, 1+N, D)
    if o_x.shape[1] > 1:
        o_x_for_sal = o_x[:, 1:, :]
    else:
        o_x_for_sal = o_x

    # Joint-space saliency
    saliency = compute_patch_saliency_joint(model, o_x_for_sal, text_features)
    saliency_vec = saliency[0]

    # Infer patch grid size
    patch_size = getattr(model.vision_model.config, "patch_size", 32)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    image_size = getattr(model.vision_model.config, "image_size", 224)
    h_p = image_size // patch_size
    w_p = image_size // patch_size
    if h_p * w_p != saliency_vec.size:
        s = int(math.sqrt(int(saliency_vec.size)))
        if s * s == saliency_vec.size:
            h_p = w_p = s

    # Visualize
    visualize_saliency_on_image(
        img,
        saliency_vec,
        (h_p, w_p),
        title=f"Saliency for concept: '{args.concept}'",
        save_path=args.out_fig,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept saliency for CLIP (joint-space over final vision layer)")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model name")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--concept", type=str, required=True, help="Text concept (e.g. 'dog', 'red car')")
    parser.add_argument("--out-fig", type=str, default=None, help="Path to save visualization image (e.g., .png)")
    args = parser.parse_args()
    main(args)

