import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
 
# --- Add project root to sys.path ---
sys.path.append(os.getcwd())
from src import config
from src.dataset import HumanActivityDataset
from src.data import get_simple_augs
from src.model import get_model
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
 
# --- Gradient-free LayerCAM ---
class GradFreeLayerCAM:
    """ Gradient-free LayerCAM via channel-weighted activation maps. """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._hook = target_layer.register_forward_hook(self._fwd_hook)
 
    def _fwd_hook(self, module, input, output):
        self._activations = output.detach()
 
    def remove_hook(self):
        self._hook.remove()
 
    @torch.no_grad()
    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Returns (H, W) numpy array normalised to [0, 1]."""
        self._activations = None
        _ = self.model(input_tensor)
        acts = self._activations
        weights = acts.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

# --- Target-layer resolution ---
def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Returns the last fine-tuned conv block."""
    if "resnet" in model_name.lower():
        return model.layer4
    elif "eff" in model_name.lower():
        return model.features[7]
    raise ValueError(f"Unknown model_name: {model_name}")

# --- FG / BG metric ---
def compute_fg_bg_ratio(cam: np.ndarray, threshold: float = 0.4) -> dict:
    """ Splits the CAM at `threshold` into foreground / background regions. """
    eps = 1e-8
    fg_mask = cam >= threshold
    bg_mask = ~fg_mask
    fg_mean = float(cam[fg_mask].mean()) if fg_mask.any() else 0.0
    bg_mean = float(cam[bg_mask].mean()) if bg_mask.any() else eps

    return {
        "fg_mean": fg_mean,
        "bg_mean": bg_mean,
        "fg_bg_ratio": fg_mean / (bg_mean + eps),
        "fg_coverage": float(fg_mask.mean()),
    }

# --- Dataset builder ---
def build_val_loader(sample_indices: np.ndarray, model_name: str) -> DataLoader:
    """ Builds a fresh HumanActivityDataset + Subset + DataLoader for each experiment. """
    crop_size, resize_size = config.IMAGE_SIZE_REGISTRY[model_name]
    transform = get_simple_augs(crop_size, resize_size)
    dataset = HumanActivityDataset(
        config.HAR_CSV_PATH, config.HAR_TRAIN_DIR, transform=transform
    )
    subset = Subset(dataset, sample_indices.tolist())
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
 
# --- Per-model evaluation ---
def evaluate_model(
    model_name: str,
    version: str,
    weights_path: str,
    sample_indices: np.ndarray,
    num_classes: int,
    device: torch.device,
    cam_threshold: float = 0.4,
) -> list:
    """Loads checkpoint, runs GradFreeLayerCAM, returns per-image metric dicts."""
    logging.info(f"  Loading {model_name} {version} from {weights_path}")
    dropout_p = 0.3
    for exp in config.EXPERIMENTS.values():
        if exp["model_name"] == model_name:
            dropout_p = exp.get("dropout_p", 0.3)
            break
    model = get_model(model_name, num_classes, dropout_p=dropout_p)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    target_layer = get_target_layer(model, model_name)
    extractor = GradFreeLayerCAM(model, target_layer)
    val_loader = build_val_loader(sample_indices, model_name)
    records = []
    for images, labels in val_loader:
        inp = images.to(device)
        cam = extractor(inp)
        metrics = compute_fg_bg_ratio(cam, threshold=cam_threshold)
        metrics.update({
            "model_name": model_name,
            "version":    version,
            "label":      labels[0].item(),
        })
        records.append(metrics)
 
    extractor.remove_hook()
    logging.info(f"    → {len(records)} samples processed.")
    return records
 
# --- Argument Parser ---
def parse_args():
    p = argparse.ArgumentParser(description="LayerCAM FG/BG Activation Ratio Quantification")
    p.add_argument("--resnet-v1-path", required=True)
    p.add_argument("--resnet-v2-path", required=True)
    p.add_argument("--resnet-v3-path", required=True)
    p.add_argument("--effnet-v1-path", required=True)
    p.add_argument("--effnet-v2-path", required=True)
    p.add_argument("--effnet-v3-path", required=True)
    p.add_argument("--num-samples", type=int,   default=200)
    p.add_argument("--cam-threshold", type=float, default=0.4)
    p.add_argument("--device", type=str,   default=None)
    p.add_argument("--output-csv", default="/kaggle/working/layercam_fg_bg_ratio.csv")
    args, _ = p.parse_known_args()
    return args

# --- Main ---
def main():
    args = parse_args() 
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"Using device: {device}")
 
    # --- 1. Derive sample indices ONCE from a fixed seed ---
    _ref = HumanActivityDataset(config.HAR_CSV_PATH, config.HAR_TRAIN_DIR, transform=None)
    num_classes = len(_ref.classes)
    all_indices = list(range(len(_ref)))
    del _ref
 
    _, val_indices = train_test_split(all_indices, test_size=0.2, random_state=config.SEED)
 
    rng = np.random.default_rng(config.SEED)
    sample_indices = rng.choice(
        val_indices, size=min(args.num_samples, len(val_indices)), replace=False
    )
    logging.info(f"Evaluating on {len(sample_indices)} fixed validation samples per model.")
 
    # --- 2. Loop through all 6 experiments ---
    experiments = [
        ("resnet18", "V1", args.resnet_v1_path),
        ("resnet18", "V2", args.resnet_v2_path),
        ("resnet18", "V3", args.resnet_v3_path),
        ("efficientnet_b2", "V1", args.effnet_v1_path),
        ("efficientnet_b2", "V2", args.effnet_v2_path),
        ("efficientnet_b2", "V3", args.effnet_v3_path),
    ]
 
    all_records = []
    for model_name, version, weights_path in experiments:
        logging.info(f"Processing {model_name} {version}...")
        records = evaluate_model(
            model_name=model_name,
            version=version,
            weights_path=weights_path,
            sample_indices=sample_indices,
            num_classes=num_classes,
            device=device,
            cam_threshold=args.cam_threshold,
        )
        all_records.extend(records)
 
    # --- 3. Save CSV ---
    df = pd.DataFrame(all_records)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Raw results saved to {args.output_csv}")
 
 
if __name__ == "__main__":
    main()