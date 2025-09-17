import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
)
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src import get_model, setup_logging

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Compute CKA similarity for pre-trained and fine-tuned models.")
    parser.add_argument('--num_classes', type=int, default=15, help='Number of output classes for fine-tuned models.')
    parser.add_argument('--num_cka_chunks', type=int, default=3, help='Number of chunks to split dataset for CKA computation.')
    parser.add_argument('--data_dir', type=str, default="/kaggle/input/ilsvrc2012-img-val-subset/ILSVRC2012_img_val_subset",
                        help='Path to ImageNet validation subset directory.')
    parser.add_argument('--output_csv_path', type=str, default="/kaggle/working/models_cka_results.csv",
                        help='Path to save CKA results CSV.')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to run models (cuda or cpu).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers.')
    parser.add_argument('--cka_experiments', type=str, required=True,
                        help='JSON string defining experiments, e.g., [{"model_name": "resnet18", "version": "V1", "path": "..."}]')
    parser.add_argument('--log_file', type=str, default='/kaggle/working/cka_log.log', help='Path to save log file.')
    return parser.parse_args()

# CKA Calculation Function
def calculate_cka_for_chunk(
    model1: nn.Module, model2: nn.Module, dataloader: DataLoader,
    layers_to_hook: List[str], device: torch.device
) -> Dict[str, float]:
    """Computes the CKA similarity for a given chunk of data."""
    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()

    activations1 = {name: [] for name in layers_to_hook}
    activations2 = {name: [] for name in layers_to_hook}

    def get_activation(name, store):
        def hook(model, input, output):
            store[name].append(output.detach().cpu())
        return hook

    hook_handles1, hook_handles2 = [], []
    try:
        for name in layers_to_hook:
            try:
                module1 = dict(model1.named_modules())[name]
                handle1 = module1.register_forward_hook(get_activation(name, activations1))
                hook_handles1.append(handle1)

                module2 = dict(model2.named_modules())[name]
                handle2 = module2.register_forward_hook(get_activation(name, activations2))
                hook_handles2.append(handle2)
            except KeyError:
                logging.warning(f"Layer '{name}' not found. Skipping.")
                continue

        with torch.no_grad():
            for images, _ in dataloader:
                _ = model1(images.to(device))
                _ = model2(images.to(device))

    finally:
        for handle in hook_handles1 + hook_handles2:
            handle.remove()

    for name in layers_to_hook:
        if activations1.get(name):
            activations1[name] = torch.cat(activations1[name], dim=0)
        if activations2.get(name):
            activations2[name] = torch.cat(activations2[name], dim=0)

    def center_gram(X):
        X = X.reshape(X.shape[0], -1)
        X = X - X.mean(dim=0)
        return X @ X.T

    def linear_CKA(X, Y):
        if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor) or X.numel() == 0 or Y.numel() == 0:
            return 0.0
        K = center_gram(X)
        L = center_gram(Y)
        hsic = (K * L).sum()
        norm_x = torch.norm(K)
        norm_y = torch.norm(L)
        return (hsic / (norm_x * norm_y)).item() if norm_x > 0 and norm_y > 0 else 0.0

    results = {}
    for name in layers_to_hook:
        act1 = activations1.get(name)
        act2 = activations2.get(name)
        if act1 is not None and act2 is not None:
            results[name] = linear_CKA(act1, act2)
        else:
            results[name] = 0.0
    return results

# Main Execution
if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file=args.log_file)

    NUM_CLASSES = args.num_classes
    NUM_CKA_CHUNKS = args.num_cka_chunks
    DATA_DIR = args.data_dir
    OUTPUT_CSV_PATH = args.output_csv_path
    DEVICE = torch.device(args.device)
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    try:
        CKA_EXPERIMENTS = json.loads(args.cka_experiments)
    except json.JSONDecodeError:
        logging.error("Invalid JSON for --cka_experiments. Exiting.")
        sys.exit(1)

    logging.info(f"Starting CKA analysis for {len(CKA_EXPERIMENTS)} experiments.")
    logging.info(f"Using device: {DEVICE}")

    # --- 1. Prepare Dataset ---
    imagenet_transforms = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_dataset = ImageFolder(root=DATA_DIR, transform=imagenet_transforms)
    index_chunks = np.array_split(range(len(imagenet_dataset)), NUM_CKA_CHUNKS)
    logging.info(f"Loaded ImageNet subset with {len(imagenet_dataset)} images, split into {NUM_CKA_CHUNKS} chunks.")

    all_results = []

    # 2. Loop Through Experiments
    for exp in CKA_EXPERIMENTS:
        model_name = exp['model_name']
        version = exp['version']
        checkpoint_path = exp['path']
        full_name = f"{model_name}_{version}"
        logging.info(f"--- Processing Experiment: {full_name} ---")

        # 2.1. Load Models
        logging.info("Loading pretrained and finetuned models...")
        # Load finetuned model
        finetuned_model = get_model(model_name, num_classes=NUM_CLASSES)
        try:
            finetuned_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        except FileNotFoundError:
            logging.error(f"Checkpoint not found at {checkpoint_path}. Skipping this experiment.")
            continue

        # Load the corresponding pre-trained reference model
        model_constructor, weights = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
        }[model_name]
        pretrained_model = model_constructor(weights=weights)

        # 2.2. Define Layers to Hook
        if 'resnet' in model_name:
            layers_to_hook = ['layer1', 'layer2', 'layer3', 'layer4']
        else:  # efficientnet
            layers_to_hook = ['features.4', 'features.5', 'features.6', 'features.7']
        logging.info(f"Hooking layers: {layers_to_hook}")

        # 2.3. Calculate CKA over all chunks
        exp_cka_scores = {name: [] for name in layers_to_hook}
        for i, chunk_indices in enumerate(index_chunks):
            logging.info(f" Processing CKA chunk {i+1}/{NUM_CKA_CHUNKS}...")
            chunk_subset = Subset(imagenet_dataset, chunk_indices)
            chunk_loader = DataLoader(chunk_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

            chunk_scores = calculate_cka_for_chunk(
                pretrained_model, finetuned_model, chunk_loader, layers_to_hook, DEVICE
            )
            for name, score in chunk_scores.items():
                exp_cka_scores[name].append(score)

        # 2.4. Aggregate and Store Results
        avg_cka_scores = {name: np.mean(scores) for name, scores in exp_cka_scores.items()}

        result_row = {'model_name': full_name}
        result_row.update(avg_cka_scores)
        all_results.append(result_row)

        logging.info(f"Average CKA Scores for {full_name}: {avg_cka_scores}")

        # 2.5. Clean up memory
        del finetuned_model, pretrained_model, avg_cka_scores
        if 'cuda' in str(DEVICE):
            torch.cuda.empty_cache()
        logging.info("Cleaned up CUDA memory.")

    # 3. Save Final Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        all_layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'features.4', 'features.5', 'features.6', 'features.7']
        ordered_cols = ['model_name'] + [col for col in all_layer_names if col in results_df.columns]
        results_df = results_df[ordered_cols]

        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        logging.info(f"Successfully saved CKA results to {OUTPUT_CSV_PATH}")
    else:
        logging.warning("No results were generated. CSV file not created.")

    logging.info("CKA analysis complete.")
