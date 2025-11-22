import os
import sys
import argparse
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src import get_model, setup_logging, get_simple_augs

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    model_name_lower = model_name.lower()
    if "resnet" in model_name_lower:
        return model.layer4[-1]
    elif "efficientnet" in model_name_lower:
        return model.features[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def generate_layer_cam(model: nn.Module, target_layer: nn.Module, 
                    input_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    activations = []
    
    def hook(module, input, output):
        activations.append(output)
    
    handle = target_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        output = model(input_tensor.to(device))
    
    handle.remove()
    
    pred_class_idx = output.argmax(dim=1).item()
    layer_activations = activations[0].squeeze(0)
    
    heatmap = torch.zeros(layer_activations.shape[1:], device=device)
    upsample_for_mask = nn.Upsample(
        size=input_tensor.shape[2:], 
        mode='bilinear', 
        align_corners=False
    )
    
    for i in range(layer_activations.shape[0]):
        activation_map = layer_activations[i].unsqueeze(0).unsqueeze(0)
        mask = upsample_for_mask(activation_map)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        with torch.no_grad():
            masked_input = input_tensor.to(device) * mask
            masked_output = model(masked_input)
            weight = torch.nn.functional.softmax(masked_output, dim=1)[0][pred_class_idx]
        
        heatmap += layer_activations[i] * weight
    
    heatmap = torch.relu(heatmap)
    final_heatmap = nn.functional.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0), 
        size=input_tensor.shape[2:], 
        mode='bilinear', 
        align_corners=False
    ).squeeze()
    
    final_heatmap = (final_heatmap - final_heatmap.min()) / \
                    (final_heatmap.max() - final_heatmap.min() + 1e-8)
    
    return final_heatmap.cpu().numpy()

def load_model(model_path: str, model_name: str, num_classes: int, 
            device: torch.device) -> nn.Module:
    print(f"Loading model from {model_path}...")
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    return model

def get_example_images(model_name: str) -> tuple[str, ...]:
    return {
        'resnet18': ("train/Image_10387.jpg", "test/Image_1129.jpg", "train/Image_1038.jpg"),
        'efficientnet_b2': ("train/Image_10613.jpg", "train/Image_3473.jpg", "train/Image_141.jpg"),
    }[model_name.lower()]

def visualize_layer_cam_comparison(
    models: list[nn.Module],
    model_name: str,
    image_paths: list[str],
    output_path: str,
    device: torch.device,
    version_labels: list[str] = ['V1', 'V2', 'V3'],
    figsize: tuple[int, int] = (15, 10)
) -> None:
    crop_size, resize_size = {
        "resnet18": (224, 256),
        "efficientnet_b2": (260, 260),
    }[model_name.lower()]
    
    transform = get_simple_augs(crop_size, resize_size)
    
    n_images = len(image_paths)
    n_versions = len(models)
    
    fig, axes = plt.subplots(
        nrows=n_images,
        ncols=n_versions,
        figsize=figsize,
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
    )
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for col_idx, (model, version_label) in enumerate(zip(models, version_labels)):
        print(f"\nProcessing {version_label}...")
        
        target_layer = get_target_layer(model, model_name)
        
        for row_idx, image_path in enumerate(image_paths):
            print(f"  Image: {os.path.basename(image_path)}")
            ax = axes[row_idx, col_idx]
            
            original_image = Image.open(image_path).convert("RGB")
            input_tensor = transform(original_image).unsqueeze(0)
            
            heatmap = generate_layer_cam(model, target_layer, input_tensor, device)
            
            ax.imshow(original_image.resize((crop_size, crop_size)))
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            
            if row_idx == 0:
                ax.set_title(version_label, fontsize=12, fontweight='bold')
            
            ax.axis('off')
            ax.set_aspect('auto')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    plt.suptitle(f'LayerCAM - {model_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"\nLayerCAM - {model_name} saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate LayerCAM visualizations comparing model versions')
    parser.add_argument('--model-v1-path', type=str, required=True, help='Path to V1 model checkpoint (.pth file)')
    parser.add_argument('--model-v2-path', type=str, required=True, help='Path to V2 model checkpoint (.pth file)')
    parser.add_argument('--model-v3-path', type=str, required=True, help='Path to V3 model checkpoint (.pth file)')
    parser.add_argument('--model-name', type=str, required=True, choices=['resnet18', 'efficientnet_b2'], help='Model architecture name')
    parser.add_argument('--num-classes', type=int, default=15, help='Number of output classes')
    parser.add_argument('--images-dir', type=str, default='/kaggle/input/human-action-recognition-har-dataset/Human Action Recognition')
    parser.add_argument('--output-path', type=str, default='/kaggle/working/layercam.png')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--figsize', type=int, nargs=2, default=(15, 10), metavar=('WIDTH', 'HEIGHT'), help='Figure size (width height)')
    parser.add_argument('--log-file', type=str, default='/kaggle/working/LayerCAM.log')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    image_names = get_example_images(args.model_name)
    image_paths = [os.path.join(args.images_dir, name) for name in image_names]
    
    print(f"Using images: {image_names}")
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
    
    print("\nLoading model versions...")
    models = []
    model_paths = [args.model_v1_path, args.model_v2_path, args.model_v3_path]
    version_labels = ['V1', 'V2', 'V3']
    
    for path, label in zip(model_paths, version_labels):
        print(f"  Loading {label} from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        
        model = load_model(path, args.model_name, args.num_classes, device)
        models.append(model)
    
    visualize_layer_cam_comparison(
        models=models,
        model_name=args.model_name,
        image_paths=image_paths,
        output_path=args.output_path,
        device=device,
        version_labels=version_labels,
        figsize=tuple(args.figsize)
    )
    
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nLayerCAM comparison completed successfully!")

if __name__ == '__main__':
    main()
