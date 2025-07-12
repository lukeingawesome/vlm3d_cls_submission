#!/usr/bin/env python3
# coding: utf-8
"""
Process CT volumes for abnormality classification
"""

from __future__ import annotations
import argparse, logging, os, sys, json, warnings, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import SimpleITK as sitk

# Local imports
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "training"))
from training.ct_transform import get_val_transform
from merlin import Merlin

# ---------------------------------------------------------------------------
#  CT on‑the‑fly pre‑processing (same settings you used for offline *.npz)
# ---------------------------------------------------------------------------
from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import EnsureChannelFirst, CastToType
from monai.transforms.spatial.array import Orientation, Spacing
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.transforms.croppad.array import SpatialPad, CenterSpatialCrop

TARGET_SPACING = (1.25, 1.25, 2.0)            # mm
TARGET_SHAPE   = (256, 256, 192)              # H, W, D
HU_WINDOW      = (-1000, 1500)
NPZ_DTYPE      = np.float16                   # storage only

ct_load_pipeline = Compose([
    LoadImage(image_only=True, dtype=np.float32),      # (Z, Y, X)
    EnsureChannelFirst(),                              # (1, Z, Y, X)
    Orientation(axcodes="RAS"),
    Spacing(pixdim=TARGET_SPACING,
            mode="trilinear", align_corners=True),
    ScaleIntensityRange(a_min=HU_WINDOW[0], a_max=HU_WINDOW[1],
                        b_min=0.0, b_max=1.0, clip=True),
    SpatialPad(spatial_size=(*TARGET_SHAPE,)),
    CenterSpatialCrop(roi_size=(*TARGET_SHAPE,)),
    CastToType(dtype=np.float16),
])

SUPPORTED_RAW_EXTS = (".nii", ".nii.gz", ".mha")
SUPPORTED_COMPRESSED_EXTS = (".npz",)          # keep this for backward compat

# Default 18 pathology labels
DEFAULT_LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening"
]

# ───────────────────────────────────────────────────────
def _hu_window_to_unit(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    """Clip a HU volume to the given window and scale to [0,1]."""
    lower, upper = center - width / 2.0, center + width / 2.0
    vol = np.clip(volume, lower, upper)
    return (vol - lower) / (upper - lower)

def preprocess_ct_volume(path, val_transform, use_3channel=False):
    """
    Accept *.npz  OR  raw *.nii / *.nii.gz / *.mha.
    Returns a 5‑D tensor (1, C, D, H, W) in fp32 ready for the model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # ────────────────── 1) already‑compressed *.npz ──────────────────
    if path.suffix.lower() in SUPPORTED_COMPRESSED_EXTS:
        with np.load(str(path)) as npz:
            if "image" not in npz:
                raise KeyError(f"'image' key missing in {path}")
            arr = npz["image"]                         # (C, H, W, D) fp16/32
        img = torch.from_numpy(arr).float()            # → fp32
    # ────────────────── 2) raw NIfTI / MetaImage ────────────────────
    elif path.name.lower().endswith(SUPPORTED_RAW_EXTS):
        vol = ct_load_pipeline(str(path))              # (1, Z, Y, X) fp16
        # Convert MetaTensor to tensor directly
        try:
            # Try to convert to tensor first (handles MetaTensor and other types)
            img = torch.as_tensor(vol).float()         # (1, D, H, W)
        except:
            # Fallback to numpy conversion
            img = torch.from_numpy(np.array(vol)).float()  # (1, D, H, W)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")

    # -------- optional three‑window conversion -----------------------
    # if use_3channel:
    #     hu = img[0] * (HU_WINDOW[1] - HU_WINDOW[0]) + HU_WINDOW[0]
    #     lung = torch.clamp((hu + 600) / 1000, 0, 1)    # centre –600, width 1000
    #     medi = torch.clamp((hu - 40)  /  400, 0, 1)    # centre   40, width  400
    #     bone = torch.clamp((hu - 700) / 1500, 0, 1)    # centre  700, width 1500
    #     img = torch.stack([lung, medi, bone], dim=0)   # (3, D, H, W)

    # ----------------------------------------------------------------
    # Apply the *runtime* val_transform you already use for CLIP
    # (e.g. random crop / resize. It expects a torch tensor.)
    if val_transform is not None:
        img = val_transform(img)

    # Add batch dim expected by encode_image: (B, C, D, H, W)
    if img.dim() == 4:
        img = img.unsqueeze(0)
    return img

def find_volumes_in_directory(input_dir: str) -> List[str]:
    """Find all .mha and .npz files in input directory."""
    input_path = Path(input_dir)
    volume_files = []
    
    # Look for supported file types
    for ext in SUPPORTED_COMPRESSED_EXTS + SUPPORTED_RAW_EXTS:
        volume_files.extend(input_path.glob(f"*{ext}"))
    
    logging.info(f"Found {len(volume_files)} volumes in {input_dir}")
    return sorted([str(f) for f in volume_files])

class CTDataset(Dataset):
    """Dataset for processing CT volumes for inference."""
    def __init__(self, volume_paths: List[str], transform=None, three_ch: bool = False):
        self.volume_paths = volume_paths
        self.transform = transform
        self.three_ch = three_ch
        logging.info(f"Created dataset with {len(volume_paths)} volumes")

    def __len__(self): 
        return len(self.volume_paths)

    def __getitem__(self, idx: int):
        volume_path = self.volume_paths[idx]
        filename = Path(volume_path).stem  # filename without extension
        
        try:
            # Use the new preprocessing pipeline
            img = preprocess_ct_volume(volume_path, val_transform=self.transform, use_3channel=self.three_ch)
            # Remove batch dimension as it will be added by DataLoader
            if img.dim() == 5:
                img = img.squeeze(0)
                
            return img, filename
            
        except Exception as e:
            logging.error(f"Failed to process {volume_path}: {e}")
            raise

# ───────────────────────────────────────────────────────
class SigLIPClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, drop: float = .1):
        super().__init__()
        self.backbone = backbone
        dim = getattr(backbone, "output_dim", 2048)
        self.head = nn.Sequential(
            nn.Dropout(drop), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(dim // 2, n_classes)
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)

# ───────────────────────────────────────────────────────
def load_model_from_checkpoint(checkpoint_path: str, labels: List[str], device: torch.device):
    """Load model from checkpoint."""
    logging.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Create model
    backbone = Merlin(ImageEmbedding=True)
    model = SigLIPClassifier(backbone, len(labels), drop=0.1)
    
    # Load state dict
    state_dict = checkpoint["model"]
    
    # Handle DDP prefix
    if any(key.startswith('module.') for key in state_dict.keys()):
        logging.info("Detected DDP checkpoint, removing 'module.' prefix...")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    # Remove EMA-related keys that might cause issues
    keys_to_remove = ['n_averaged']
    for key in keys_to_remove:
        if key in state_dict:
            logging.info(f"Removing unexpected key '{key}' from state_dict")
            del state_dict[key]
    
    # Try to load with strict=True first
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info("Model loaded successfully with strict=True")
    except Exception as e:
        logging.warning(f"Failed to load with strict=True: {e}")
        # Try with strict=False and report what's missing/unexpected
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing keys ({len(missing_keys)}): {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            
            logging.info("Model loaded with strict=False")
        except Exception as e2:
            logging.error(f"Failed to load model even with strict=False: {e2}")
            raise RuntimeError(f"Could not load model weights: {e2}")
    
    model.to(device)
    model.eval()
    
    logging.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    return model

# ───────────────────────────────────────────────────────
@torch.inference_mode()
def process_volumes(model, loader, device, labels):
    """Process volumes and generate predictions."""
    model.eval()
    predictions = []
    
    for batch_idx, (x, filenames) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # Convert to list if single item
        if isinstance(filenames, str):
            filenames = [filenames]
        
        # Process each volume in the batch
        for i, filename in enumerate(filenames):
            prob_dict = {}
            for j, label in enumerate(labels):
                prob_dict[label] = float(probs[i, j])
            
            predictions.append({
                "input_image_name": filename,
                "scores": prob_dict
            })
            
        logging.info(f"Processed batch {batch_idx + 1}, volumes: {len(filenames)}")
    
    return predictions

# ───────────────────────────────────────────────────────
def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Process CT volumes for abnormality classification")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", required=True, help="Directory containing input volumes")
    parser.add_argument("--output_path", required=True, help="Output JSON file path")
    parser.add_argument("--device", default="auto", help="Device to use (auto-detects CUDA if available, use 'cpu' to force CPU)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--three_channel", action="store_true", help="Use three-channel windowing")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Setup device - automatically use CUDA if available
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
        logging.info(f"Using device: {device} (forced by user)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using device: {device} (auto-detected)")
    else:
        device = torch.device("cpu")
        logging.info(f"Using device: {device} (CUDA not available)")
    
    # Find volumes in input directory
    volume_paths = find_volumes_in_directory(args.input_dir)
    if not volume_paths:
        logging.error(f"No volumes found in {args.input_dir}")
        return
    
    # Create dataset
    transform = get_val_transform()
    dataset = CTDataset(volume_paths, transform, args.three_channel)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, DEFAULT_LABELS, device)
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                       num_workers=0, pin_memory=True)
    
    # Process volumes
    logging.info("Starting inference...")
    predictions = process_volumes(model, loader, device, DEFAULT_LABELS)
    
    # Create output structure
    output = {
        "name": "Generated probabilities",
        "type": "Abnormality Classification",
        "version": {"minor": 0.1},
        "predictions": predictions
    }
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Final summary
    end_time = time.time()
    processing_time = end_time - start_time
    
    logging.info(f"Processed {len(predictions)} volumes in {processing_time:.2f} seconds")
    logging.info(f"Average time per volume: {processing_time/len(predictions):.2f} seconds")
    logging.info(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main() 