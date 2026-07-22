"""
V8 Dataset Module for NCCT Brain Stroke Segmentation.

Extends V7 NCCTDataset with:
  - Signed distance transform maps (for Boundary Loss)
  - Raw image tensors (for rHU / NWU Loss)
  - Updated visualization and dataloader utilities
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
#  Distance transform helpers
# ---------------------------------------------------------------------------

def compute_signed_distance_transform(mask_np, normalize=True):
    """
    Compute signed distance transform from a binary mask.

    Args:
        mask_np:  numpy array (H, W), 0 = background, 1 = foreground.
        normalize: If True, normalise to [-1, 1] range.

    Returns:
        sdt: Signed distance transform (positive inside object,
             negative outside).
    """
    mask_int = (mask_np > 0.5).astype(np.uint8)

    # Distance from foreground boundary (positive inside)
    dist_in = distance_transform_edt(mask_int)
    # Distance from background boundary (positive outside)
    dist_out = distance_transform_edt(1 - mask_int)

    # Signed distance: positive inside, negative outside
    sdt = dist_in - dist_out

    if normalize:
        # Normalize by half the image size to avoid scale dependency
        max_dist = max(mask_np.shape) / 2.0
        sdt = np.clip(sdt / max_dist, -1.0, 1.0)

    return sdt


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class NCCTDatasetV8(Dataset):
    """Enhanced dataset for V8 with distance maps and raw image tensors.

    Each sample returns a 5-tuple:
        (image, mask, dist_map, label, raw_img)
    """

    def __init__(self, split_dir, limit_size=None, transform=None,
                 return_raw=True):
        """
        Args:
            split_dir:  Path to a split folder containing ``images/`` and
                        ``masks/``.
            limit_size: If set, only use the first N samples (for debugging).
            transform:  Albumentations composition (spatial + tensor
                        conversion).  Should include ``ToTensorV2()``.
            return_raw: If True, the 5th output element is a copy of the
                        normalised image tensor (proxy for rHU loss).
                        If False, a zero placeholder is returned.
        """
        super().__init__()
        self.split_dir = split_dir
        self.transform = transform
        self.return_raw = return_raw

        img_dir = os.path.join(split_dir, "images")
        mask_dir = os.path.join(split_dir, "masks")

        self.image_paths = sorted(
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
        )
        self.mask_paths = sorted(
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
        )

        if limit_size is not None:
            self.image_paths = self.image_paths[:limit_size]
            self.mask_paths = self.mask_paths[:limit_size]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load PIL images ---
        img = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # --- Apply transforms ---
        if self.transform is not None:
            img_np = np.array(img).astype(np.uint8)
            mask_np = np.array(mask).astype(np.uint8)
            augmented = self.transform(image=img_np, mask=mask_np)
            # ToTensorV2 produces (C, H, W) tensors and divides uint8 by 255
            img_tensor = augmented["image"]           # (1, H, W), ~[0, 1]
            mask_tensor = augmented["mask"].float()   # (1, H, W), ~[0, 1]
        else:
            img_np = np.array(img).astype(np.float32)
            mask_np = np.array(mask).astype(np.float32)
            img_tensor = torch.from_numpy(img_np).unsqueeze(0) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) / 255.0

        # --- Binarise mask ---
        mask_bin_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_bin_np).float()

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # --- Compute signed distance transform ---
        sdt = compute_signed_distance_transform(
            mask_bin_np.squeeze(), normalize=True
        )
        dist_tensor = torch.from_numpy(sdt).unsqueeze(0).float()

        # --- Output elements ---
        # raw_img: proxy for rHU / NWU loss (uses normalised intensities)
        if self.return_raw:
            raw_img = img_tensor.clone()
        else:
            raw_img = torch.tensor([[0.0]])

        # Label: 1 if any lesion present, else 0
        label = torch.tensor(
            [1.0 if mask_tensor.max() > 0 else 0.0], dtype=torch.float32
        )

        return img_tensor, mask_tensor, dist_tensor, label, raw_img


# ---------------------------------------------------------------------------
#  Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloaders_v8(base_dir, batch_size=4, limit_size=None):
    """Create train / val / test DataLoaders for the V8 dataset.

    Args:
        base_dir:   Root directory containing ``train/``, ``val/``, ``test/``.
        batch_size: Batch size for all loaders.
        limit_size: If set, limit each split to this many samples.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])

    train_dataset = NCCTDatasetV8(
        os.path.join(base_dir, "train"),
        limit_size=limit_size,
        transform=train_transform,
        return_raw=True,
    )
    val_dataset = NCCTDatasetV8(
        os.path.join(base_dir, "val"),
        limit_size=limit_size,
        transform=val_transform,
        return_raw=True,
    )
    test_dataset = NCCTDatasetV8(
        os.path.join(base_dir, "test"),
        limit_size=limit_size,
        transform=val_transform,
        return_raw=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
#  Visualisation
# ---------------------------------------------------------------------------

def visualize_v8_sample(images, masks, dist_maps, raw_images, num_samples=3):
    """Visualise dataset samples alongside their distance maps.

    Args:
        images:      Tensor of shape (B, 1, H, W) – normalised input.
        masks:       Tensor of shape (B, 1, H, W) – binary ground truth.
        dist_maps:   Tensor of shape (B, 1, H, W) – signed distance.
        raw_images:  Tensor of shape (B, 1, H, W) – raw (proxy) image.
        num_samples: Number of samples to display (default 3).
    """
    num_samples = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(num_samples, 4,
                             figsize=(16, 4 * num_samples))

    # Handle single-row case where axes is 1-D
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        axes[i, 0].imshow(images[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 0].set_title("Input Image (norm)")

        axes[i, 1].imshow(masks[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(dist_maps[i, 0].cpu().numpy(),
                          cmap="RdBu", vmin=-1, vmax=1)
        axes[i, 2].set_title("Signed Distance")

        axes[i, 3].imshow(raw_images[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 3].set_title("Raw Image")

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()
