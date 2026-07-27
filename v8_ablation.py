"""
v8_ablation.py — Ablation Study for Physics-Informed Loss Functions
=====================================================================
NCCT Brain Stroke Segmentation V8

Quantifies the individual and joint contribution of each physics-informed
loss component through a structured ablation experiment.

Ablation Design (10 configurations, fixed UNet+ResNet34 architecture):
──────────────────────────────────────────────────────────────────────
  Add-one-in (marginal contribution over baseline):
    1. Baseline       (Focal + Tversky only)
    2. +ACE           (curvature-aware regularization)
    3. +Boundary      (distance transform boundary refinement)
    4. +NWU           (Net Water Uptake / CT physics)
    5. +Symmetry      (bilateral anatomical prior)
    6. Full           (all 4 components)

  Leave-one-out (redundancy / synergy in full model):
    7. Full − ACE
    8. Full − Boundary
    9. Full − NWU
   10. Full − Symmetry

Metrics: Dice, IoU, HD95, Sensitivity, Specificity, Precision
Reproducibility: Comprehensive seed management (PyTorch, NumPy, Python,
                 CuDNN deterministic mode)

References:
    Chen et al. "Learning Euler's Elastica Model" — IEEE TMI 2022
    Kervadec et al. "Boundary loss" — MIDL 2019 / MedIA 2021
    Broocks et al. "Net water uptake" — 2022–2025
    Ni et al. "Asymmetry Disentanglement Network" — MICCAI 2022
"""

import os
import sys
import random
import time
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Project imports
# ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v8_losses import ACELoss, BoundaryLoss, NetWaterUptakeLoss, BilateralSymmetryLoss
from v8_dataset import get_dataloaders_v8

# ══════════════════════════════════════════════════════════════════════
# 1.  Reproducibility
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducible training.

    Controls: Python ``random``, NumPy, PyTorch (CPU + CUDA),
    CuDNN deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ══════════════════════════════════════════════════════════════════════
# 2.  Configurable Ablation Loss
# ══════════════════════════════════════════════════════════════════════

class AblationNCCTLoss(nn.Module):
    """Configurable multi-component loss for ablation studies.

    Each physics-informed component can be toggled on/off via its λ weight.
    Setting λ=0 disables the component entirely.

    Components:
        * Base: Weighted Focal Loss + Tversky Loss (always active)
        * ACE:  Curvature-aware regularization (Euler's Elastica)
        * Boundary: Boundary distance transform loss
        * NWU:  Net Water Uptake (rHU) CT-physics constraint
        * Symmetry: Bilateral anatomical symmetry prior
    """

    def __init__(
        self,
        # Base loss params
        bce_weight: float = 0.5,
        gamma: float = 2.0,
        pos_weight: float = 66.0,
        alpha_tversky: float = 0.3,
        beta_tversky: float = 0.7,
        # Component λ weights (0 = disabled)
        lambda_ace: float = 0.0,
        lambda_boundary: float = 0.0,
        lambda_rhu: float = 0.0,
        lambda_sym: float = 0.0,
        target_nwu: float = 8.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.alpha_tversky = alpha_tversky
        self.beta_tversky = beta_tversky
        self.lambda_ace = lambda_ace
        self.lambda_boundary = lambda_boundary
        self.lambda_rhu = lambda_rhu
        self.lambda_sym = lambda_sym

        # Sub-loss modules (always instantiated, λ controls contribution)
        self.ace_loss = ACELoss()
        self.boundary_loss = BoundaryLoss()
        self.nwu_loss = NetWaterUptakeLoss(target_nwu=target_nwu)
        self.sym_loss = BilateralSymmetryLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dist_maps: Optional[torch.Tensor] = None,
        raw_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits:     (B, 1, H, W) — raw network output
            targets:    (B, 1, H, W) — binary ground-truth masks
            dist_maps:  (B, 1, H, W) — signed distance transforms (for Boundary)
            raw_images: (B, 1, H, W) — raw pixel intensities (for NWU, Symmetry)

        Returns:
            Scalar loss tensor.
        """
        # ── 1. Weighted Focal Loss ─────────────────────────────────
        pos_weight_t = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight_t, reduction="none"
        )
        pt = torch.exp(-bce)
        focal_loss = ((1.0 - pt) ** self.gamma * bce).mean()

        # ── 2. Tversky Loss ────────────────────────────────────────
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        fps = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fns = ((1.0 - probs) * targets).sum(dim=(2, 3))

        tversky_score = (intersection + 1e-6) / (
            intersection
            + self.alpha_tversky * fps
            + self.beta_tversky * fns
            + 1e-6
        )
        tversky_loss = (1.0 - tversky_score).mean()

        # ── Base (always active) ───────────────────────────────────
        base_loss = self.bce_weight * focal_loss + (1.0 - self.bce_weight) * tversky_loss
        total = base_loss

        # ── 3–6. Physics-informed components (λ-gated) ────────────
        if self.lambda_ace > 0:
            total += self.lambda_ace * self.ace_loss(logits)

        if self.lambda_boundary > 0 and dist_maps is not None:
            total += self.lambda_boundary * self.boundary_loss(logits, dist_maps)

        if self.lambda_rhu > 0 and raw_images is not None:
            total += self.lambda_rhu * self.nwu_loss(logits, raw_images)

        if self.lambda_sym > 0 and raw_images is not None:
            total += self.lambda_sym * self.sym_loss(logits, raw_images)

        return total


# ══════════════════════════════════════════════════════════════════════
# 3.  Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: float = 1.0,
) -> Dict[str, float]:
    """Compute segmentation metrics between predicted and ground-truth masks.

    Args:
        pred_mask:  (H, W) — binary prediction (0/1).
        gt_mask:    (H, W) — binary ground truth (0/1).
        voxel_spacing: Voxel spacing in mm (for HD95, default 1.0).

    Returns:
        Dictionary with keys: dice, iou, sensitivity, specificity,
        precision, hd95.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    # ── Confusion matrix elements ──────────────────────────────────
    tp = np.sum(pred & gt).astype(float)
    fp = np.sum(pred & ~gt).astype(float)
    fn = np.sum(~pred & gt).astype(float)
    tn = np.sum(~pred & ~gt).astype(float)

    # ── Standard metrics ───────────────────────────────────────────
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    # ── HD95: 95th percentile Hausdorff Distance ───────────────────
    hd95 = _compute_hd95(pred, gt, voxel_spacing)

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "hd95": hd95,
    }


def _compute_hd95(
    pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0
) -> float:
    """95th-percentile Hausdorff Distance via distance transforms."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    pred_surf = distance_transform_edt(~pred)
    gt_surf = distance_transform_edt(~gt)

    # Distances from prediction surface → GT surface
    d_pred_to_gt = gt_surf[pred] * spacing
    # Distances from GT surface → prediction surface
    d_gt_to_pred = pred_surf[gt] * spacing

    d_pred_to_gt.sort()
    d_gt_to_pred.sort()

    hd95_pred = d_pred_to_gt[int(len(d_pred_to_gt) * 0.95)] if len(d_pred_to_gt) > 0 else 0.0
    hd95_gt = d_gt_to_pred[int(len(d_gt_to_pred) * 0.95)] if len(d_gt_to_pred) > 0 else 0.0

    return float(max(hd95_pred, hd95_gt))


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across samples into mean ± std.

    Args:
        metrics_list: List of per-sample metric dicts.

    Returns:
        Nested dict: {metric_name: {"mean": ..., "std": ...}}.
    """
    keys = metrics_list[0].keys()
    aggregated = {}
    for k in keys:
        values = [m[k] for m in metrics_list if not (isinstance(m[k], float) and np.isnan(m[k]))]
        if len(values) == 0:
            aggregated[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            aggregated[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return aggregated


# ══════════════════════════════════════════════════════════════════════
# 4.  Model
# ══════════════════════════════════════════════════════════════════════

# Lazy import for smp to avoid requiring it at module load time
_model_cache = {}

def get_model(architecture: str = "unet_resnet34", device: str = "cuda") -> nn.Module:
    """Create and cache a segmentation model.

    Args:
        architecture: One of ``"unet_resnet34"``, ``"nccseg"``.
        device: Target device.

    Returns:
        PyTorch model on the specified device.
    """
    key = architecture
    if key in _model_cache:
        return _model_cache[key]

    if architecture == "unet_resnet34":
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None,
        )
    elif architecture == "nccseg":
        model = _NCCTSegModel()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)
    _model_cache[key] = model
    return model


class _NCCTSegModel(nn.Module):
    """Custom CSP-based segmentation model (no external dependencies)."""
    class _CSPBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            mid = out_ch // 2
            self.main = nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
                nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            )
            self.trans = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            )

        def forward(self, x):
            return self.trans(torch.cat([self.main(x), self.skip(x)], dim=1))

    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(1, 32, 3, padding=1)
        self.l1 = self._CSPBlock(32, 64)
        self.l2 = self._CSPBlock(64, 128)
        self.l3 = self._CSPBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.u1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = self._CSPBlock(128 + 256, 128)
        self.d2 = self._CSPBlock(64 + 128, 64)
        self.d3 = self._CSPBlock(32 + 64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x0 = F.relu(self.c0(x))
        x1 = self.l1(x0); p1 = self.pool(x1)
        x2 = self.l2(p1); p2 = self.pool(x2)
        x3 = self.l3(p2); f = self.pool(x3)
        d = self.d1(torch.cat([self.u1(f), x3], 1))
        d = self.d2(torch.cat([self.u2(d), x2], 1))
        d = self.d3(torch.cat([self.u3(d), x1], 1))
        return self.out(d)


# ══════════════════════════════════════════════════════════════════════
# 5.  Training and Evaluation
# ══════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    desc: str = "",
) -> float:
    """Train for one epoch.

    Returns:
        Mean training loss.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        img, msk, dm, _, rw = [x.cuda() for x in batch]
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, msk, dm, rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Evaluate model on a dataloader.

    Args:
        model:     Trained model.
        loader:    DataLoader (val or test).
        criterion: Loss function (if None, no loss is computed).

    Returns:
        Tuple of (aggregated_metrics_dict, mean_loss or NaN).
    """
    model.eval()
    all_metrics = []
    total_loss = 0.0
    n_batches = 0

    for img, msk, dm, _, rw in loader:
        img, msk, dm, rw = [x.cuda() for x in (img, msk, dm, rw)]
        logits = model(img)
        probs = torch.sigmoid(logits)

        if criterion is not None:
            total_loss += criterion(logits, msk, dm, rw).item()
            n_batches += 1

        # Per-sample metrics (move to CPU for numpy processing)
        preds_np = (probs > 0.5).cpu().numpy()  # (B, 1, H, W)
        targets_np = (msk > 0.5).cpu().numpy()

        for b in range(preds_np.shape[0]):
            sample_metrics = compute_metrics(
                preds_np[b, 0], targets_np[b, 0]
            )
            all_metrics.append(sample_metrics)

    agg = aggregate_metrics(all_metrics)
    avg_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    return agg, avg_loss


def train_ablation_model(
    architecture: str,
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    seed: int = 42,
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-4,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """Train a single ablation configuration from scratch.

    Args:
        architecture: Model architecture name.
        config:       Loss configuration dict with keys matching
                      ``AblationNCCTLoss.__init__``.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        seed:         Random seed for reproducibility.
        epochs:       Maximum number of epochs.
        patience:     Early stopping patience (val loss).
        lr:           Initial learning rate.
        lr_patience:  LR scheduler patience.
        lr_factor:    LR scheduler decay factor.
        verbose:      Print per-epoch progress.

    Returns:
        Tuple of:
            - Best validation metrics (dict with mean ± std).
            - Training history dict (train_loss, val_loss, val_dice lists).
    """
    set_seed(seed)

    # ── Model ──────────────────────────────────────────────────────
    model = get_model(architecture)
    model.train()

    # ── Loss ───────────────────────────────────────────────────────
    criterion = AblationNCCTLoss(**config).cuda()

    # ── Optimiser & scheduler ──────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")
    best_metrics = None
    best_state = None
    early_stop_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    if verbose:
        config_name = _config_name(config)
        print(f"\n{'─' * 60}")
        print(f"  {config_name}  |  seed={seed}")
        print(f"{'─' * 60}")
        print(f"{'Ep':<4} {'T-Loss':<8} {'V-Loss':<8} {'V-Dice':<8} {'LR':<10} {'Time':<7}")
        print(f"{'─' * 50}")

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────
        train_loss = train_epoch(model, train_loader, criterion, optimizer)

        # ── Validate ───────────────────────────────────────────────
        val_metrics, val_loss = evaluate(model, val_loader, criterion)
        val_dice = val_metrics.get("dice", {}).get("mean", 0.0)

        current_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        scheduler.step(val_loss)

        if verbose:
            print(f"{epoch+1:<4} {train_loss:<8.4f} {val_loss:<8.4f} {val_dice:<8.4f} {current_lr:<10.6f} {dt:<7.2f}s")

        # ── Early stopping (monitor val loss) ──────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_state = deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                if verbose:
                    print(f"  ⏹  Early stopping at epoch {epoch+1}")
                break

    # ── Restore best model ─────────────────────────────────────────
    model.load_state_dict(best_state)

    if verbose:
        print(f"{'─' * 50}")
        print(f"  Best val Dice: {best_metrics['dice']['mean']:.4f} ± {best_metrics['dice']['std']:.4f}")
        print(f"{'─' * 60}\n")

    return best_metrics, history


def _config_name(config: Dict) -> str:
    """Generate a human-readable name from a loss config."""
    parts = ["Base"]
    if config.get("lambda_ace", 0) > 0:
        parts.append(f"ACE({config['lambda_ace']})")
    if config.get("lambda_boundary", 0) > 0:
        parts.append(f"BND({config['lambda_boundary']})")
    if config.get("lambda_rhu", 0) > 0:
        parts.append(f"NWU({config['lambda_rhu']})")
    if config.get("lambda_sym", 0) > 0:
        parts.append(f"SYM({config['lambda_sym']})")
    return " + ".join(parts)


@torch.inference_mode()
def test_model(
    model: nn.Module, test_loader: DataLoader
) -> Dict[str, Dict[str, float]]:
    """Evaluate a trained model on the test set.

    Returns:
        Aggregated metrics dictionary.
    """
    model.eval()
    all_metrics = []

    for img, msk, _, _, _ in test_loader:
        logits = model(img.cuda())
        probs = torch.sigmoid(logits)

        preds_np = (probs > 0.5).cpu().numpy()
        targets_np = (msk > 0.5).cpu().numpy()

        for b in range(preds_np.shape[0]):
            all_metrics.append(compute_metrics(preds_np[b, 0], targets_np[b, 0]))

    return aggregate_metrics(all_metrics)


# ══════════════════════════════════════════════════════════════════════
# 6.  Ablation Configuration Definitions
# ══════════════════════════════════════════════════════════════════════

# Default λ weights (from V8 best config)
LAMBDA_DEFAULTS = {
    "ace": 0.01,
    "boundary": 0.1,
    "rhu": 0.01,
    "sym": 0.05,
}

ABLATION_CONFIGS = {
    # ── Add-one-in ───────────────────────────────────────────────
    "Baseline": {
        "lambda_ace": 0.0,
        "lambda_boundary": 0.0,
        "lambda_rhu": 0.0,
        "lambda_sym": 0.0,
    },
    "+ACE": {
        "lambda_ace": LAMBDA_DEFAULTS["ace"],
        "lambda_boundary": 0.0,
        "lambda_rhu": 0.0,
        "lambda_sym": 0.0,
    },
    "+Boundary": {
        "lambda_ace": 0.0,
        "lambda_boundary": LAMBDA_DEFAULTS["boundary"],
        "lambda_rhu": 0.0,
        "lambda_sym": 0.0,
    },
    "+NWU": {
        "lambda_ace": 0.0,
        "lambda_boundary": 0.0,
        "lambda_rhu": LAMBDA_DEFAULTS["rhu"],
        "lambda_sym": 0.0,
    },
    "+Symmetry": {
        "lambda_ace": 0.0,
        "lambda_boundary": 0.0,
        "lambda_rhu": 0.0,
        "lambda_sym": LAMBDA_DEFAULTS["sym"],
    },
    "Full": {
        "lambda_ace": LAMBDA_DEFAULTS["ace"],
        "lambda_boundary": LAMBDA_DEFAULTS["boundary"],
        "lambda_rhu": LAMBDA_DEFAULTS["rhu"],
        "lambda_sym": LAMBDA_DEFAULTS["sym"],
    },
    # ── Leave-one-out ────────────────────────────────────────────
    "Full − ACE": {
        "lambda_ace": 0.0,
        "lambda_boundary": LAMBDA_DEFAULTS["boundary"],
        "lambda_rhu": LAMBDA_DEFAULTS["rhu"],
        "lambda_sym": LAMBDA_DEFAULTS["sym"],
    },
    "Full − Boundary": {
        "lambda_ace": LAMBDA_DEFAULTS["ace"],
        "lambda_boundary": 0.0,
        "lambda_rhu": LAMBDA_DEFAULTS["rhu"],
        "lambda_sym": LAMBDA_DEFAULTS["sym"],
    },
    "Full − NWU": {
        "lambda_ace": LAMBDA_DEFAULTS["ace"],
        "lambda_boundary": LAMBDA_DEFAULTS["boundary"],
        "lambda_rhu": 0.0,
        "lambda_sym": LAMBDA_DEFAULTS["sym"],
    },
    "Full − Symmetry": {
        "lambda_ace": LAMBDA_DEFAULTS["ace"],
        "lambda_boundary": LAMBDA_DEFAULTS["boundary"],
        "lambda_rhu": LAMBDA_DEFAULTS["rhu"],
        "lambda_sym": 0.0,
    },
}

ABLATION_GROUPS = {
    "Add-one-in": ["Baseline", "+ACE", "+Boundary", "+NWU", "+Symmetry", "Full"],
    "Leave-one-out": ["Full", "Full − ACE", "Full − Boundary", "Full − NWU", "Full − Symmetry"],
    "All": list(ABLATION_CONFIGS.keys()),
}


# ══════════════════════════════════════════════════════════════════════
# 7.  Run Full Ablation
# ══════════════════════════════════════════════════════════════════════

def run_ablation(
    architecture: str = "unet_resnet34",
    seeds: List[int] = (42, 43, 44),
    config_names: Optional[List[str]] = None,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 4,
    limit_data: Optional[int] = None,
    data_dir: str = "dataset",
    verbose: bool = True,
    run_test: bool = True,
) -> Dict[str, Dict]:
    """Run the full loss ablation experiment.

    Args:
        architecture:  Model architecture (``"unet_resnet34"`` or ``"nccseg"``).
        seeds:         Random seeds for replication.
        config_names:  Configs to run (default: all 10).
        epochs:        Max epochs per training run.
        patience:      Early stopping patience.
        batch_size:    Batch size.
        limit_data:    Limit dataset size for debugging.
        data_dir:      Dataset root directory.
        verbose:       Print progress.
        run_test:      Also evaluate on test set.

    Returns:
        Nested dict::

            {
                config_name: {
                    seed: {
                        "val_metrics": aggregated_val_metrics,
                        "test_metrics": aggregated_test_metrics or None,
                        "history": training_history_dict,
                    }
                }
            }
    """
    if config_names is None:
        config_names = ABLATION_GROUPS["All"]

    # ── Load data ──────────────────────────────────────────────────
    if verbose:
        print("=" * 70)
        print("  NCCT V8 — Loss Ablation Study")
        print(f"  Architecture: {architecture}")
        print(f"  Configurations: {len(config_names)}")
        print(f"  Seeds per config: {seeds}")
        print(f"  Total training runs: {len(config_names) * len(seeds)}")
        print("=" * 70)
        print("\nLoading dataset...")
        sys.stdout.flush()

    train_loader, val_loader, test_loader = get_dataloaders_v8(
        data_dir, batch_size=batch_size, limit_size=limit_data
    )

    if verbose:
        print(f"  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}"
              f"  Test: {len(test_loader.dataset)}")

    # ── Run each configuration ─────────────────────────────────────
    results = {}

    for cfg_name in config_names:
        config = ABLATION_CONFIGS[cfg_name]
        results[cfg_name] = {}

        for seed in seeds:
            model = get_model(architecture)  # fresh model each time
            best_metrics, history = train_ablation_model(
                architecture=architecture,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                seed=seed,
                epochs=epochs,
                patience=patience,
                verbose=verbose,
            )

            # Test evaluation (best model already restored in train_ablation_model)
            test_metrics = None
            if run_test:
                test_metrics = test_model(model, test_loader)

            results[cfg_name][seed] = {
                "val_metrics": best_metrics,
                "test_metrics": test_metrics,
                "history": history,
            }

            if verbose:
                print(f"  ✓ {cfg_name} (seed={seed}) done — val Dice: {best_metrics['dice']['mean']:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════
# 8.  Result Aggregation & Display
# ══════════════════════════════════════════════════════════════════════

def aggregate_results(
    results: Dict[str, Dict],
    metric_key: str = "val_metrics",
    target: str = "dice",
) -> Dict[str, Dict[str, float]]:
    """Aggregate ablation results across seeds for a given metric.

    Args:
        results:     Raw results dict from ``run_ablation``.
        metric_key:  ``"val_metrics"`` or ``"test_metrics"``.
        target:      Metric name (``"dice"``, ``"iou"``, ``"hd95"``, etc.).

    Returns:
        Dict mapping config_name → {"mean": ..., "std": ..., "seeds": [values]}.
    """
    aggregated = {}
    for cfg_name, seed_dict in results.items():
        values = []
        for seed, data in seed_dict.items():
            metrics = data.get(metric_key)
            if metrics and target in metrics:
                values.append(metrics[target]["mean"])
        if values:
            aggregated[cfg_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "seeds": values,
            }
    return aggregated


def print_ablation_table(
    results: Dict[str, Dict],
    metric_key: str = "val_metrics",
    metrics: Optional[List[str]] = None,
    sort_by: str = "dice",
) -> None:
    """Print a formatted ablation results table.

    Args:
        results:    Raw results dict from ``run_ablation``.
        metric_key: ``"val_metrics"`` or ``"test_metrics"``.
        metrics:    Metrics to display (default: all).
        sort_by:    Metric to sort by.
    """
    if metrics is None:
        metrics = ["dice", "iou", "sensitivity", "specificity", "precision", "hd95"]

    # Aggregate all metrics
    agg = {}
    for m in metrics:
        agg[m] = aggregate_results(results, metric_key=metric_key, target=m)

    # Build header
    header_parts = [f"{'Config':<20}"]
    for m in metrics:
        header_parts.append(f"{m.upper():<14}")
    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print(f"\n  [{metric_key.upper()}]")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    # Sort by requested metric
    sorted_configs = sorted(
        agg[sort_by].keys(),
        key=lambda c: agg[sort_by][c]["mean"],
        reverse=True,
    )

    for cfg_name in sorted_configs:
        row = [f"{cfg_name:<20}"]
        for m in metrics:
            if cfg_name in agg[m]:
                mean = agg[m][cfg_name]["mean"]
                std = agg[m][cfg_name]["std"]
                if m == "hd95":
                    row.append(f"{mean:<6.2f}±{std:<5.2f}")
                else:
                    row.append(f"{mean*100:<6.2f}±{std*100:<5.2f}%")
            else:
                row.append(f"{'N/A':<14}")
        print(f"  {' | '.join(row)}")

    print(f"  {sep}\n")


def plot_ablation_results(
    results: Dict[str, Dict],
    metric_key: str = "val_metrics",
    title: str = "Loss Ablation — NCCT Stroke Segmentation",
    save_path: Optional[str] = None,
) -> None:
    """Plot ablation results as grouped bar charts.

    Args:
        results:    Raw results dict from ``run_ablation``.
        metric_key: ``"val_metrics"`` or ``"test_metrics"``.
        title:      Plot title.
        save_path:  Optional path to save figure.
    """
    metrics_to_plot = ["dice", "iou", "sensitivity", "precision", "hd95"]
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    config_names = list(results.keys())
    x = np.arange(len(config_names))
    width = 0.6

    colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        agg = aggregate_results(results, metric_key=metric_key, target=metric)

        means = [agg[c]["mean"] if c in agg else 0 for c in config_names]
        stds = [agg[c]["std"] if c in agg else 0 for c in config_names]

        bars = ax.bar(x, means, width, yerr=stds, capsize=3,
                      color=colors, edgecolor="gray", linewidth=0.5)

        # Highlight baseline and full
        for i, c in enumerate(config_names):
            if c == "Baseline":
                bars[i].set_facecolor("#e74c3c")
                bars[i].set_edgecolor("darkred")
            elif c == "Full":
                bars[i].set_facecolor("#2ecc71")
                bars[i].set_edgecolor("darkgreen")

        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")

        if metric == "hd95":
            ax.set_ylabel("HD95 (pixels)")
        else:
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)

        ax.axhline(y=agg.get("Baseline", {}).get("mean", 0),
                   color="red", linestyle="--", alpha=0.5,
                   label="Baseline" if ax_idx == 0 else "")
        ax.axhline(y=agg.get("Full", {}).get("mean", 0),
                   color="green", linestyle="--", alpha=0.5,
                   label="Full" if ax_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=9)
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure to {save_path}")
    plt.show()


def plot_training_curves(
    results: Dict[str, Dict],
    config_names: Optional[List[str]] = None,
    seed: int = 42,
    metric: str = "dice",
) -> None:
    """Plot training curves for a subset of configurations.

    Args:
        results:       Raw results dict.
        config_names:  Configs to plot (default: all add-one-in).
        seed:          Which seed's history to use.
        metric:        ``"dice"`` or ``"loss"``.
    """
    if config_names is None:
        config_names = ABLATION_GROUPS["Add-one-in"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for cfg_name in config_names:
        if cfg_name not in results or seed not in results[cfg_name]:
            continue
        history = results[cfg_name][seed]["history"]

        axes[0].plot(history["train_loss"], label=cfg_name, lw=1.5)
        axes[1].plot(history["val_dice"], label=cfg_name, lw=1.5)

    axes[0].set(xlabel="Epoch", ylabel="Train Loss", title="Training Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set(xlabel="Epoch", ylabel="Val Dice", title="Validation Dice")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Training Curves (seed={seed})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# 9.  Main (for command-line execution)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V8 Loss Ablation Study")
    parser.add_argument("--architecture", default="unet_resnet34",
                        choices=["unet_resnet34", "nccseg"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config names (default: all 10)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-data", type=int, default=None,
                        help="Limit samples for fast debugging")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip test evaluation")
    args = parser.parse_args()

    results = run_ablation(
        architecture=args.architecture,
        seeds=args.seeds,
        config_names=args.configs,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        limit_data=args.limit_data,
        data_dir=args.data_dir,
        run_test=not args.no_test,
    )

    print("\n" + "=" * 70)
    print("  ABLATION RESULTS — VALIDATION")
    print("=" * 70)
    print_ablation_table(results, metric_key="val_metrics")

    if not args.no_test:
        print("\n" + "=" * 70)
        print("  ABLATION RESULTS — TEST")
        print("=" * 70)
        print_ablation_table(results, metric_key="test_metrics")

    plot_ablation_results(results, metric_key="val_metrics")
    plot_training_curves(results)
