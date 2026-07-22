"""
v8_losses.py — Physics-Informed Loss Functions for NCCT Brain Stroke Segmentation

Implements 5 enhanced loss functions based on CT physics and anatomical priors:
  1. ACELoss         — Euler's Elastica curvature-aware regularization
  2. BoundaryLoss    — Boundary distance transform loss for unbalanced segmentation
  3. NetWaterUptakeLoss — rHU / Net Water Uptake consistency constraint
  4. BilateralSymmetryLoss — Anatomical symmetry prior for hypodense lesions
  5. PhysicsEnhancedNCCTLoss — Combined training loss (Focal + Tversky + all physics)
  6. BaselineNCCTLoss — V7-compatible loss for ablation comparisons

All operations use pure PyTorch (no numpy) and accept 4D tensors (B, 1, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Class 1: ACELoss (Euler's Elastica)
# ---------------------------------------------------------------------------

class ACELoss(nn.Module):
    """
    Euler's Elastica-based curvature-aware regularization loss.

    Replaces the generic Total-Variation (TV) smoothness term with a
    curvature-weighted edge penalty:
        L_ace = (alpha + beta * kappa^2) * |nabla u|

    References:
        Chen et al. "Learning Euler's Elastica Model for Medical Image
        Segmentation" (IEEE TMI 2022)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Probability map in [0, 1], shape (B, 1, H, W)
        probs = torch.sigmoid(logits)

        # ---- |nabla u| : gradient magnitude via finite differences ----
        # x-direction: difference between adjacent columns
        grad_x = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])      # (B, 1, H, W-1)
        # y-direction: difference between adjacent rows
        grad_y = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])      # (B, 1, H-1, W)

        grad_mag = torch.mean(grad_x) + torch.mean(grad_y)                # scalar

        # ---- Curvature approximation via Laplacian ----
        # Second-difference in x direction  (central difference)
        lap_x = torch.abs(
            probs[:, :, :, 2:] + probs[:, :, :, :-2] - 2.0 * probs[:, :, :, 1:-1]
        )                                                                 # (B, 1, H, W-2)
        # Second-difference in y direction
        lap_y = torch.abs(
            probs[:, :, 2:, :] + probs[:, :, :-2, :] - 2.0 * probs[:, :, 1:-1, :]
        )                                                                 # (B, 1, H-2, W)

        curvature = torch.mean(lap_x) + torch.mean(lap_y)                 # scalar, kappa

        # ---- ACE: (alpha + beta * kappa^2) * |nabla u| ----
        ace = (self.alpha + self.beta * curvature ** 2) * grad_mag / 2.0

        return ace


# ---------------------------------------------------------------------------
# Class 2: BoundaryLoss
# ---------------------------------------------------------------------------

class BoundaryLoss(nn.Module):
    """
    Boundary loss for highly unbalanced segmentation.

    Uses pre-computed signed distance transforms to penalise predictions
    that fall outside the ground-truth boundary.

        L_boundary = (1/N) * sum(p_i * D_G_i)

    where p_i is the predicted probability and D_G_i is the signed distance
    (negative inside GT, positive outside GT).

    References:
        Kervadec et al. "Boundary loss for highly unbalanced segmentation"
        (MIDL 2019 / MedIA 2021)
    """

    def forward(self, logits: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:    Raw network logits, shape (B, 1, H, W)
            dist_maps: Pre-computed signed distance maps, shape (B, 1, H, W).
                       Positive outside GT boundary, negative inside GT.
        Returns:
            Scalar boundary loss (can be negative when prediction is inside GT).
        """
        probs = torch.sigmoid(logits)
        # Loss is positive when prediction extends outside the GT boundary
        loss = (probs * dist_maps).sum(dim=(2, 3)).mean()
        return loss


# ---------------------------------------------------------------------------
# Class 3: NetWaterUptakeLoss (rHU / NWU Consistency)
# ---------------------------------------------------------------------------

class NetWaterUptakeLoss(nn.Module):
    """
    CT-specific physics constraint based on Net Water Uptake (NWU / rHU).

        NWU = (1 - HU_lesion / HU_contralateral) * 100

    Expected NWU for acute infarct is ~7-10%. The loss penalises predicted
    regions whose NWU deviates from this target.

    References:
        − Broocks et al. "Net water uptake in acute ischemic stroke"
        − Minnerup et al. "Computed tomography-based quantification of
          lesion water uptake"
    """

    def __init__(self, target_nwu: float = 8.0):
        super().__init__()
        self.target_nwu = target_nwu

    def forward(self, logits: torch.Tensor, raw_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:      Network logits, shape (B, 1, H, W)
            raw_images:  Pre-normalisation CT pixel intensities, shape (B, 1, H, W)
        Returns:
            Scalar loss penalising NWU deviation from target.
        """
        probs = torch.sigmoid(logits)

        # Mirror the images and predictions for contralateral comparison
        img_mirror = torch.flip(raw_images, dims=[-1])
        probs_mirror = torch.flip(probs, dims=[-1])

        # Mean HU in the predicted lesion region
        hu_lesion = (probs * raw_images).sum(dim=(2, 3)) / (
            probs.sum(dim=(2, 3)) + 1e-8
        )  # (B, 1)

        # Mean HU in the mirrored (contralateral) region
        hu_contra = (probs_mirror * img_mirror).sum(dim=(2, 3)) / (
            probs_mirror.sum(dim=(2, 3)) + 1e-8
        )  # (B, 1)

        # Net Water Uptake percentage
        nwu = (1.0 - hu_lesion / (hu_contra + 1e-8)) * 100.0

        # Penalise deviation from the expected physiological range
        loss = torch.abs(nwu - self.target_nwu).mean()
        return loss


# ---------------------------------------------------------------------------
# Class 4: BilateralSymmetryLoss
# ---------------------------------------------------------------------------

class BilateralSymmetryLoss(nn.Module):
    """
    Anatomical symmetry prior for hypodense stroke lesions.

    Healthy brains are approximately left-right symmetric. Stroke lesions
    appear hypodense (darker) on NCCT, breaking this symmetry.

        L_sym = mean(probs * ReLU(I_mirror - I))

    The asymmetry map ReLU(I_mirror - I) is positive where the mirrored
    image is brighter than the original — indicating a potential hypodense
    lesion. The loss penalises predictions in regions without asymmetry.

    References:
        Ni et al. "Symmetry-Enhanced Attention Network" (MICCAI 2022)
        Liang et al. "ADN: Asymmetric Difference Network"
    """

    def forward(self, logits: torch.Tensor, raw_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:      Network logits, shape (B, 1, H, W)
            raw_images:  Pre-normalisation CT pixel intensities, shape (B, 1, H, W)
        Returns:
            Scalar symmetry loss.
        """
        probs = torch.sigmoid(logits)

        # Horizontal mirror flip
        img_mirror = torch.flip(raw_images, dims=[-1])

        # Asymmetry map: positive only where the mirrored image is brighter
        # (i.e. the original is darker / hypodense)
        asym_map = F.relu(img_mirror - raw_images)

        # Penalise predictions in regions that show no asymmetry
        loss = (probs * (1.0 - asym_map)).mean()
        return loss


# ---------------------------------------------------------------------------
# Class 5: PhysicsEnhancedNCCTLoss (Combined)
# ---------------------------------------------------------------------------

class PhysicsEnhancedNCCTLoss(nn.Module):
    """
    Combined multi-component loss for NCCT brain stroke segmentation.

    Components:
        1. Weighted Focal Loss     (class imbalance)
        2. Tversky Loss            (precision-recall trade-off)
        3. ACE Loss                (curvature-aware regularisation, replaces TV)
        4. Boundary Loss           (boundary alignment)
        5. Net Water Uptake Loss   (CT physics)
        6. Bilateral Symmetry Loss (anatomical prior)

    Usage:
        loss_fn = PhysicsEnhancedNCCTLoss(...)
        total = loss_fn(logits, targets, dist_maps, raw_images)
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        gamma: float = 2.0,
        pos_weight: float = 66.0,
        alpha_tversky: float = 0.3,
        beta_tversky: float = 0.7,
        lambda_ace: float = 0.01,
        lambda_boundary: float = 0.1,
        lambda_rhu: float = 0.01,
        lambda_sym: float = 0.05,
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

        # Sub-loss modules
        self.ace_loss = ACELoss()
        self.boundary_loss = BoundaryLoss()
        self.nwu_loss = NetWaterUptakeLoss(target_nwu=target_nwu)
        self.sym_loss = BilateralSymmetryLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dist_maps: torch.Tensor,
        raw_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:     Network logits, shape (B, 1, H, W)
            targets:    Ground-truth binary masks, shape (B, 1, H, W)
            dist_maps:  Signed distance maps for boundary loss, shape (B, 1, H, W)
            raw_images: Pre-normalisation CT intensities, shape (B, 1, H, W)
        Returns:
            Scalar combined loss.
        """
        # ---- 1. Weighted Focal Loss ----
        pos_weight_t = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight_t, reduction="none"
        )
        pt = torch.exp(-bce)  # probability of correct prediction
        focal_loss = ((1.0 - pt) ** self.gamma * bce).mean()

        # ---- 2. Tversky Loss ----
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
        tversky_loss = 1.0 - tversky_score.mean()

        # ---- 3. ACE Loss (curvature-aware, replaces TV) ----
        ace = self.ace_loss(logits)

        # ---- 4. Boundary Loss ----
        boundary = self.boundary_loss(logits, dist_maps)

        # ---- 5. Net Water Uptake Loss ----
        rhu = self.nwu_loss(logits, raw_images)

        # ---- 6. Bilateral Symmetry Loss ----
        sym = self.sym_loss(logits, raw_images)

        # ---- Combine ----
        base_loss = self.bce_weight * focal_loss + (1.0 - self.bce_weight) * tversky_loss
        total = (
            base_loss
            + self.lambda_ace * ace
            + self.lambda_boundary * boundary
            + self.lambda_rhu * rhu
            + self.lambda_sym * sym
        )

        return total


# ---------------------------------------------------------------------------
# Class 6: BaselineNCCTLoss (V7-compatible ablation)
# ---------------------------------------------------------------------------

class BaselineNCCTLoss(nn.Module):
    """
    V7-compatible baseline loss (Weighted Focal + Tversky) for ablation
    comparisons.  Does NOT include any physics-informed components.

    Usage:
        loss_fn = BaselineNCCTLoss(...)
        loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        gamma: float = 2.0,
        pos_weight: float = 66.0,
        alpha: float = 0.3,
        beta: float = 0.7,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Network logits, shape (B, 1, H, W)
            targets: Ground-truth binary masks, shape (B, 1, H, W)
        Returns:
            Scalar combined baseline loss.
        """
        # ---- Weighted Focal Loss ----
        pos_weight_t = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight_t, reduction="none"
        )
        pt = torch.exp(-bce)
        focal_loss = ((1.0 - pt) ** self.gamma * bce).mean()

        # ---- Tversky Loss ----
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        fps = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fns = ((1.0 - probs) * targets).sum(dim=(2, 3))

        tversky_score = (intersection + 1e-6) / (
            intersection + self.alpha * fps + self.beta * fns + 1e-6
        )
        tversky_loss = 1.0 - tversky_score.mean()

        # ---- Combine ----
        total = self.bce_weight * focal_loss + (1.0 - self.bce_weight) * tversky_loss
        return total
