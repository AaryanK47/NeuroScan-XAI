"""
Custom CNN for binary brain tumour classification (Tumour / Non-Tumour)
on BraTS 2021 FLAIR MRI slices.

Architecture follows Chapter 6.4 of the project report:
  Conv -> ReLU -> MaxPool  (x4 blocks, increasing channels)
  -> Dropout
  -> Fully Connected layers
  -> Sigmoid (binary classification)

Kept deliberately simple (not a pretrained transfer-learning backbone) so that:
  1. It matches "custom CNN" as stated in the report/PPT.
  2. Grad-CAM, LRP, and SHAP all stay fast and interpretable on it.
  3. It trains in reasonable time on CPU for demo purposes, and scales
     to GPU for real BraTS training without any code changes.
"""

import torch
import torch.nn as nn
from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule


class BrainTumorCNN(nn.Module):
    def __init__(self, in_channels: int = 1, input_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 128 -> 64

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 64 -> 32

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 32 -> 16

            # Block 4 (last conv block -> used as Grad-CAM target layer)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        flat_dim = 128 * (input_size // 16) * (input_size // 16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            # Sigmoid is applied at inference time / in the loss (BCEWithLogitsLoss)
            # so the raw logit is exposed for Captum/SHAP attribution methods.
        )

        self._attach_lrp_rules()

    def _attach_lrp_rules(self):
        """
        Captum's LRP implementation needs an explicit propagation rule on
        every layer type it doesn't already recognize (e.g. Flatten,
        Dropout). Conv2d / Linear / MaxPool2d already default to
        EpsilonRule inside Captum, so we only need to patch the rest.
        """
        for module in self.classifier:
            if isinstance(module, (nn.Flatten, nn.Dropout)):
                module.rule = IdentityRule()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logit, shape (B, 1)

    @property
    def gradcam_target_layer(self):
        """Last convolutional layer — used by Grad-CAM."""
        return self.features[9]  # Conv2d(64, 128, ...)
