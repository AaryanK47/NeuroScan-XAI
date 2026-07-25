"""
Unified XAI module: Grad-CAM (region-level), LRP (pixel-level),
and SHAP (feature-level) explanations for the BrainTumorCNN.

All three functions take:
  model        - BrainTumorCNN in eval() mode
  input_tensor - shape (1, 1, H, W), already normalized
  device       - torch.device

and return a (H, W) numpy heatmap normalized to [0, 1], ready to be
overlaid on the original MRI slice.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from captum.attr import LayerGradCam


def _normalize_heatmap(h: np.ndarray) -> np.ndarray:
    h = h - h.min()
    if h.max() > 1e-8:
        h = h / h.max()
    return h


def _resize_to_input(h: np.ndarray, size) -> np.ndarray:
    return cv2.resize(h, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# 1. Grad-CAM — region-level explanation
# ---------------------------------------------------------------------------
def grad_cam(model, input_tensor, device, target_class: int = 0):
    input_tensor = input_tensor.to(device).requires_grad_(True)
    gc = LayerGradCam(model, model.gradcam_target_layer)
    attribution = gc.attribute(input_tensor, target=0)  # single logit output at index 0
    attribution = attribution.squeeze().detach().cpu().numpy()
    relu_attr = np.maximum(attribution, 0)  # ReLU on the CAM, as in the original paper
    # On a saturated/very confident logit the gradient signal can end up
    # uniformly negative, which would zero out the standard ReLU-clamped
    # CAM entirely. Fall back to the unclamped, min-max normalized map so
    # the explanation still reflects relative spatial importance.
    attribution = relu_attr if relu_attr.max() > 1e-8 else attribution
    attribution = _normalize_heatmap(attribution)
    h, w = input_tensor.shape[-2:]
    attribution = _resize_to_input(attribution, (h, w))
    attribution = np.clip(attribution, 0, 1)
    return attribution


# ---------------------------------------------------------------------------
# 2. Layer-wise Relevance Propagation — pixel-level explanation
#
# Implemented as a manual epsilon-LRP pass (Montavon et al., 2019,
# "Layer-Wise Relevance Propagation: An Overview") using the standard
# "gradient trick": for a weighted layer z = f(x), relevance flows back as
#     R_in = x * grad_x[ sum( z * (R_out / (z + eps)) ) ]
# ReLU / Dropout are elementwise, non-mixing operations and simply pass
# relevance through unchanged, as is standard practice in LRP for CNNs.
# ---------------------------------------------------------------------------
def _lrp_weighted_layer(layer, layer_input, R_out, eps=1e-6):
    layer_input = layer_input.clone().detach().requires_grad_(True)
    z = layer(layer_input)
    z = z + eps * torch.sign(z).clamp(min=1e-9)  # stabilize division near zero
    s = (R_out / z).detach()
    (z * s).sum().backward()
    R_in = (layer_input * layer_input.grad).detach()
    return R_in


def lrp(model, input_tensor, device, eps: float = 1e-6):
    model = model.to(device)
    x = input_tensor.to(device).detach()

    # ---- forward pass, caching the input to every layer ----
    layers = list(model.features) + list(model.classifier)
    activations = [x]
    a = x
    with torch.no_grad():
        for layer in layers:
            a = layer(a)
            activations.append(a)

    # ---- backward relevance pass ----
    R = activations[-1].clone()  # relevance initialized at the output logit

    for i in reversed(range(len(layers))):
        layer = layers[i]
        layer_input = activations[i]

        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
            R = _lrp_weighted_layer(layer, layer_input, R, eps=eps)
        elif isinstance(layer, nn.Flatten):
            R = R.view(layer_input.shape)
        else:
            # ReLU, Dropout (eval-mode no-op): pass relevance through unchanged
            pass

    attribution = R.squeeze().detach().cpu().numpy()
    attribution = np.abs(attribution)  # relevance magnitude per pixel
    attribution = _normalize_heatmap(attribution)
    return attribution


# ---------------------------------------------------------------------------
# 3. SHAP — feature-level (patch) contribution explanation
# ---------------------------------------------------------------------------
def shap_explanation(model, input_tensor, device, background_tensor, n_samples: int = 50):
    """
    Uses GradientExplainer (efficient for CNNs) with a small background
    set drawn from the training distribution. Returns a (H, W) SHAP
    relevance map, and also the raw per-pixel signed values in case the
    caller wants to distinguish positive/negative contribution.
    """
    import shap

    model_input = input_tensor.to(device)
    background = background_tensor.to(device)

    explainer = shap.GradientExplainer(model, background)
    shap_values, _ = explainer.shap_values(model_input, nsamples=n_samples, ranked_outputs=1)

    # shap_values: list with one array of shape (1, 1, H, W) for our single output
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    sv = np.array(sv).squeeze()
    signed = sv.copy()
    magnitude = _normalize_heatmap(np.abs(sv))
    return magnitude, signed


# ---------------------------------------------------------------------------
# Overlay helper — used by the API to turn a heatmap into a viewable PNG
# ---------------------------------------------------------------------------
def overlay_heatmap(base_image_uint8: np.ndarray, heatmap: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.45):
    """
    base_image_uint8 : (H, W) grayscale uint8 MRI slice
    heatmap          : (H, W) float in [0, 1]
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    base_bgr = cv2.cvtColor(base_image_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 1 - alpha, color_heatmap, alpha, 0)
    return overlay  # BGR uint8, ready to encode as PNG
