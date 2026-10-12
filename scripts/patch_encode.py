"""Fixed raw-pixel preparation for the Full-QCNN pilot (spec Section 5)."""
from __future__ import annotations
import functools
import math
import numpy as np
import torch
from PIL import Image

@functools.lru_cache(maxsize=None)
def morton_order_indices(h: int, w: int, c: int) -> np.ndarray:
    """Return flat HWC indices ordered by a 3-axis Morton/Z-order key."""
    if min(h, w, c) <= 0:
        raise ValueError("Morton dimensions must be positive")
    row, col, channel = np.indices((h, w, c))
    code = np.zeros((h, w, c), dtype=np.uint64)
    for bit in range(max(h, w, c).bit_length()):
        code |= ((row >> bit) & 1).astype(np.uint64) << (3 * bit)
        code |= ((col >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
        code |= ((channel >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
    return np.argsort(code.reshape(-1), kind="stable")

def _as_hwc(image):
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D RGB image, got {tuple(image.shape)}")
    if image.shape[-1] == 3:
        return image
    if image.shape[0] == 3:
        return image.permute(1, 2, 0) if torch.is_tensor(image) else np.transpose(image, (1, 2, 0))
    raise ValueError(f"expected RGB image, got {tuple(image.shape)}")

def per_image_angle_normalize(image):
    """Map an image to [0, pi] using its own statistics only."""
    x = _as_hwc(image)
    if torch.is_tensor(x):
        x = x.float() / (255.0 if x.max() > 1.0 else 1.0)
        x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
        return (x - x.amin()) / (x.amax() - x.amin()).clamp_min(1e-6) * math.pi
    x = np.asarray(x, dtype=np.float32) / (255.0 if x.max() > 1.0 else 1.0)
    x = (x - x.mean()) / max(float(x.std()), 1e-6)
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-6) * math.pi

def extract_patches(image_224x224x3, patch: int = 16):
    """Return [n_patches, patch*patch*3] in fixed raster-grid order."""
    x = _as_hwc(image_224x224x3)
    h, w, c = x.shape
    if h != w or h % patch or w % patch or c != 3:
        raise ValueError(f"expected square RGB image divisible by {patch}, got {tuple(x.shape)}")
    grid = h // patch
    if torch.is_tensor(x):
        return x.reshape(grid, patch, grid, patch, c).permute(0, 2, 1, 3, 4).reshape(grid * grid, -1)
    return x.reshape(grid, patch, grid, patch, c).transpose(0, 2, 1, 3, 4).reshape(grid * grid, -1)

def amplitude_prep_vector(patch_vec, n_qubits: int = 10, patch_shape=(16, 16, 3)):
    """Morton-reorder, zero-pad, and L2-normalize one patch or a patch batch."""
    expected, target = int(np.prod(patch_shape)), 2 ** n_qubits
    if patch_vec.shape[-1] != expected or expected > target:
        raise ValueError("patch shape does not fit requested amplitude register")
    order = morton_order_indices(*patch_shape)
    if torch.is_tensor(patch_vec):
        x = patch_vec[..., torch.as_tensor(order, device=patch_vec.device)]
        x = torch.nn.functional.pad(x, (0, target - expected))
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        fallback = torch.zeros_like(x); fallback[..., 0] = 1.0
        return torch.where(norm > 1e-8, x / norm.clamp_min(1e-8), fallback)
    x = np.asarray(patch_vec, dtype=np.float32)[..., order]
    x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, target - expected)])
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)

def channel_amplitude_prep(patches, patch: int = 16):
    """Fallback: prepare three Morton-ordered 16x16 channel registers (8 qubits)."""
    x = patches.reshape(-1, patch, patch, 3).permute(0, 3, 1, 2).reshape(-1, 3, patch * patch)
    order = morton_order_indices(patch, patch, 1)
    x = x[..., torch.as_tensor(order, device=x.device)]
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    fallback = torch.zeros_like(x); fallback[..., 0] = 1.0
    return torch.where(norm > 1e-8, x / norm.clamp_min(1e-8), fallback)

def load_image_patches(path: str, patch: int = 16, n_qubits: int = 10, encoding="amplitude10") -> torch.Tensor:
    with Image.open(path) as image:
        pixels = torch.from_numpy(np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32).copy())
    patches = extract_patches(per_image_angle_normalize(pixels), patch)
    if encoding == "amplitude10":
        return amplitude_prep_vector(patches, n_qubits, (patch, patch, 3)).float()
    if encoding == "amplitude10_mag":
        amplitudes = amplitude_prep_vector(patches, n_qubits, (patch, patch, 3))
        patch_hwc = patches.reshape(-1, patch, patch, 3)
        channel_means = patch_hwc.mean(dim=(1, 2))
        pre_normalization_norm = torch.linalg.vector_norm(patches, dim=-1, keepdim=True)
        magnitude_angle = pre_normalization_norm / math.sqrt(patch * patch * 3)
        fixed_scalars = torch.cat((channel_means, magnitude_angle), dim=-1)
        return torch.cat((amplitudes, fixed_scalars), dim=-1).float()
    if encoding == "channel8":
        return channel_amplitude_prep(patches, patch).float()
    raise ValueError(f"unknown encoding {encoding!r}")
