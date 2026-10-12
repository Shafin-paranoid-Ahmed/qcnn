"""Dataset-free portability and differentiability check; no result files written."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch
from patch_encode import load_image_patches
from qcnn_model import build_qcnn


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    pixels = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "synthetic.png"
        Image.fromarray(pixels).save(path)
        for encoding, attention in (("amplitude10", False), ("amplitude10_mag", True)):
            inputs = load_image_patches(str(path), encoding=encoding).unsqueeze(0)
            model = build_qcnn(5, encoding=encoding, attention=attention)
            output = model(inputs, return_attention=attention)
            logits = output[0] if attention else output
            assert logits.shape == (1, 5) and torch.isfinite(logits).all()
            if attention:
                weights = output[1]
                assert weights.shape == (1, 196) and torch.isfinite(weights).all()
            torch.nn.functional.cross_entropy(logits, torch.tensor([2])).backward()
            for key, value in model.named_parameters():
                assert value.grad is not None and torch.isfinite(value.grad).all(), key
            print(f"PASS {encoding} attention={attention}: finite logits and gradients")


if __name__ == "__main__":
    main()
