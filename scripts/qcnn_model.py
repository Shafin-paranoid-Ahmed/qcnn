"""Shared patch circuit and measured-between-levels quantum hierarchy (spec 6)."""
from __future__ import annotations
import math
import pennylane as qml
import torch
from torch import nn

class PatchAttentionGate(nn.Module):
    """Shared scalar scorer and mean-preserving softmax gate over patches."""
    def __init__(self, feature_dim):
        super().__init__()
        self.scorer = nn.Linear(feature_dim, 1)

    def forward(self, features, return_attention=False):
        if features.ndim != 3 or features.shape[1] != 196:
            raise ValueError(f"expected [batch,196,features], got {tuple(features.shape)}")
        attention = torch.softmax(self.scorer(features).squeeze(-1), dim=1)
        gated = features * (196.0 * attention).unsqueeze(-1)
        return (gated, attention) if return_attention else gated

def _ring_layer(weights, wires):
    for wire, pair in zip(wires, weights):
        qml.RY(pair[0], wires=wire)
        qml.RZ(pair[1], wires=wire)
    for index, wire in enumerate(wires):
        qml.CNOT(wires=[wire, wires[(index + 1) % len(wires)]])

class PatchQCNN(nn.Module):
    """Raw-pixel 16x16 patch QCNN with trainable quantum pooling at both levels."""
    def __init__(self, n_classes, d_conv=2, k_readout=4, reupload_L=1,
                 encoding="amplitude10", attention=False):
        super().__init__()
        if not 1 <= reupload_L <= 3 or d_conv not in (2, 3, 4) or k_readout != 4:
            raise ValueError("pilot supports d_conv in {2,3,4}, L in {1,2,3}, and four readouts")
        if encoding not in ("amplitude10", "amplitude10_mag", "channel8"):
            raise ValueError("encoding must be amplitude10, amplitude10_mag, or channel8")
        self.d_conv, self.reupload_L, self.encoding = d_conv, reupload_L, encoding
        self.attention_enabled = bool(attention)
        patch_width = 10 if encoding in ("amplitude10", "amplitude10_mag") else 8
        patch_wires, pool1_wires, pool2_wires = list(range(patch_width)), list(range(4)), list(range(8))
        patch_device = qml.device("default.qubit", wires=patch_wires)
        pool1_device = qml.device("default.qubit", wires=pool1_wires)
        pool2_device = qml.device("default.qubit", wires=pool2_wires)

        @qml.qnode(patch_device, interface="torch", diff_method="backprop")
        def patch_qnode(inputs, weights):
            amplitude_inputs = inputs[..., :1024] if encoding == "amplitude10_mag" else inputs
            qml.AmplitudeEmbedding(amplitude_inputs, wires=patch_wires, normalize=True, pad_with=0.0)
            if encoding == "amplitude10_mag":
                for index, wire in enumerate(patch_wires[-4:]):
                    qml.RY(inputs[..., 1024 + index], wires=wire)
            for round_index in range(reupload_L):
                if round_index:
                    qml.AngleEmbedding(inputs[..., :patch_width] * math.pi, wires=patch_wires, rotation="Y")
                for layer in weights[round_index]:
                    _ring_layer(layer, patch_wires)
            return [qml.expval(qml.Z(wire)) for wire in patch_wires[-4:]]

        @qml.qnode(pool1_device, interface="torch", diff_method="backprop")
        def pool1_qnode(inputs, weights):
            for start in range(0, 16, 4):
                qml.AngleEmbedding(inputs[..., start:start + 4], wires=pool1_wires, rotation="Y")
            for layer in weights:
                _ring_layer(layer, pool1_wires)
            return [qml.expval(qml.Z(wire)) for wire in pool1_wires]

        @qml.qnode(pool2_device, interface="torch", diff_method="backprop")
        def pool2_qnode(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=pool2_wires, normalize=True, pad_with=0.0)
            for layer in weights:
                _ring_layer(layer, pool2_wires)
            return [qml.expval(qml.Z(wire)) for wire in pool2_wires]

        self.patch_layer = qml.qnn.TorchLayer(patch_qnode, {"weights": (reupload_L, d_conv, patch_width, 2)})
        self.attention = PatchAttentionGate(4) if self.attention_enabled else None
        self.pool1_layer = qml.qnn.TorchLayer(pool1_qnode, {"weights": (2, 4, 2)})
        self.pool2_layer = qml.qnn.TorchLayer(pool2_qnode, {"weights": (2, 8, 2)})
        self.readout = nn.Linear(8, n_classes)
        expected_params = 138 if self.attention_enabled else 133
        if encoding == "amplitude10_mag" and reupload_L == 1 and self.trainable_parameter_count != expected_params:
            raise AssertionError(f"amplitude10_mag L=1 must have exactly {expected_params} trainable parameters, got {self.trainable_parameter_count}")

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, patches, return_attention=False):
        expected = {"amplitude10": (196, 1024), "amplitude10_mag": (196, 1028),
                    "channel8": (196, 3, 256)}[self.encoding]
        if tuple(patches.shape[1:]) != expected:
            raise ValueError(f"expected [batch,{','.join(map(str, expected))}] raw-pixel patches, got {tuple(patches.shape)}")
        batch = patches.shape[0]
        if self.encoding in ("amplitude10", "amplitude10_mag"):
            patch_features = self.patch_layer(patches.reshape(-1, expected[-1])).reshape(batch, 14, 14, 4)
        else:
            patch_features = self.patch_layer(patches.reshape(-1, 256)).reshape(batch, 196, 3, 4).mean(dim=2).reshape(batch, 14, 14, 4)
        flat_features = patch_features.reshape(batch, 196, 4)
        attention = None
        if self.attention is not None:
            flat_features, attention = self.attention(flat_features, return_attention=True)
        elif return_attention:
            raise ValueError("attention weights requested from a model with attention disabled")
        patch_features = flat_features.reshape(batch, 14, 14, 4)
        cells = patch_features.reshape(batch, 7, 2, 7, 2, 4).permute(0, 1, 3, 2, 4, 5).reshape(-1, 16)
        level1 = self.pool1_layer(cells).reshape(batch, 7, 7, 4)
        global_features = self.pool2_layer(level1.reshape(batch, 196))
        logits = self.readout(global_features)
        return (logits, attention) if return_attention else logits

def build_qcnn(n_classes, d_conv=2, k_readout=4, reupload_L=1,
               encoding="amplitude10", attention=False):
    return PatchQCNN(n_classes, d_conv=d_conv, k_readout=k_readout,
                     reupload_L=reupload_L, encoding=encoding, attention=attention)
