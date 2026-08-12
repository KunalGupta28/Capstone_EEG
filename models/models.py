"""
models/models.py
================
Six deep learning model architectures for EEG Motor Imagery classification.
All models accept input shape: (batch, n_channels, n_times)

Changes from baseline:
  - Replaced vanilla RNN with Temporal Convolutional Network (TCN)
    (vanilla RNN averaged 50.2% on binary — coin-flip level)
  - All models accept optional `hidden_size` for adaptive sizing
    based on available training data (smaller datasets get smaller models)
  - All models accept **kwargs to silently ignore extra parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. EEGNet
# ---------------------------------------------------------------------------
class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al., 2018).
    Reference: https://arxiv.org/abs/1611.08024

    Input:  (B, C, T)  — channels-first
    Output: (B, num_classes)
    """

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, dropout: float = 0.5,
                 F1: int = 8, D: int = 2, F2: int = 16, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Adaptive kernel size based on sampling rate / signal length
        kern_len = min(64, n_times // 2)
        kern_pad = kern_len // 2

        # Block 1 — temporal convolution
        self.block1 = nn.Sequential(
            # (B, 1, C, T) → temporal conv
            nn.Conv2d(1, F1, kernel_size=(1, kern_len), padding=(0, kern_pad), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 1 — depthwise spatial convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2 — separable convolution
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size()
        self.classifier = nn.Linear(self._flat_size, num_classes)

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self.block1(dummy)
            x = self.depthwise(x)
            x = self.separable(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Spatial-Temporal Encoder Block
# ---------------------------------------------------------------------------
class SpatialTemporalEncoder(nn.Module):
    """
    Decoupled spatial-temporal projection block for raw EEG processing.
    Separates temporal frequency filters from spatial electrode filters.
    Input shape:  (B, C, T)
    Output shape: (B, out_channels, T_reduced)
    """
    def __init__(self, n_channels: int, n_times: int, out_channels: int = 40, kern_len: int = 25, pool_len: int = 4):
        super().__init__()
        # 1. Temporal filter (convolve time per channel independently)
        self.temporal = nn.Conv2d(
            1, out_channels, kernel_size=(1, kern_len), 
            padding=(0, kern_len // 2), bias=False
        )
        # 2. Spatial filter (fully-connect across electrodes)
        self.spatial = nn.Conv2d(
            out_channels, out_channels, kernel_size=(n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        
        # Downsample the temporal length using average pooling
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_len)) if pool_len > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.unsqueeze(1)          # (B, 1, C, T)
        x = self.temporal(x)        # (B, out_channels, C, T)
        x = self.spatial(x)         # (B, out_channels, 1, T)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)            # (B, out_channels, 1, T_reduced)
        x = x.squeeze(2)            # (B, out_channels, T_reduced)
        return x


# ---------------------------------------------------------------------------
# 2. CNN (ShallowConvNet)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """
    ShallowConvNet (Schirrmeister et al., 2017) optimized for motor imagery.
    Uses spatial-temporal decoupled convolutions and log-variance pooling.
    
    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 dropout: float = 0.5, hidden_size: int = 40, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        
        # Temporal conv (25 time steps = 100ms at 250Hz)
        self.temporal = nn.Conv2d(1, hidden_size, kernel_size=(1, 25), bias=False)
        # Spatial conv (combine electrodes)
        self.spatial = nn.Conv2d(hidden_size, hidden_size, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(hidden_size)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        
        self._flat_size = self._get_flat_size()
        self.classifier = nn.Linear(self._flat_size, num_classes)

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self.temporal(dummy)
            x = self.spatial(x)
            x = self.bn(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            x = self.dropout(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 3. TCN (replaces vanilla RNN — kept as "RNN" key in registry for compat)
# ---------------------------------------------------------------------------
class TCN(nn.Module):
    """
    Temporal Convolutional Network with dilated causal convolutions,
    prepended with a decoupled SpatialTemporalEncoder.
    
    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 dropout: float = 0.5, hidden_size: int = 128, **kwargs):
        super().__init__()
        c1 = min(40, hidden_size)
        # Prepend encoder to separate spatial electrode weights and temporal filtering
        self.encoder = SpatialTemporalEncoder(
            n_channels=n_channels, n_times=n_times, out_channels=c1, pool_len=4
        )
        
        c2 = min(64, hidden_size)
        c3 = min(128, hidden_size)

        self.tcn = nn.Sequential(
            # Dilated conv block 1 (receptive field: 7)
            nn.Conv1d(c1, c2, kernel_size=7, padding=3, dilation=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Dilated conv block 2 (receptive field: 7*2=14)
            nn.Conv1d(c2, c2, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Dilated conv block 3 (receptive field: 5*4=20)
            nn.Conv1d(c2, c3, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.tcn(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 4. LSTM
# ---------------------------------------------------------------------------
class LSTM(nn.Module):
    """
    Bidirectional 2-layer LSTM prepended with a decoupled SpatialTemporalEncoder
    to make recurrent sequences shorter, clean, and highly trainable.

    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        c1 = min(40, hidden_size)
        self.encoder = SpatialTemporalEncoder(
            n_channels=n_channels, n_times=n_times, out_channels=c1, pool_len=4
        )
        
        # Cap layers for small hidden sizes
        if hidden_size <= 32:
            num_layers = 1

        self.lstm = nn.LSTM(
            input_size=c1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)            # (B, c1, T_reduced)
        x = x.permute(0, 2, 1)         # (B, T_reduced, c1)
        _, (h_n, _) = self.lstm(x)     # h_n: (num_layers*2, B, hidden)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        h = self.dropout(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# 5. CNN + RNN (now CNN + TCN)
# ---------------------------------------------------------------------------
class CNNTCN(nn.Module):
    """
    Decoupled spatial-temporal CNN encoder producing a clean feature sequence
    fed into dilated temporal convolutions. Replaces CNN+RNN.

    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 hidden_size: int = 128, dropout: float = 0.5, **kwargs):
        super().__init__()
        c1 = min(40, hidden_size)
        self.encoder = SpatialTemporalEncoder(
            n_channels=n_channels, n_times=n_times, out_channels=c1, pool_len=2
        )
        
        c2 = min(64, hidden_size)
        c3 = min(128, hidden_size)

        # CNN encoder: 2 conv blocks
        self.net = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=7, padding=3),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            # Dilated temporal conv
            nn.Conv1d(c3, c3, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 6. CNN + LSTM
# ---------------------------------------------------------------------------
class CNNLSTM(nn.Module):
    """
    Decoupled spatial-temporal CNN encoder producing a clean feature sequence
    fed into a bidirectional LSTM.

    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 hidden_size: int = 128, dropout: float = 0.5, **kwargs):
        super().__init__()
        c1 = min(40, hidden_size)
        self.encoder = SpatialTemporalEncoder(
            n_channels=n_channels, n_times=n_times, out_channels=c1, pool_len=2
        )
        
        c2 = min(64, hidden_size)
        c3 = min(128, hidden_size)

        # CNN encoder: 2 conv blocks
        self.cnn = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=7, padding=3),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
        )
        
        # Cap LSTM layers for small hidden
        num_layers = 1 if hidden_size <= 32 else 2

        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)            # (B, c1, T//2)
        x = self.cnn(x)                # (B, c3, T//8)
        x = x.permute(0, 2, 1)         # (B, T//8, c3)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h = self.dropout(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "EEGNet":   EEGNet,
    "CNN":      CNN,
    "RNN":      TCN,        # TCN replaces vanilla RNN (kept key for compat)
    "LSTM":     LSTM,
    "CNN+RNN":  CNNTCN,     # CNN+TCN replaces CNN+RNN (kept key for compat)
    "CNN+LSTM": CNNLSTM,
}
