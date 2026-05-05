"""
models/models.py
================
Six deep learning model architectures for EEG Motor Imagery classification.
All models accept input shape: (batch, n_channels, n_times)
Output: single logit (used with BCEWithLogitsLoss)
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
    Output: (B, 1)     — single logit
    """

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, dropout: float = 0.5,
                 F1: int = 8, D: int = 2, F2: int = 16, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1 — temporal convolution
        self.block1 = nn.Sequential(
            # (B, 1, C, T) → temporal conv
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
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
# 2. CNN
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """
    Three-block 1-D CNN for EEG classification.

    Input:  (B, C, T)
    Output: (B, 1)
    """

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),  # global avg pool → (B, 256, 1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)          # (B, 256, 1)
        x = x.view(x.size(0), -1) # (B, 256)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 3. RNN
# ---------------------------------------------------------------------------
class RNN(nn.Module):
    """
    Bidirectional 2-layer vanilla RNN.

    EEG input treated as a temporal sequence: each time step has C features.
    Input:  (B, C, T) → transposed to (B, T, C) inside forward()
    Output: (B, 1)
    """

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        _, h_n = self.rnn(x)           # h_n: (num_layers*2, B, hidden)
        # Concat last forward & backward hidden from top layer
        h_fwd = h_n[-2]               # (B, hidden)
        h_bwd = h_n[-1]               # (B, hidden)
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden*2)
        h = self.dropout(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# 4. LSTM
# ---------------------------------------------------------------------------
class LSTM(nn.Module):
    """
    Bidirectional 2-layer LSTM.

    Input:  (B, C, T)
    Output: (B, 1)
    """

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)         # (B, T, C)
        _, (h_n, _) = self.lstm(x)     # h_n: (num_layers*2, B, hidden)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        h = self.dropout(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# 5. CNN + RNN
# ---------------------------------------------------------------------------
class CNNRNN(nn.Module):
    """
    CNN encoder producing a feature sequence fed into a bidirectional RNN.

    Input:  (B, C, T)
    Output: (B, 1)
    """

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, hidden_size: int = 128,
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        # CNN encoder: 2 conv blocks, preserving time axis
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
        )
        # RNN over the time-compressed feature sequence
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)            # (B, 128, T//4)
        x = x.permute(0, 2, 1)        # (B, T//4, 128)
        _, h_n = self.rnn(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h = self.dropout(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# 6. CNN + LSTM
# ---------------------------------------------------------------------------
class CNNLSTM(nn.Module):
    """
    CNN encoder producing a feature sequence fed into a bidirectional LSTM.

    Input:  (B, C, T)
    Output: (B, 1)
    """

    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4, hidden_size: int = 128,
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)            # (B, 128, T//4)
        x = x.permute(0, 2, 1)        # (B, T//4, 128)
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
    "RNN":      RNN,
    "LSTM":     LSTM,
    "CNN+RNN":  CNNRNN,
    "CNN+LSTM": CNNLSTM,
}
