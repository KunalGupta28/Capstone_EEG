"""
models/models.py
================
Seven deep learning model architectures for EEG Motor Imagery classification.
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
                 F1: int = 8, D: int = 2, F2: int = 16, sampling_rate: float = 100.0, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Adaptive kernel size: strictly preserve old logic for binary datasets
        # to guarantee no degradation. For multiclass, use fs // 2 (e.g. 125 for 250Hz).
        if num_classes <= 2:
            kern_len = min(64, n_times // 2)
        else:
            kern_len = int(sampling_rate) // 2
            
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

        # Fix temporal dim to a constant size before the classifier,
        # so parameter count is independent of input T (avoids overfitting
        # on long-signal datasets like BCI-4-2a with T=660).
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        self._flat_size = F2 * 8
        self.classifier = nn.Linear(self._flat_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.adaptive_pool(x)
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

        # Fix temporal dim to a constant size before the classifier,
        # so parameter count is independent of input T.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        self._flat_size = hidden_size * 8
        self.classifier = nn.Linear(self._flat_size, num_classes)

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
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 3. DeepConvNet
# ---------------------------------------------------------------------------
class DeepConvNet(nn.Module):
    """
    DeepConvNet (Schirrmeister et al., 2017).
    Standard deep convolutional network for EEG.
    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False),
            nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 2))
        self.classifier = nn.Linear(200 * 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
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
# 5. EEG Conformer (Convolutional Transformer)
# ---------------------------------------------------------------------------
class EEGConformer(nn.Module):
    """
    Simplified EEG Conformer / Convolutional Transformer.
    Extracts spatial-temporal features via CNN, then uses Self-Attention.
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 dropout: float = 0.5, hidden_size: int = 40, **kwargs):
        super().__init__()
        
        # Convolutional module (Spatial-Temporal)
        self.temporal = nn.Conv2d(1, hidden_size, kernel_size=(1, 25), padding=(0, 12), bias=False)
        self.spatial = nn.Conv2d(hidden_size, hidden_size, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(hidden_size)
        self.pool = nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 7))
        
        # Transformer module
        self.d_model = hidden_size
        self.n_heads = 8 if self.d_model % 8 == 0 else (4 if self.d_model % 4 == 0 else 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.pool(x)
        
        # x shape: (B, hidden_size, 1, T_reduced)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T_reduced, hidden_size)
        
        x = self.transformer(x)
        x = x.permute(0, 2, 1) # (B, hidden_size, T_reduced)
        x = self.adaptive_pool(x)
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
# 7. EEG Graph Convolutional Network (ST-GCN)
# ---------------------------------------------------------------------------
class GraphConvBlock(nn.Module):
    """
    A single Spatial-Temporal Graph Convolution block.
    
    Spatial: Graph convolution via learnable adjacency matrix A.
             X_out = ReLU(BN(A @ X @ W))
    Temporal: 1D convolution along time axis after graph mixing.
    
    Includes a residual connection to prevent the "over-smoothing" problem
    that plagues deep GCNs (where all node features converge after stacking).
    """
    def __init__(self, in_features: int, out_features: int, n_nodes: int,
                 temporal_kernel: int = 9, dropout: float = 0.5):
        super().__init__()
        
        # Learnable adjacency matrix (raw, before symmetrisation)
        self.A_raw = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01)
        
        # Graph convolution weights
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.bn_graph = nn.BatchNorm1d(n_nodes)
        
        # Temporal convolution (operates along time after graph mixing)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(out_features, out_features,
                      kernel_size=temporal_kernel,
                      padding=temporal_kernel // 2, bias=False),
            nn.BatchNorm1d(out_features),
        )
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection (when in_features != out_features)
        self.residual = (nn.Linear(in_features, out_features, bias=False)
                         if in_features != out_features else nn.Identity())
    
    def _get_adjacency(self):
        """Create a symmetric, normalised adjacency matrix with self-loops."""
        # Symmetrise: A = (A_raw + A_raw^T) / 2
        A_sym = (self.A_raw + self.A_raw.T) / 2.0
        # Add self-loops (identity)
        A = A_sym + torch.eye(A_sym.size(0), device=A_sym.device)
        # Row-normalise so messages are averaged, not summed
        D_inv = 1.0 / (A.sum(dim=1, keepdim=True) + 1e-6)
        return A * D_inv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T, F)  where N=nodes(channels), T=time, F=features
        returns: (B, N, T, F_out)
        """
        B, N, T, F_in = x.shape
        residual = self.residual(x)  # (B, N, T, F_out)
        
        # --- Spatial: Graph Convolution ---
        # Feature projection: (B, N, T, F) -> (B, N, T, F_out)
        x = self.W(x)
        F_out = x.shape[-1]
        # Reshape for adjacency multiplication: (B*T, N, F_out)
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, F_out)
        A = self._get_adjacency()  # (N, N)
        x = torch.bmm(A.unsqueeze(0).expand(B * T, -1, -1), x)  # (B*T, N, F_out)
        # Reshape back: (B, N, T, F_out)
        x = x.reshape(B, T, N, F_out).permute(0, 2, 1, 3)
        
        # BatchNorm over node dimension
        x = x.reshape(B * T, N, F_out)
        x = self.bn_graph(x)
        x = x.reshape(B, N, T, F_out)
        
        x = self.elu(x)
        
        # --- Temporal: 1D Conv along time ---
        # Reshape: (B*N, F_out, T)
        x = x.reshape(B * N, T, F_out).permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = self.elu(x)
        # Reshape back: (B, N, T, F_out)
        x = x.permute(0, 2, 1).reshape(B, N, T, F_out)
        
        # Residual + dropout
        x = self.dropout(x + residual)
        return x


class EEG_GCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for EEG classification.
    
    Key design decisions:
    - Learnable symmetric adjacency matrix: adapts to any channel count
      (critical for LS-BJOA which dynamically selects subsets).
    - Residual connections in each GCN block to prevent over-smoothing.
    - Temporal embedding via 1D conv before graph operations.
    
    Input:  (B, C, T) — C channels, T time steps
    Output: (B, num_classes)
    """
    def __init__(self, n_channels: int, n_times: int, num_classes: int = 4,
                 dropout: float = 0.5, hidden_size: int = 64, **kwargs):
        super().__init__()
        
        feat_dim = min(32, hidden_size)
        gcn_dim  = min(64, hidden_size)
        
        # Temporal embedding: project raw time-series into feature space per channel
        # Input: (B, C, T) -> treat each channel independently
        self.temporal_embed = nn.Sequential(
            nn.Conv1d(1, feat_dim, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ELU(),
            nn.AvgPool1d(4),  # downsample time by 4x
        )
        
        # Two stacked ST-GCN blocks with increasing feature dimension
        self.gcn1 = GraphConvBlock(feat_dim, gcn_dim, n_nodes=n_channels,
                                   temporal_kernel=9, dropout=dropout)
        self.gcn2 = GraphConvBlock(gcn_dim, gcn_dim, n_nodes=n_channels,
                                   temporal_kernel=5, dropout=dropout)
        
        # Global pooling: average over both nodes and time
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(gcn_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        
        # --- Temporal Embedding ---
        # Process each channel independently through the same temporal conv
        # Reshape: (B*C, 1, T)
        x = x.reshape(B * C, 1, T)
        x = self.temporal_embed(x)       # (B*C, feat_dim, T_reduced)
        feat_dim = x.shape[1]
        T_red = x.shape[2]
        # Reshape to graph format: (B, C, T_reduced, feat_dim)
        x = x.reshape(B, C, feat_dim, T_red).permute(0, 1, 3, 2)
        
        # --- Graph Convolution Blocks ---
        x = self.gcn1(x)  # (B, C, T_reduced, gcn_dim)
        x = self.gcn2(x)  # (B, C, T_reduced, gcn_dim)
        
        # --- Readout ---
        # Average over nodes: (B, T_reduced, gcn_dim)
        x = x.mean(dim=1)
        # Pool over time: (B, gcn_dim)
        x = x.permute(0, 2, 1)  # (B, gcn_dim, T_reduced)
        x = self.pool(x).squeeze(-1)  # (B, gcn_dim)
        
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "EEGNet":      EEGNet,
    "CNN":         CNN,
    "LSTM":        LSTM,
    "CNN+LSTM":    CNNLSTM,
    "DeepConvNet": DeepConvNet,
    "Conformer":   EEGConformer,
    "GraphNet":    EEG_GCN,
}

