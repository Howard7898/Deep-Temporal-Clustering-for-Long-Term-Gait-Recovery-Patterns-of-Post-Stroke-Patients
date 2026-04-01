#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py
=========
Model architecture definitions for the DTCRP (Deep Temporal Clustering for Recovery Pattern) model.

This module contains three main classes:
  - TAE_encoder  : CNN + Bi-LSTM encoder that maps time-series input → latent space z
  - TAE_decoder  : Upsample + ConvTranspose1d decoder that reconstructs z → time-series x̂
  - TAE           : Full Temporal AutoEncoder (encoder + decoder)
  - ClusterNet   : Full end-to-end model (TAE + soft-assignment temporal clustering layer)

Reference:
  Teng et al., "Deep Temporal Clustering for Long-Term Gait Recovery Patterns
  of Post-Stroke Patients using Joint Kinematic Data", ICCAI 2025.
"""

import gc

import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering

from utils import compute_similarity


# ══════════════════════════════════════════════════════════
# 1. TAE Encoder
# ══════════════════════════════════════════════════════════

class TAE_encoder(nn.Module):
    """
    Temporal AutoEncoder — Encoder.

    Architecture:
        Input  (batch, 8 joints, T timesteps)
          → Conv1d + LeakyReLU + MaxPool1d
          → Bi-LSTM (hidden=50, bidirectional)  ×1
          → Bi-LSTM (hidden=1,  bidirectional)  ×1
          → Latent feature z  (batch, n_hidden, 1)

    The bidirectional LSTM outputs are summed across directions so that the
    effective hidden size remains `hidden_lstm_i` rather than 2×.

    Args:
        filter_1 (int):         Number of output channels for Conv1d (default: 50).
        filter_lstm (list[int]): Hidden sizes for the two Bi-LSTM layers (default: [50, 1]).
        pooling (int):          Pooling factor for MaxPool1d.
    """

    def __init__(self, filter_1: int, filter_lstm: list, pooling: int):
        super().__init__()

        self.hidden_lstm_1 = filter_lstm[0]   # 50
        self.hidden_lstm_2 = filter_lstm[1]   # 1
        self.pooling       = pooling
        self.n_hidden      = None              # Set after first forward pass

        # ── CNN Block ─────────────────────────────────────────────
        # Input:  (batch, 8, T)
        # Output: (batch, filter_1=50, n_hidden)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=8,          # 8 gait joint channels
                out_channels=filter_1,  # 50 feature maps
                kernel_size=10,
                stride=1,
                padding=5,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.pooling),
        )

        # ── Bi-LSTM Layer 1 ───────────────────────────────────────
        # Input:  (batch, n_hidden, 50)    [after CNN permute]
        # Output: (batch, n_hidden, 100)   [bidirectional → summed to 50]
        self.lstm_1 = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
            bidirectional=True,
        )

        # ── Bi-LSTM Layer 2 ───────────────────────────────────────
        # Input:  (batch, n_hidden, 50)
        # Output: (batch, n_hidden, 2)    [bidirectional → summed to 1]
        self.lstm_2 = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input time-series into latent representation z.

        Args:
            x (Tensor): Input, shape (batch, 8, T).

        Returns:
            features (Tensor): Latent vector, shape (batch, n_hidden, 1).
        """
        # CNN: (batch, 8, T) → (batch, 50, n_hidden)
        out_cnn = self.conv_layer(x)

        # Permute for LSTM: (batch, 50, n_hidden) → (batch, n_hidden, 50)
        out_cnn = out_cnn.permute(0, 2, 1)

        # Bi-LSTM 1: (batch, n_hidden, 50) → sum bidirectional → (batch, n_hidden, 50)
        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1 = torch.sum(
            out_lstm1.view(out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1),
            dim=2,
        )

        # Bi-LSTM 2: (batch, n_hidden, 50) → sum bidirectional → (batch, n_hidden, 1)
        features, _ = self.lstm_2(out_lstm1)
        features = torch.sum(
            features.view(features.shape[0], features.shape[1], 2, self.hidden_lstm_2),
            dim=2,
        )

        # Record n_hidden on first pass (used by TAE to configure decoder)
        if self.n_hidden is None:
            self.n_hidden = features.shape[1]

        return features  # (batch, n_hidden, 1)


# ══════════════════════════════════════════════════════════
# 2. TAE Decoder
# ══════════════════════════════════════════════════════════

class TAE_decoder(nn.Module):
    """
    Temporal AutoEncoder — Decoder.

    Reconstructs the original time-series from the latent representation z.

    Architecture:
        z  (batch, n_hidden, 1)
          → Upsample to (batch, n_hidden, 8 × pooling)
          → ConvTranspose1d (kernel=10, padding=4)
          → Reshape → x̂  (batch, 8, 600)

    Args:
        n_hidden (int): Size of the latent dimension (output of encoder).
        pooling  (int): Pooling factor used in the encoder (for size matching).
    """

    def __init__(self, n_hidden: int, pooling: int):
        super().__init__()

        self.pooling  = pooling
        self.n_hidden = n_hidden

        # Upsample latent vector back to encoder's pre-pool temporal resolution
        self.up_layer = nn.Upsample(size=8 * pooling)

        # Transpose convolution to reconstruct fine-grained temporal signal
        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.n_hidden,
            out_channels=self.n_hidden,
            kernel_size=10,
            stride=1,
            padding=int((10 - 1) / 2),  # 'same' padding: (kernel_size - 1) / 2
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct time-series from latent features.

        Args:
            features (Tensor): Latent vector, shape (batch, n_hidden, 1).

        Returns:
            out_deconv (Tensor): Reconstructed data, shape (batch, 8, 600).
        """
        # Upsample: (batch, n_hidden, 1) → (batch, n_hidden, 8×pooling)
        upsampled = self.up_layer(features)

        # ConvTranspose1d then trim to exact length
        out_deconv = self.deconv_layer(upsampled)[:, :, : 8 * self.pooling].contiguous()

        # Reshape to match original input format: (batch, 8 joints, 600 timesteps)
        out_deconv = out_deconv.view(out_deconv.shape[0], 8, 600)

        return out_deconv


# ══════════════════════════════════════════════════════════
# 3. Full Temporal AutoEncoder (TAE)
# ══════════════════════════════════════════════════════════

class TAE(nn.Module):
    """
    Temporal AutoEncoder (TAE) — Full encoder-decoder model.

    Combines TAE_encoder and TAE_decoder. Used for:
      (a) Standalone pretraining via MSE reconstruction loss.
      (b) Feature extraction inside ClusterNet.

    Args:
        args:          Parsed arguments (device, serie_size, etc.).
        pool (int):    MaxPool1D pooling factor.
        filter_1 (int):         Conv1d output channels (default: 50).
        filter_lstm (list[int]): Bi-LSTM hidden sizes (default: [50, 1]).
    """

    def __init__(self, args, pool: int, filter_1: int = 50, filter_lstm: list = [50, 1]):
        super().__init__()

        self.pooling     = int(pool)
        self.filter_1    = filter_1
        self.filter_lstm = filter_lstm

        # Build encoder
        self.tae_encoder = TAE_encoder(
            filter_1=self.filter_1,
            filter_lstm=self.filter_lstm,
            pooling=self.pooling,
        )

        # Infer n_hidden by running a dummy forward pass
        n_hidden      = self._get_hidden(args.serie_size, args.device)
        args.n_hidden = n_hidden

        # Build decoder using inferred n_hidden
        self.tae_decoder = TAE_decoder(n_hidden=args.n_hidden, pooling=self.pooling)

    def _get_hidden(self, serie_size: int, device: torch.device) -> int:
        """
        Determine the latent dimension n_hidden by running a dummy forward pass.

        This avoids hardcoding n_hidden and allows flexible input lengths.

        Args:
            serie_size (int):    Length of one time-series (T).
            device:              Torch device.

        Returns:
            n_hidden (int): Temporal dimension of the latent space.
        """
        dummy_input = torch.randn((1, 8, serie_size)).to(device)
        probe = TAE_encoder(
            filter_1=self.filter_1,
            filter_lstm=self.filter_lstm,
            pooling=self.pooling,
        ).to(device)

        with torch.no_grad():
            _ = probe(dummy_input)

        n_hid = probe.n_hidden

        # Clean up GPU memory
        del probe, dummy_input
        gc.collect()
        torch.cuda.empty_cache()

        return n_hid

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encode and decode the input time-series.

        Args:
            x (Tensor): Input, shape (batch, 8, T).

        Returns:
            features   (Tensor): Squeezed latent vector, shape (batch, n_hidden).
            out_deconv (Tensor): Reconstructed output,   shape (batch, 8, 600).
        """
        features   = self.tae_encoder(x)          # (batch, n_hidden, 1)
        out_deconv = self.tae_decoder(features)    # (batch, 8, 600)
        return features.squeeze(2), out_deconv     # squeeze last dim of features


# ══════════════════════════════════════════════════════════
# 4. ClusterNet (TAE + Temporal Clustering Layer)
# ══════════════════════════════════════════════════════════

class ClusterNet(nn.Module):
    """
    Full DTCRP Model: Temporal AutoEncoder + Temporal Clustering Layer.

    The clustering layer assigns each latent vector z to soft cluster memberships
    using a Student's t-distribution, following the DEC / DTC formulation.

    Training jointly minimizes:
      - MSE reconstruction loss    (via TAE)
      - KL divergence clustering loss (z vs. learned centroids)

    Args:
        args:          Parsed arguments (alpha, n_hidden, device, weight paths).
        pool (int):    MaxPool1D pooling factor.
        n_clusters (int): Number of clusters K.
    """

    def __init__(self, args, pool: int, n_clusters: int):
        super().__init__()

        # ── Load pretrained TAE ───────────────────────────────────
        self.tae = TAE(args, pool)
        self.tae.load_state_dict(
            torch.load(args.path_weights_ae, map_location=args.device)
        )

        # ── Clustering parameters ─────────────────────────────────
        self.alpha_     = args.alpha        # Degrees of freedom for t-distribution
        self.centr_size = args.n_hidden     # Dimensionality of cluster centroids
        self.n_clusters = n_clusters
        self.device     = args.device

    def init_centroids(self, x: torch.Tensor) -> tuple:
        """
        Initialize cluster centroids using Agglomerative Clustering.

        Runs encoder on the full dataset to get latent vectors, then applies
        complete-linkage hierarchical clustering (with correlation distance)
        to set initial centroid positions. These are registered as learnable
        nn.Parameter objects.

        Args:
            x (Tensor): Full input dataset, shape (N, 8, T).

        Returns:
            z           (Tensor):    All latent vectors, shape (N, n_hidden).
            z_np        (np.ndarray): CPU copy of z.
            centroids_  (Tensor):    Initialized centroid matrix, shape (K, n_hidden).
        """
        z, _ = self.tae(x.squeeze().detach())
        z_np  = z.detach().cpu()

        # Agglomerative clustering with precomputed correlation distance matrix
        assignments = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage="complete",
            affinity="precomputed",
        ).fit_predict(compute_similarity(z_np, z_np))

        # Compute mean latent vector per cluster as initial centroid
        centroids_ = torch.zeros((self.n_clusters, self.centr_size), device=self.device)
        for cluster_ in range(self.n_clusters):
            index_cluster = [k for k, idx in enumerate(assignments) if idx == cluster_]
            centroids_[cluster_] = torch.mean(z.detach()[index_cluster], dim=0)

        # Register as learnable parameter (updated during end-to-end training)
        self.centroids = nn.Parameter(centroids_)

        return z, z_np, centroids_

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through encoder + clustering layer.

        Computes:
          Q : Soft assignment distribution   (Student's t-distribution)
          P : Target/auxiliary distribution  (sharpened Q, used for KL loss)

        Equations (DTC formulation):
          Q_ij = (1 + ||z_i - µ_j||² / α)^(-(α+1)/2)
                 ─────────────────────────────────────
                 Σ_j' (1 + ||z_i - µ_j'||² / α)^(-(α+1)/2)

          P_ij = Q_ij² / Σ_i Q_ij
                 ──────────────────
                 Σ_j' P_ij'

        Args:
            x (Tensor): Input data, shape (batch, 8, T).

        Returns:
            z         (Tensor): Latent features,             shape (batch, n_hidden).
            x_reconstr(Tensor): Reconstructed time-series,   shape (batch, 8, 600).
            Q         (Tensor): Soft cluster assignments,    shape (batch, K).
            P         (Tensor): Target distribution,         shape (batch, K).
        """
        # Encode and decode
        z, x_reconstr = self.tae(x)

        # Compute correlation-based similarity between z and all centroids
        similarity = compute_similarity(z, self.centroids)  # (batch, K)

        # ── Soft assignment Q (Student's t-distribution) ──────────
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        Q = Q / torch.sum(Q, dim=1, keepdim=True)           # Row-normalize

        # ── Target distribution P (sharpen Q) ────────────────────
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0, keepdim=True)
        P = P / torch.sum(P, dim=1, keepdim=True)           # Row-normalize

        return z, x_reconstr, Q, P
