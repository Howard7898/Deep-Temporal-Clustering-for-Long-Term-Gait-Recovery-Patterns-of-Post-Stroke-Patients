#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========
Utility functions for the DTCRP clustering model.

Provides:
  - compute_CE         : Complexity Estimate (CE) of a sequence
  - compute_similarity : Pearson correlation-based distance between latent vectors and centroids

These distance metrics are used in the Temporal Clustering Layer to compute
soft cluster assignments via a Student's t-distribution.

Reference:
  Teng et al., "Deep Temporal Clustering for Long-Term Gait Recovery Patterns
  of Post-Stroke Patients using Joint Kinematic Data", ICCAI 2025.
"""

import torch


def compute_CE(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Complexity Estimate (CE) for each sample in a batch.

    CE measures the "roughness" of a sequence by computing the L2 norm of
    consecutive differences. Higher CE indicates more complex (irregular) sequences.

    Formula:
        CE(x_i) = sqrt( Σ_t (x_i[t+1] - x_i[t])² )

    Args:
        x (Tensor): Input sequences, shape (n, n_hidden).

    Returns:
        Tensor: CE value per sample, shape (n, 1).
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))


def compute_similarity(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation-based distance between latent vectors and cluster centroids.

    This is used as the distance metric in the Temporal Clustering Layer's
    Student's t-distribution soft assignment formula.

    Distance formula:
        d(z_i, µ_j) = sqrt( 2 × (1 − r(z_i, µ_j)) )

    where r(·, ·) is the Pearson correlation coefficient:
        r(z_i, µ_j) = [ E(z_i · µ_j) − mean(z_i) × mean(µ_j) ]
                       ──────────────────────────────────────────
                              std(z_i) × std(µ_j)

    A distance of 0 means perfect positive correlation (identical patterns),
    and a distance of 2 means perfect negative correlation.

    Args:
        z         (Tensor): Latent feature matrix,  shape (batch_size, n_hidden).
        centroids (Tensor): Cluster centroid matrix, shape (n_clusters, n_hidden).

    Returns:
        Tensor: Pairwise distance matrix, shape (batch_size, n_clusters).
                Each entry [i, j] is the distance from sample i to centroid j.
    """
    n_clusters = centroids.shape[0]
    n_hidden   = centroids.shape[1]
    bs         = z.shape[0]

    # ── Standard deviations ───────────────────────────────────────
    # Expand to (batch_size, n_clusters) for pairwise computation
    std_z   = torch.std(z,         dim=1).unsqueeze(1).expand(bs, n_clusters)
    std_cen = torch.std(centroids, dim=1).unsqueeze(0).expand(bs, n_clusters)

    # ── Means ─────────────────────────────────────────────────────
    mean_z   = torch.mean(z,         dim=1).unsqueeze(1).expand(bs, n_clusters)
    mean_cen = torch.mean(centroids, dim=1).unsqueeze(0).expand(bs, n_clusters)

    # ── Cross-term E[z_i · µ_j] ──────────────────────────────────
    # Expand z and centroids to (batch_size, n_clusters, n_hidden) for element-wise product
    z_expand   = z.unsqueeze(1).expand(bs, n_clusters, n_hidden)
    cen_expand = centroids.unsqueeze(0).expand(bs, n_clusters, n_hidden)

    # Mean of element-wise product along the feature dimension
    prod_expec = torch.mean(z_expand * cen_expand, dim=2)  # (batch_size, n_clusters)

    # ── Pearson correlation coefficient r ─────────────────────────
    pearson_corr = (prod_expec - mean_z * mean_cen) / (std_z * std_cen)

    # ── Correlation-based distance ────────────────────────────────
    # Range: [0, 2] where 0 = identical, 2 = perfectly anti-correlated
    return torch.sqrt(2 * (1 - pearson_corr))
