#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtc_longitudinal.py
====================
Main execution script for the Deep Temporal Clustering for Recovery Pattern (DTCRP) model.

This script:
  1. Loads longitudinal gait kinematic data (joint angles + angular velocities)
  2. Pretrains a Temporal AutoEncoder (TAE) via MSE reconstruction loss
  3. Trains the full ClusterNet (TAE + Clustering Layer) end-to-end
  4. Evaluates clustering quality using the Silhouette Coefficient
  5. Saves results (xlsx), model weights (.pth), and cluster assignments (.pkl)

Reference:
  Teng et al., "Deep Temporal Clustering for Long-Term Gait Recovery Patterns
  of Post-Stroke Patients using Joint Kinematic Data", ICCAI 2025.
  DOI: 10.1109/ICCAI66501.2025.00105

Usage:
  $ python dtc_longitudinal.py

Author: howard
"""

# ─────────────────────────────────────────────────────────
# Standard library imports
# ─────────────────────────────────────────────────────────
import os
import time
import math
import pickle
import random
import itertools as it

# ─────────────────────────────────────────────────────────
# Third-party imports
# ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from pandas import ExcelWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import mplcursors

from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import KFold, TimeSeriesSplit

# ─────────────────────────────────────────────────────────
# Local module imports
# ─────────────────────────────────────────────────────────
from config1 import get_arguments
from models import ClusterNet, TAE
from load_long_custom_data import get_long_loader


# ══════════════════════════════════════════════════════════
# SECTION 1: Reproducibility
# ══════════════════════════════════════════════════════════

def fix_seed(SEED: int) -> None:
    """
    Fix all random seeds for full reproducibility across runs.

    Note:
        Setting `cudnn.deterministic = True` may reduce GPU throughput.
        Use only when exact reproducibility is required.

    Args:
        SEED (int): Integer seed value (e.g., 42).
    """
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True   # Ensures determinism; may slow down training
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════
# SECTION 2: TAE Pretraining
# ══════════════════════════════════════════════════════════

def pretrain_autoencoder(args, lr_ae: float, epochs_ae: int, pool: int, verbose: bool = True):
    """
    Pretrain the Temporal AutoEncoder (TAE) using masked MSE reconstruction loss.

    The TAE is trained independently before the full clustering model to learn
    a meaningful latent representation of the gait time-series data.
    Missing sessions are handled via a binary mask (1 = valid, 0 = missing).

    Args:
        args:        Parsed argument object (device, paths, etc.)
        lr_ae:       Learning rate for the Adam optimizer.
        epochs_ae:   Number of pretraining epochs.
        pool:        MaxPool1D pooling size passed to TAE architecture.
        verbose:     If True, prints per-epoch loss values.

    Returns:
        all_pretrain_loss (list[float]): Mean MSE loss per epoch.
        inputs (Tensor):                Last batch of input data (for reference).
    """
    all_pretrain_loss = []
    print("Pretraining autoencoder...\n")

    # ── Model setup ──────────────────────────────────────
    tae = TAE(args, pool)
    tae = tae.to(args.device)

    loss_ae = nn.MSELoss(reduction='none')  # Element-wise loss for mask application
    optimizer = torch.optim.Adam(tae.parameters(), lr=lr_ae)
    tae.train()

    # ── Training loop ─────────────────────────────────────
    for epoch in range(epochs_ae):
        all_loss = 0

        for batch_idx, (inputs, _, mask) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            mask   = mask.to(args.device)

            optimizer.zero_grad()
            z, x_reconstr = tae(inputs)

            # Apply mask: only compute loss on valid (non-missing) time points
            loss_mse = loss_ae(inputs.squeeze(1), x_reconstr)
            loss_mse = torch.sum(loss_mse * mask) / torch.sum(mask)

            loss_mse.backward()
            optimizer.step()

            all_loss += loss_mse.item()

        epoch_loss = all_loss / (batch_idx + 1)
        if verbose:
            print(f"  [Pretrain] Epoch {epoch:>3d}/{epochs_ae} | Loss: {epoch_loss:.5f}")
        all_pretrain_loss.append(epoch_loss)

    print("Pretraining complete.\n")

    # Save pretrained TAE weights for later use in ClusterNet
    torch.save(tae.state_dict(), args.path_weights_ae)

    return all_pretrain_loss, inputs


# ══════════════════════════════════════════════════════════
# SECTION 3: Centroid Initialization
# ══════════════════════════════════════════════════════════

def initalize_centroids(X: np.ndarray):
    """
    Initialize cluster centroids using Agglomerative Clustering on the TAE latent space.

    This runs a forward pass through the frozen TAE encoder to extract latent
    features, then applies complete-linkage hierarchical clustering to set
    initial centroid positions.

    Args:
        X (np.ndarray): Scaled input data of shape (N, 8, T).

    Returns:
        z (Tensor):          Latent representations, shape (N, n_hidden).
        z_np (np.ndarray):   Same as z on CPU as numpy array.
        centroids_ (Tensor): Initialized centroid positions, shape (n_clusters, n_hidden).
    """
    X_tensor = torch.from_numpy(X).type(torch.FloatTensor).to(args.device)
    z, z_np, centroids_ = model.init_centroids(X_tensor)
    return z, z_np, centroids_


# ══════════════════════════════════════════════════════════
# SECTION 4: KL Divergence Loss
# ══════════════════════════════════════════════════════════

def kl_loss_function(input: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence between the target distribution P and soft assignment Q.

    KL(P || Q) = sum_j [ P_ij * log(P_ij / Q_ij) ]

    This encourages the soft cluster assignments Q to align with the sharpened
    target distribution P, refining cluster boundaries over training.

    Args:
        input (Tensor): Target distribution P, shape (batch_size, n_clusters).
        pred  (Tensor): Soft assignment Q,     shape (batch_size, n_clusters).

    Returns:
        Tensor: Scalar mean KL divergence loss.
    """
    out = input * torch.log(input / pred)
    return torch.mean(torch.sum(out, dim=1))


# ══════════════════════════════════════════════════════════
# SECTION 5: ClusterNet Training (One Epoch)
# ══════════════════════════════════════════════════════════

def train_ClusterNET(epoch: int, args, verbose: bool) -> tuple:
    """
    Train the full ClusterNet model for one epoch.

    The total loss combines:
      - MSE reconstruction loss (masked): measures data reconstruction quality
      - KL divergence clustering loss (masked): refines cluster assignments

    Both losses are masked to exclude missing gait session data.

    Args:
        epoch   (int):  Current epoch index (0-based).
        args:           Parsed argument namespace.
        verbose (bool): If True, prints per-epoch loss.

    Returns:
        train_loss (float):       Mean total loss for the epoch.
        all_gt (np.ndarray):      Ground-truth sample labels.
        all_preds (np.ndarray):   Predicted cluster assignments.
        preds (Tensor):           Raw prediction tensor (last batch).
        all_z (np.ndarray):       All latent feature vectors.
        kl_loss_mean (float):     Mean KL divergence loss.
        mse_loss_mean (float):    Mean MSE reconstruction loss.
    """
    model.train()
    train_loss   = 0
    kl_loss_sum  = 0
    mse_loss_sum = 0
    all_preds, all_gt, all_z = [], [], []

    for batch_idx, (inputs, labels, mask) in enumerate(trainloader):
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        labels = labels.to(args.device)
        mask   = mask.to(args.device)

        optimizer_clu.zero_grad()

        # Forward pass through ClusterNet → latent z, reconstruction x̂, Q, P
        z, x_reconstr, Q, P = model(inputs)

        # ── MSE loss (masked) ─────────────────────────────
        loss_mse = loss1(inputs.squeeze(1), x_reconstr)
        loss_mse = torch.sum(loss_mse * mask) / torch.sum(mask)

        # ── KL divergence loss (masked) ───────────────────
        loss_KL = kl_loss_function(P, Q)
        loss_KL = torch.sum(loss_KL * mask) / torch.sum(mask)

        # ── Total loss & backprop ─────────────────────────
        total_loss = loss_mse + loss_KL
        total_loss.backward()
        optimizer_clu.step()

        # ── Collect predictions ───────────────────────────
        preds = torch.max(Q, dim=1)[1]   # Hard cluster assignment (argmax of Q)
        all_preds.append(preds.cpu().detach())
        all_z.append(z.cpu().detach())
        all_gt.append(labels.cpu().detach())

        train_loss   += total_loss.item()
        kl_loss_sum  += loss_KL.item()
        mse_loss_sum += loss_mse.item()

        print(f"  [Batch {batch_idx}] MSE: {loss_mse.item():.5f} | KL: {loss_KL.item():.5f}")

    n_batches = batch_idx + 1
    if verbose:
        print(f"  [Train] Epoch {epoch + 1} | Total Loss: {train_loss / n_batches:.5f}")

    # Concatenate all batch outputs
    all_gt    = torch.cat(all_gt,    dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_z     = torch.cat(all_z,     dim=0).numpy()

    return (
        train_loss   / n_batches,
        all_gt,
        all_preds,
        preds,
        all_z,
        kl_loss_sum  / n_batches,
        mse_loss_sum / n_batches,
    )


# ══════════════════════════════════════════════════════════
# SECTION 6: Full Training Loop
# ══════════════════════════════════════════════════════════

def training_function(args, max_epochs: int, verbose: bool = True) -> tuple:
    """
    Run the full ClusterNet training loop for `max_epochs` epochs.

    Initializes cluster centroids from the pretrained TAE latent space before
    training begins. After training, saves the full model weights.

    Args:
        args:         Parsed argument namespace.
        max_epochs:   Total number of clustering-phase epochs.
        verbose:      If True, prints per-epoch training loss.

    Returns:
        all_gt (np.ndarray):          Ground-truth labels for all samples.
        all_preds (np.ndarray):       Final cluster assignments for all samples.
        all_train_loss (list[float]): Total loss per epoch.
        preds (Tensor):               Final batch prediction tensor.
        all_z (np.ndarray):           Final latent features for all samples.
        kl_losses (list[float]):      KL divergence loss per epoch.
        mse_losses (list[float]):     MSE reconstruction loss per epoch.
    """
    # Initialize centroids before clustering training begins
    z, z_np, centroids_ = initalize_centroids(X_scaled)

    all_train_loss = []
    kl_losses      = []
    mse_losses     = []

    print("Training full ClusterNet model...")

    for epoch in range(max_epochs):
        train_loss, all_gt, all_preds, preds, all_z, kl_loss, mse_loss = \
            train_ClusterNET(epoch, args, verbose=verbose)

        all_train_loss.append(train_loss)
        kl_losses.append(kl_loss)
        mse_losses.append(mse_loss)

    # Save full model weights
    torch.save(model.state_dict(), args.path_weights_main)

    return all_gt, all_preds, all_train_loss, preds, all_z, kl_losses, mse_losses


# ══════════════════════════════════════════════════════════
# SECTION 7: Output Utilities
# ══════════════════════════════════════════════════════════

def output_xlsx(df: pd.DataFrame, filename: str, move_dir: str) -> None:
    """
    Save a DataFrame as a timestamped Excel (.xlsx) file.

    Args:
        df        (pd.DataFrame): Data to save.
        filename  (str):          Base filename (timestamp appended automatically).
        move_dir  (str):          Target directory. Created if it does not exist.
    """
    if not os.path.exists(move_dir):
        os.makedirs(move_dir)

    todaydate = time.strftime("%y%m%d%H%M")
    name      = f"{filename}_{todaydate}.xlsx"

    writer = ExcelWriter(os.path.join(move_dir, name))
    df.to_excel(writer, 'Sheet1', index=False)
    writer.close()
    print(f"  [Saved] {os.path.join(move_dir, name)}")


def output_lossgraph(n_clusters: int, fold: int) -> None:
    """
    Save the current matplotlib figure as a PNG loss graph.

    Args:
        n_clusters (int): Number of clusters (used in filename).
        fold       (int): Cross-validation fold index (used in filename).
    """
    mother_path = "./figures/"
    todaydate   = time.strftime("%y%m%d")
    todaytime   = time.strftime("%H%M")
    img_path    = os.path.join(mother_path, todaydate)

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    plt.savefig(os.path.join(img_path, f"{todaytime}_cluster_{n_clusters}_fold_{fold + 1}.png"))


# ══════════════════════════════════════════════════════════
# SECTION 8: Main Execution Block
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 8.1  Random Seed ──────────────────────────────────
    # To change the seed, modify SEED below (see README for details)
    SEED = 42
    fix_seed(SEED)

    # ── 8.2  Argument Parsing & Path Setup ───────────────
    parser = get_arguments()
    args   = parser.parse_args()

    # Resolve data directory path (e.g., data/GaitCycleLong/)
    args.path_data = args.path_data.format(args.dataset_name)
    if not os.path.exists(args.path_data):
        os.makedirs(args.path_data)

    # Resolve model weights directory (e.g., models_weights/GaitCycleLong/)
    path_weights = args.path_weights.format(args.dataset_name)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)

    # Define specific weight file paths
    args.path_weights_ae   = os.path.join(path_weights, "autoencoder_weight_n.pth")
    args.path_weights_main = os.path.join(path_weights, "full_model_weigths_n.pth")

    # ── 8.3  Device Setup ─────────────────────────────────
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # CUDA event timers for measuring training duration
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # ── 8.4  Result DataFrames ────────────────────────────
    # Per-cluster silhouette results
    df_result = pd.DataFrame(
        columns=['n_cluster', 'cluster', '# of members', 'sil_per_cluster', 'sil_global', 'STD']
    )
    # Per-trial hyperparameter results
    df_hyper = pd.DataFrame(
        columns=['n_cluster', 'lr_ae', 'lr_cluster', 'pooling', 'ep_cluster', 'empty', 'batch', 'SC', 'STD', 'time']
    )
    # Best hyperparameter configuration per cluster count
    df_max = pd.DataFrame(
        columns=['n_cluster', 'empty', 're_cluster', 'lr_ae', 'lr_cluster', 'pooling', 'ep_cluster', 'batch', 'SC', 'STD', 'time']
    )

    # ── 8.5  Cluster Range ────────────────────────────────
    # Modify init_cluster and end_cluster to change the search range
    init_cluster = 3    # Minimum number of clusters to evaluate
    end_cluster  = 10   # Maximum number of clusters to evaluate
    criterion    = 0    # Reserved for future stopping criteria

    time_lst = []
    sil_lst  = []
    all_time = 0

    # ── 8.6  Output Directory Setup ───────────────────────
    mother_path = "./figures/"
    todaydate   = time.strftime("%y%m%d")
    todaytime   = time.strftime("%H%M")
    img_path    = os.path.join(mother_path, todaydate) + "/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # ── 8.7  Hyperparameter Grid ──────────────────────────
    # Add multiple values per key to enable grid search.
    # Best values (from paper) are set as defaults.
    hyper_dict = {
        'lr_ae':          [1e-4],   # TAE pretraining learning rate
        'lr_cluster':     [1e-8],   # Clustering phase learning rate
        'max_epochs_ae':  [100],    # TAE pretraining epochs
        'max_epochs_clu': [200],    # Clustering phase epochs
        'pooling':        [25],     # MaxPool1D pooling factor
        'batch':          [8],      # Mini-batch size
    }

    # Generate all combinations of hyperparameters
    allNames      = sorted(hyper_dict)
    combinations  = it.product(*(hyper_dict[Name] for Name in allNames))
    hyper_params  = list(combinations)

    ccnt  = 0  # Row counter for df_max
    trial = 0  # Row counter for df_hyper

    # ── 8.8  Hyperparameter Loop ──────────────────────────
    for (batch, lr_ae, lr_cluster, max_epochs_ae, max_epochs_clu, pool) in hyper_params:

        fix_seed(SEED)  # Re-fix seed for each hyperparameter combination
        cnt = 0         # Row counter for df_result

        # ── 8.8a  Cluster Count Loop ──────────────────────
        for n_clusters in np.arange(init_cluster, end_cluster + 1):
            try:
                print("=" * 40)
                print(f"n_clusters = {n_clusters}")

                means_lst = []  # Per-cluster silhouette score list

                start.record()

                # ── Load data & build DataLoader ──────────
                trainset, X_scaled, data_y = get_long_loader(args)
                args.serie_size = X_scaled.shape[2]   # Time-series length (T)
                trainloader = DataLoader(
                    trainset, batch_size=batch, shuffle=True, num_workers=0
                )

                # ── Pretrain TAE ──────────────────────────
                all_pretrain_loss, inputs = pretrain_autoencoder(
                    args, lr_ae, max_epochs_ae, pool
                )

                # ── Build ClusterNet (loads pretrained TAE weights) ──
                model = ClusterNet(args, pool, n_clusters)
                model = model.to(args.device)

                # Loss & optimizer for the clustering phase
                loss1         = nn.MSELoss(reduction='none')
                optimizer_clu = torch.optim.SGD(
                    model.parameters(), lr=lr_cluster, momentum=args.momentum
                )

                # ── Train ClusterNet ──────────────────────
                all_gt, all_preds, all_train_loss, preds, all_z, kl_losses, mse_losses = \
                    training_function(args, max_epochs_clu)

                # ── Evaluate Clustering Quality ───────────
                if len(set(all_preds)) == 1:
                    # Degenerate case: all samples assigned to one cluster
                    sil_per_clu = -1
                    for label in range(n_clusters):
                        means_lst.append(sil_per_clu)
                    silhouette = -1
                    calinski   = -1
                else:
                    # Compute per-cluster and global silhouette scores
                    sil_per_clu = silhouette_samples(all_z, all_preds)
                    for label in range(n_clusters):
                        means_lst.append(
                            round(sil_per_clu[all_preds == label].mean(), 4)
                        )
                    silhouette = round(silhouette_score(all_z, all_preds), 4)
                    calinski   = round(calinski_harabasz_score(all_z, all_preds), 4)

                sil_lst.append(silhouette)
                end.record()
                torch.cuda.synchronize()

                # ── Cluster Membership Analysis ───────────
                # Count members per cluster
                all_preds_eval = pd.DataFrame({
                    'cluster': pd.Categorical(all_preds, categories=range(0, n_clusters))
                })
                df_eval      = all_preds_eval['cluster'].value_counts(sort=False)
                df_eval_mean = df_eval.mean()
                df_ss        = pd.DataFrame((df_eval - df_eval_mean) ** 2)

                # ── Timing ────────────────────────────────
                mili_time = start.elapsed_time(end)   # elapsed_time returns milliseconds
                op_time   = 1e-3 * mili_time          # Convert to seconds
                time_lst.append(op_time)
                all_time += op_time

                # ── Store per-cluster results ─────────────
                for c in range(n_clusters):
                    dev = round(math.sqrt(df_ss.sum() / X_scaled.shape[0]), 4)
                    df_result.loc[cnt] = [n_clusters, c, df_eval[c], means_lst[c], silhouette, dev]
                    cnt += 1

                # ── Map sample labels to patient names ────
                all_gt_data = pd.DataFrame(all_gt, columns=['label'])
                all_gt_name = pd.DataFrame(columns=['name'])

                for i in range(X_scaled.shape[0]):
                    for j in range(X_scaled.shape[0]):
                        if all_gt_data['label'][i] == data_y['label'][j]:
                            all_gt_name.loc[i] = data_y['name'][j]

                all_gt_data = pd.concat([all_gt_data, all_gt_name, all_preds_eval], axis=1)

                # Build {cluster_id → [patient_names]} dictionary
                name_dict = {}
                name_lst  = []
                for l in range(n_clusters):
                    for k in range(X_scaled.shape[0]):
                        if all_gt_data['cluster'][k] == l:
                            name_lst.append(all_gt_data['name'][k])
                    name_dict[l] = name_lst
                    name_lst = []

                # Count empty clusters (no assigned patients)
                empty_count = sum(
                    1 for v in name_dict.values()
                    if v is None or (isinstance(v, list) and len(v) == 0)
                )

                # ── Log hyperparameter trial result ───────
                df_hyper.loc[trial] = [
                    n_clusters, lr_ae, lr_cluster, pool,
                    max_epochs_clu, empty_count, batch,
                    silhouette, dev, op_time
                ]
                trial += 1

                # ── Plot Training Loss Curves ──────────────
                plt.figure()
                plt.plot(
                    all_pretrain_loss + mse_losses,
                    label='Autoencoder Loss (MSE)'
                )
                plt.plot(
                    range(len(all_pretrain_loss), len(all_pretrain_loss) + len(kl_losses)),
                    kl_losses,
                    label='Clustering Loss (KL Divergence)'
                )
                plt.plot(
                    range(len(all_pretrain_loss), len(all_pretrain_loss) + len(all_train_loss)),
                    all_train_loss,
                    label='Total Loss'
                )
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.ylim(0, 1)
                plt.legend()
                plt.title(
                    f"Loss Curves | n_cluster={n_clusters} | "
                    f"lr_ae={lr_ae} | lr_clu={lr_cluster} | pool={pool}"
                )

                # Add interactive cursor (hover tooltip)
                cursor = mplcursors.cursor(hover=True)
                cursor.connect(
                    "add",
                    lambda sel: sel.annotation.set_text(
                        f"Epoch: {sel.index}, Loss: {kl_losses[sel.index]:.4f}"
                    )
                )

                # Save figure as high-resolution PNG
                fig_name = (
                    f"{todaytime}_kl_divergence_loss_"
                    f"{n_clusters}_lr{lr_ae}_{lr_cluster}_pool_{pool}_b_{batch}.png"
                )
                plt.savefig(img_path + fig_name, dpi=300)
                plt.show()

                print("-" * 40)
                print(f"Silhouette Score | n_clusters={n_clusters}: {silhouette}")
                print("=" * 40)

                # ── Save cluster assignment pickle ─────────
                root_pkl = './data/pickle/'
                pkl_path = os.path.join(root_pkl, todaydate) + '/'
                if not os.path.exists(pkl_path):
                    os.makedirs(pkl_path)

                pkl_name = (
                    f"{todaytime}_cluster_{n_clusters}_"
                    f"lr{lr_ae}_{lr_cluster}_pool_{pool}_b_{batch}.pkl"
                )
                with open(pkl_path + pkl_name, 'wb') as handle:
                    pickle.dump(name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            except ValueError as e:
                # Skip invalid cluster configurations (e.g., too few unique samples)
                print(f"  [Warning] Skipped n_clusters={n_clusters}: {e}")
                pass

        # ── Per-hyperparameter Summary ────────────────────
        print(f"Overall operation time: {all_time:.2f} sec")
        print("=" * 40)

        # Save per-cluster silhouette results to Excel
        output_xlsx(
            df=df_result,
            filename=f"result_lr{lr_ae}_{lr_cluster}_pool{pool}_ep{max_epochs_clu}_b{batch}",
            move_dir='./results/',
        )

        # Plot global silhouette score vs. number of clusters
        plt.figure()
        plt.plot(df_result['n_cluster'], df_result['sil_global'], 'bo-', label='Silhouette Index')
        plt.xticks(range(2, 11))
        plt.ylim([-0.1, 1.0])
        plt.title(
            f"Grid Tuning n_cluster SC | pool={pool} | "
            f"lr_ae={lr_ae} | lr_clu={lr_cluster} | batch={batch}"
        )
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.legend()
        plt.show()

    # ── 8.9  Final Summary & Best Config Selection ────────
    # Compute effective cluster count (excluding empty clusters)
    df_hyper['re_cluster'] = df_hyper['n_cluster'] - df_hyper['empty']
    df_hyper = df_hyper.sort_values(by=['re_cluster'], ascending=True)
    df_hyper.reset_index(drop=True, inplace=True)
    df_hyper = df_hyper[[
        'n_cluster', 'empty', 're_cluster', 'lr_ae', 'lr_cluster',
        'pooling', 'ep_cluster', 'batch', 'SC', 'STD', 'time'
    ]]

    # For each effective cluster count, select the config with the best silhouette score
    # Priority: 0 empty clusters > 1 empty > 2 empty
    thresh = 2
    for ccc in np.arange(init_cluster, end_cluster + 1):
        try:
            df_ccc = df_hyper[df_hyper['re_cluster'] == ccc]
            df_ddd = df_ccc[df_ccc['empty'] <= thresh]
            df_ddd.reset_index(drop=True, inplace=True)

            zero = (df_ddd['empty'] == 0).sum()
            one  = (df_ddd['empty'] == 1).sum()
            two  = (df_ddd['empty'] == 2).sum()

            if zero > 0:
                df_mmm = df_ddd.loc[df_ddd[df_ddd['empty'] == 0]['SC'].idxmax()]
            elif one > 0:
                df_mmm = df_ddd.loc[df_ddd[df_ddd['empty'] == 1]['SC'].idxmax()]
            elif two > 0:
                df_mmm = df_ddd.loc[df_ddd[df_ddd['empty'] == 2]['SC'].idxmax()]

            df_max.loc[ccnt] = df_mmm
            df_mmm = pd.DataFrame()

        except Exception:
            pass
        ccnt += 1

    # Save final summary tables to Excel
    output_xlsx(df=df_hyper, filename='result_total',     move_dir='./results_tot/')
    output_xlsx(df=df_max,   filename='result_total_max', move_dir='./results_tot/')
