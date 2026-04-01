#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_long_custom_data.py
========================
Custom dataset loader for longitudinal gait kinematic data (DTCRP model).

Loads joint angle and angular velocity data from two Excel files, applies
z-score normalization, restructures into patient-wise 3D tensors, and
generates a binary mask to handle missing gait sessions.

Missing data handling strategy:
  - Missing values (NaN) are replaced with 999 as a sentinel value.
  - A binary mask (1=valid, 0=missing) is created to exclude these positions
    from loss computation during training.

Data format:
  - Each patient has up to 8 assessment sessions
    (weeks 2, 3, 4, 6, 8, 10, 12, 24 post-stroke).
  - Sessions are recorded as consecutive rows in the Excel sheet.
  - Columns represent time-series features (joint angles or velocities).

Reference:
  Teng et al., "Deep Temporal Clustering for Long-Term Gait Recovery Patterns
  of Post-Stroke Patients using Joint Kinematic Data", ICCAI 2025.

Author: howard
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


# ══════════════════════════════════════════════════════════
# 1. Custom Dataset Class
# ══════════════════════════════════════════════════════════

class CustomDataset(Dataset):
    """
    PyTorch Dataset for longitudinal gait kinematic time-series.

    Each sample contains:
      - A 3D tensor of shape (8 sessions, T timesteps, F features)
        representing one patient's complete longitudinal gait data.
      - An integer label (patient index, 0-based).
      - A binary mask of the same shape as the time-series, where
        1 indicates valid data and 0 indicates a missing session.

    Args:
        time_series (Tensor): Patient gait data, shape (N, 8, T, F) or (N, F, 8).
        labels      (array):  Integer label per patient, shape (N,).
        mask        (Tensor): Binary validity mask, same shape as time_series.
    """

    def __init__(self, time_series: torch.Tensor, labels: np.ndarray, mask: torch.Tensor):
        self.time_series = time_series
        self.labels      = labels
        self.mask        = mask

        print(f"[Dataset] Time series shape : {self.time_series.shape}")
        print(f"[Dataset] Mask shape        : {self.mask.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve one patient's data by index.

        Returns:
            time_series[idx] (Tensor): Patient gait tensor.
            labels[idx]      (int):    Patient label (integer).
            mask[idx]        (Tensor): Binary mask tensor.
        """
        return self.time_series[idx], self.labels[idx], self.mask[idx]


# ══════════════════════════════════════════════════════════
# 2. Data Loader Function
# ══════════════════════════════════════════════════════════

def get_long_loader(args) -> tuple:
    """
    Load, preprocess, and package longitudinal gait data into a CustomDataset.

    Processing pipeline:
      1. Load joint angle and angular velocity Excel files.
      2. Concatenate into a single feature matrix.
      3. Replace NaN (missing sessions) with sentinel value 999.
      4. Apply z-score normalization (mean=0, std=1) per feature.
      5. Reshape from (N_rows, F) to (N_patients, 8_sessions, T, F).
      6. Create binary mask: 1 where original value ≠ 0, else 0.
      7. Wrap in CustomDataset.

    ⚠️  Before running, set the correct file paths for:
        - `filename_angle` : path to the joint angle Excel file
        - `filename_vel`   : path to the joint angular velocity Excel file

    Args:
        args: Parsed argument namespace (not directly used here but kept
              for interface consistency with the rest of the pipeline).

    Returns:
        trainset  (CustomDataset): Dataset ready for DataLoader.
        X_re      (np.ndarray):    Scaled & reshaped data, shape (N, 8, T).
        data_y    (pd.DataFrame):  DataFrame with patient name and label columns.
    """

    # ── Step 1: Load Excel data ───────────────────────────────────
    # ⚠️  Update these paths to match your local data files
    filename_angle = " "   # TODO: e.g., "data/GaitCycleLong/joint_angles.xlsx"
    filename_vel   = " "   # TODO: e.g., "data/GaitCycleLong/joint_velocities.xlsx"

    df_angle = pd.read_excel(
        filename_angle, engine='openpyxl', sheet_name='Sheet1', header=0
    )
    # Preserve patient name column before dropping the unnamed index column
    df_angle_2 = df_angle.rename(columns={'Unnamed: 0': 'name'})
    df_angle   = df_angle.drop('Unnamed: 0', axis=1)

    df_vel = pd.read_excel(
        filename_vel, engine='openpyxl', sheet_name='Sheet1', header=0
    )
    df_vel = df_vel.drop('Unnamed: 0', axis=1)

    # Concatenate angle and velocity features along columns
    df_total = pd.concat([df_angle, df_vel], axis=1)

    # ── Step 2: Handle missing data ───────────────────────────────
    # Replace NaN with sentinel value 999 (will be masked out during training)
    df_total    = df_total.fillna(999)
    tensor_data = torch.tensor(df_total.values, dtype=torch.float)

    # ── Step 3: Z-score normalization ────────────────────────────
    # TimeSeriesScalerMeanVariance normalizes each feature to mean=0, std=1
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(tensor_data.cpu().numpy())

    # ── Step 4: Reshape to patient-level 3D structure ─────────────
    # Each patient has exactly 8 consecutive rows (one per session).
    # Grouping: rows 0–7 → patient 0, rows 8–15 → patient 1, etc.
    df_angle_re = pd.DataFrame()
    array_lst   = []

    n_patients = int(X_scaled.shape[0] / 8)

    for i in range(n_patients):
        # Extract 8 rows for patient i: shape (8, T, F)
        X_new  = X_scaled[8 * i : 8 * i + 8]
        # Transpose to (T, 8, F) for temporal-first convention, then append
        X_newt = X_new.transpose()
        array_lst.append(X_newt)

        # Record one representative row per patient (first session)
        df_angle_target = df_angle_2.iloc[8 * i]
        df_angle_re     = pd.concat([df_angle_re, df_angle_target], axis=1)

    # Clean up patient name DataFrame
    df_angle_re = df_angle_re.transpose().reset_index(drop=True)
    # Extract base patient ID (e.g., "PT001" from "PT001_week2")
    df_angle_re['name'] = df_angle_re['name'].str.split('_').str[0]

    # Concatenate all patients and permute to (N, 8_sessions, T)
    X_re = np.concatenate(array_lst, axis=0)
    X_re = np.transpose(X_re, (0, 2, 1))

    # ── Step 5: Generate integer labels ──────────────────────────
    shape = X_re.shape
    y     = np.arange(1, shape[0] + 1)
    y     = LabelEncoder().fit_transform(y)  # Ensures labels start from 0
    assert y.min() == 0, "Label encoding failed: minimum label is not 0."

    # ── Step 6: Create binary mask ────────────────────────────────
    # Mask = 1 where data is valid (original data ≠ 0 sentinel), else 0
    # Note: After scaling, the 999 sentinel is also scaled, so we check ≠ 0
    mask = np.where(X_re != 0, 1, 0).astype(float)
    mask = torch.tensor(mask, dtype=torch.float)

    # ── Step 7: Wrap in CustomDataset ────────────────────────────
    trainset = CustomDataset(
        time_series=torch.tensor(X_re, dtype=torch.float),
        labels=y,
        mask=mask,
    )

    # Build label DataFrame with patient names for downstream analysis
    data_y = pd.DataFrame(y, columns=['label'])
    data_y = pd.concat([df_angle_re['name'], data_y], axis=1)

    return trainset, X_re, data_y
