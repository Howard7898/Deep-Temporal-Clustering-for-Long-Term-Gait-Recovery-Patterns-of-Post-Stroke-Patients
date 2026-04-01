#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config1.py
==========
Argument parser for hyperparameters that are kept fixed following the
recommendations of the original DTC paper.

Parameters that require frequent tuning (learning rates, epochs, pooling,
batch size) are handled directly in `dtc_longitudinal.py` via `hyper_dict`.
This file manages the stable configuration values.

Reference:
  Teng et al., "Deep Temporal Clustering for Long-Term Gait Recovery Patterns
  of Post-Stroke Patients using Joint Kinematic Data", ICCAI 2025.
"""

import argparse


def get_arguments() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the DTCRP model.

    Returns:
        argparse.ArgumentParser: Parser with all argument definitions.

    Arguments:
        --dataset_name  (str):   Name of the dataset folder under `data/`.
                                 Default: "GaitCycleLong"
        --path_data     (str):   Template path to the data directory.
                                 Default: "data/{}"  (filled with dataset_name)
        --path_weights  (str):   Template path to the model weights directory.
                                 Default: "models_weights/{}/"
        --alpha         (int):   Degrees of freedom for the Student's t-distribution
                                 in the soft cluster assignment formula.
                                 Default: 1  (as recommended by DTC paper)
        --max_patience  (int):   Early stopping patience — number of epochs to wait
                                 without validation loss improvement before stopping.
                                 Default: 5
        --momentum      (float): SGD momentum coefficient for ClusterNet training.
                                 Default: 0.9
    """
    parser = argparse.ArgumentParser(
        description="DTCRP: Deep Temporal Clustering for Gait Recovery Patterns"
    )

    # ── Data paths ────────────────────────────────────────────────
    parser.add_argument(
        "--dataset_name",
        default="GaitCycleLong",
        type=str,
        help="Name of the dataset subfolder inside data/. Default: 'GaitCycleLong'.",
    )
    parser.add_argument(
        "--path_data",
        default="data/{}",
        type=str,
        help="Template path to the data directory. '{}' is replaced by dataset_name.",
    )
    parser.add_argument(
        "--path_weights",
        default="models_weights/{}/",
        type=str,
        help="Template path to the model weights directory. '{}' is replaced by dataset_name.",
    )

    # ── Clustering hyperparameter ─────────────────────────────────
    parser.add_argument(
        "--alpha",
        type=int,
        default=1,
        help=(
            "Degrees of freedom (alpha) for the Student's t-distribution used in "
            "soft cluster assignment. alpha=1 corresponds to a Cauchy distribution "
            "and is recommended by the original DTC paper. Default: 1."
        ),
    )

    # ── Training control ──────────────────────────────────────────
    parser.add_argument(
        "--max_patience",
        type=int,
        default=5,
        help=(
            "Early stopping patience: number of consecutive epochs without validation "
            "loss improvement before training is halted. Default: 5."
        ),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help=(
            "Momentum coefficient for the SGD optimizer used during ClusterNet "
            "fine-tuning. Default: 0.9 (standard SGD momentum)."
        ),
    )

    return parser
