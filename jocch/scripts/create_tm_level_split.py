#!/usr/bin/env python3
"""
Create the TM-level train/validation/test split used as leakage-safe control.
This script uses only data/hellchar/hellchar.csv and data/hellchar/cliplets.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ALLOWED_LETTERS = [
    "Alpha", "Beta", "Chi", "Delta", "Epsilon", "Eta", "Gamma",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Omega", "Omicron",
    "Phi", "Pi", "Psi", "Rho", "Sigma", "Tau", "Theta",
    "Upsilon", "Xi", "Zeta"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hellchar")
    ap.add_argument("--output-dir", default="data/splits")
    ap.add_argument("--max-seed", type=int, default=5000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "hellchar.csv"
    cliplet_dir = data_dir / "cliplets"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["letter"].isin(ALLOWED_LETTERS)].copy()

    def resolve_cliplet_path(filename):
        p = cliplet_dir / str(filename)
        if p.exists():
            return str(p)
        matches = list(cliplet_dir.rglob(str(filename)))
        return str(matches[0]) if matches else None

    df["path"] = df["filename"].apply(resolve_cliplet_path)
    if df["path"].isna().any():
        raise FileNotFoundError("Some cliplet paths are missing.")
    df["TM"] = df["TM"].astype(str)

    all_letters = sorted(df["letter"].unique())

    def split_score(train_df, val_df, test_df):
        train_missing = len(set(all_letters) - set(train_df["letter"]))
        val_missing = len(set(all_letters) - set(val_df["letter"]))
        test_missing = len(set(all_letters) - set(test_df["letter"]))
        missing_penalty = 100000 * (train_missing + val_missing + test_missing)
        n = len(train_df) + len(val_df) + len(test_df)
        size_penalty = abs(len(train_df) - 0.70*n) + abs(len(val_df) - 0.10*n) + abs(len(test_df) - 0.20*n)
        full_dist = df["letter"].value_counts(normalize=True).reindex(all_letters).fillna(0)
        test_dist = test_df["letter"].value_counts(normalize=True).reindex(all_letters).fillna(0)
        val_dist = val_df["letter"].value_counts(normalize=True).reindex(all_letters).fillna(0)
        dist_penalty = 5000 * (np.abs(test_dist - full_dist).sum() + 0.5 * np.abs(val_dist - full_dist).sum())
        return missing_penalty + size_penalty + dist_penalty

    best, best_info = None, None
    for seed in range(args.max_seed):
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        trainval_idx, test_idx = next(gss_test.split(df, groups=df["TM"]))
        trainval_df = df.iloc[trainval_idx].copy()
        test_df = df.iloc[test_idx].copy()

        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 10000)
        train_idx_rel, val_idx_rel = next(gss_val.split(trainval_df, groups=trainval_df["TM"]))
        train_df = trainval_df.iloc[train_idx_rel].copy()
        val_df = trainval_df.iloc[val_idx_rel].copy()

        train_tms, val_tms, test_tms = set(train_df["TM"]), set(val_df["TM"]), set(test_df["TM"])
        if train_tms & val_tms or train_tms & test_tms or val_tms & test_tms:
            continue
        score = split_score(train_df, val_df, test_df)
        if best is None or score < best:
            best = score
            best_info = (seed, train_df, val_df, test_df, score)

    if best_info is None:
        raise RuntimeError("Could not create a valid TM-level split.")

    seed, train_df, val_df, test_df, score = best_info
    train_df["split"] = "train"; val_df["split"] = "val"; test_df["split"] = "test"
    tm_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    summary_counts = pd.DataFrame({
        "train": train_df["letter"].value_counts().sort_index(),
        "val": val_df["letter"].value_counts().sort_index(),
        "test": test_df["letter"].value_counts().sort_index(),
    }).fillna(0).astype(int)

    split_csv = out / "hellchar_tm_level_split_seed.csv"
    counts_csv = out / "hellchar_tm_level_split_class_counts.csv"
    tm_split_df.to_csv(split_csv, index=False)
    summary_counts.to_csv(counts_csv)

    print("Best seed:", seed)
    print("Best score:", score)
    print("Train examples:", len(train_df), "| TMs:", train_df["TM"].nunique())
    print("Val examples:", len(val_df), "| TMs:", val_df["TM"].nunique())
    print("Test examples:", len(test_df), "| TMs:", test_df["TM"].nunique())
    print("Saved:", split_csv)
    print("Saved:", counts_csv)

if __name__ == "__main__":
    main()
