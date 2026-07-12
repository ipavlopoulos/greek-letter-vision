#!/usr/bin/env python3
"""
Shared utilities for graphic-compensation analyses.
This file is safe for an anonymized repository: it uses only relative paths.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

VOWELS = {"Alpha", "Epsilon", "Eta", "Iota", "Omicron", "Upsilon", "Omega"}

def phonetic_group(letter: str) -> str:
    return "vowel" if str(letter) in VOWELS else "consonant"

def pair_category(a: str, b: str) -> str:
    return "same-category" if phonetic_group(a) == phonetic_group(b) else "vowel-consonant"

def make_pairwise_scores(cm_df: pd.DataFrame) -> pd.DataFrame:
    classes = list(map(str, cm_df.index))
    cm_df = cm_df.copy()
    cm_df.index = classes
    cm_df.columns = list(map(str, cm_df.columns))
    if classes != list(cm_df.columns):
        raise ValueError("Confusion-matrix index and columns do not match.")
    cm = cm_df.values
    rows = []
    for i, a in enumerate(classes):
        for j in range(i + 1, len(classes)):
            b = classes[j]
            count_ab = int(cm[i, j])
            count_ba = int(cm[j, i])
            total = count_ab + count_ba
            support_pair = int(cm[i, :].sum() + cm[j, :].sum())
            rows.append({
                "pair": f"{a}--{b}",
                "letter_1": a,
                "letter_2": b,
                "group_1": phonetic_group(a),
                "group_2": phonetic_group(b),
                "category": pair_category(a, b),
                "count_1_to_2": count_ab,
                "count_2_to_1": count_ba,
                "total_confusions": total,
                "pair_support": support_pair,
                "normalized_confusion": total / support_pair if support_pair else 0.0,
            })
    return pd.DataFrame(rows).sort_values(
        ["normalized_confusion", "total_confusions"],
        ascending=False
    ).reset_index(drop=True)

def permutation_test_mean_difference(pair_df: pd.DataFrame, n_perm: int = 100000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    same = pair_df[pair_df["category"] == "same-category"]["normalized_confusion"].to_numpy(float)
    vc = pair_df[pair_df["category"] == "vowel-consonant"]["normalized_confusion"].to_numpy(float)
    observed = vc.mean() - same.mean()
    pooled = np.concatenate([same, vc])
    n_same = len(same)
    diffs = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(pooled)
        diffs[k] = perm[n_same:].mean() - perm[:n_same].mean()
    return {
        "measure": "Normalized confusion",
        "same_category_mean": float(same.mean()),
        "vowel_consonant_mean": float(vc.mean()),
        "difference_vc_minus_same": float(observed),
        "p_one_sided": float((np.sum(diffs >= observed) + 1) / (n_perm + 1)),
        "p_two_sided": float((np.sum(np.abs(diffs) >= abs(observed)) + 1) / (n_perm + 1)),
    }

def exposure_weighted_test(pair_df: pd.DataFrame, n_perm: int = 100000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    errors = pair_df["total_confusions"].to_numpy(float)
    exposure = pair_df["pair_support"].to_numpy(float)
    cats = pair_df["category"].to_numpy()
    same_mask = cats == "same-category"
    vc_mask = cats == "vowel-consonant"
    def rate(mask):
        return errors[mask].sum() / exposure[mask].sum()
    obs = rate(vc_mask) - rate(same_mask)
    n_same = same_mask.sum()
    n = len(pair_df)
    diffs = np.empty(n_perm)
    for k in range(n_perm):
        perm_idx = rng.permutation(n)
        perm_same = perm_idx[:n_same]
        perm_vc = perm_idx[n_same:]
        diffs[k] = errors[perm_vc].sum() / exposure[perm_vc].sum() - errors[perm_same].sum() / exposure[perm_same].sum()
    same_rate = rate(same_mask)
    vc_rate = rate(vc_mask)
    return {
        "measure": "Exposure-weighted confusion rate",
        "same_errors": int(errors[same_mask].sum()),
        "same_exposure": int(exposure[same_mask].sum()),
        "same_rate": float(same_rate),
        "vc_errors": int(errors[vc_mask].sum()),
        "vc_exposure": int(exposure[vc_mask].sum()),
        "vc_rate": float(vc_rate),
        "difference_vc_minus_same": float(obs),
        "rate_ratio_vc_over_same": float(vc_rate / same_rate) if same_rate else None,
        "p_one_sided": float((np.sum(diffs >= obs) + 1) / (n_perm + 1)),
        "p_two_sided": float((np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)),
    }

def load_summary(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        return json.load(f)

def resolve_checkpoint(summary_path: str | Path, explicit_checkpoint: str | Path | None = None) -> Path:
    if explicit_checkpoint:
        return Path(explicit_checkpoint)
    summary = load_summary(summary_path)
    cp = Path(str(summary.get("checkpoint", "")))
    if cp.exists():
        return cp
    # anonymized-repo fallback: same folder as summary
    for candidate in [
        Path(summary_path).parent / "best_convnextv2_tiny_lf_dscl.pth",
        Path(summary_path).parent / "best_timm_model.pth",
    ]:
        if candidate.exists():
            return candidate
    return cp
