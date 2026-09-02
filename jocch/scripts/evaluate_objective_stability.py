#!/usr/bin/env python3
"""
Evaluate CE, LF+CE and LF+DSCL ConvNeXt-V2 runs and reproduce the objective-stability table.
Assumes the three checkpoints/summaries have already been trained.
"""
from __future__ import annotations
import argparse, json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from scripts.train_timm_backbone import TimmClassifier, build_loaders
from scripts.graphic_compensation_utils import make_pairwise_scores, permutation_test_mean_difference

def evaluate_run(run_name, cfg, data_dir, output_root, seed=42, image_size=64, n_perm=100000):
    with open(cfg["summary"]) as f:
        summary = json.load(f)
    args = Namespace(
        data_dir=Path(data_dir), batch_size=64, num_workers=2, seed=seed,
        test_size=0.2, val_size=0.1, image_size=image_size, otsu=False,
        no_standard_augmentation=False, use_lf=cfg["use_lf"], use_rectangular_erasure=False,
        use_rect_lacuna=False, lacuna_min=1, lacuna_max=4
    )
    _, _, test_loader, label_encoder = build_loaders(args)
    classes = label_encoder.classes_.tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier("convnextv2_tiny", num_classes=len(classes), pretrained=False, in_chans=1, image_size=image_size).to(device)
    state = torch.load(cfg["checkpoint"], map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state); model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
            all_pred.extend(pred.tolist()); all_true.extend(y.numpy().tolist())
    true_labels = label_encoder.inverse_transform(np.array(all_true))
    pred_labels = label_encoder.inverse_transform(np.array(all_pred))
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    cm_df = pd.DataFrame(confusion_matrix(true_labels, pred_labels, labels=classes), index=classes, columns=classes)

    run_out = output_root / run_name.replace("+", "_plus_")
    run_out.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(run_out / "test_confusion_matrix_counts.csv")
    pair_df = make_pairwise_scores(cm_df)
    pair_df.to_csv(run_out / "all_unordered_pairs.csv", index=False)
    pair_df[pair_df["total_confusions"] > 0].to_csv(run_out / "top_unordered_pairs_nonzero.csv", index=False)
    pd.DataFrame(classification_report(true_labels, pred_labels, labels=classes, target_names=classes, output_dict=True, zero_division=0)).T.to_csv(run_out / "classification_report.csv")

    global_stats = permutation_test_mean_difference(pair_df, n_perm=n_perm, seed=seed)
    global_row = {"Run": run_name, "Accuracy": acc, "Macro-F1": macro_f1, "Weighted-F1": weighted_f1, **global_stats}
    return pair_df, global_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="results/graphic_compensation/objective_stability_CE_LFCE_LFDSCL_64")
    ap.add_argument("--ce-summary", default="models/convnextv2_tiny_CE_A100/timm_summary.json")
    ap.add_argument("--ce-checkpoint", default="models/convnextv2_tiny_CE_A100/best_timm_model.pth")
    ap.add_argument("--lfce-summary", default="models/convnextv2_tiny_LF_CE_A100/timm_summary.json")
    ap.add_argument("--lfce-checkpoint", default="models/convnextv2_tiny_LF_CE_A100/best_timm_model.pth")
    ap.add_argument("--lfdscl-summary", default="models/convnextv2_lf_dscl/convnextv2_tiny_lf_dscl_summary.json")
    ap.add_argument("--lfdscl-checkpoint", default="models/convnextv2_lf_dscl/best_convnextv2_tiny_lf_dscl.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--n-perm", type=int, default=100000)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    runs = {
        "CE": {"summary": Path(args.ce_summary), "checkpoint": Path(args.ce_checkpoint), "use_lf": False},
        "LF+CE": {"summary": Path(args.lfce_summary), "checkpoint": Path(args.lfce_checkpoint), "use_lf": True},
        "LF+DSCL": {"summary": Path(args.lfdscl_summary), "checkpoint": Path(args.lfdscl_checkpoint), "use_lf": True},
    }

    pair_dfs, global_rows = {}, []
    for run_name, cfg in runs.items():
        pair_df, row = evaluate_run(run_name, cfg, args.data_dir, out, seed=args.seed, image_size=args.image_size, n_perm=args.n_perm)
        pair_dfs[run_name] = pair_df
        global_rows.append(row)

    pd.DataFrame(global_rows).to_csv(out / "global_pairwise_tests_CE_LFCE_LFDSCL.csv", index=False)

    top_tables = {run: df[df["total_confusions"] > 0].head(10).assign(rank=range(1, min(10, len(df[df["total_confusions"] > 0])) + 1)) for run, df in pair_dfs.items()}
    all_top_pairs = sorted(set().union(*[set(t["pair"].tolist()) for t in top_tables.values()]))
    rows = []
    for pair in all_top_pairs:
        category = None
        for df in pair_dfs.values():
            r = df[df["pair"] == pair]
            if len(r):
                category = r.iloc[0]["category"]; break
        row = {"Pair": pair, "Category": category}
        present = 0
        for run in ["CE", "LF+CE", "LF+DSCL"]:
            top = top_tables[run]; full = pair_dfs[run]
            in_top = pair in set(top["pair"])
            if in_top:
                present += 1
                row[f"{run} rank"] = int(top[top["pair"] == pair].iloc[0]["rank"])
            else:
                row[f"{run} rank"] = ""
            r_full = full[full["pair"] == pair].iloc[0]
            row[f"{run} conf."] = int(r_full["total_confusions"])
            row[f"{run} norm."] = float(r_full["normalized_confusion"])
        row["Top-10 recurrence"] = f"{present}/3"
        rows.append(row)
    stability = pd.DataFrame(rows)
    stability["_rec_n"] = stability["Top-10 recurrence"].str[0].astype(int)
    stability["_is_vc"] = (stability["Category"] == "vowel-consonant").astype(int)
    stability["_lfdscl_norm"] = stability["LF+DSCL norm."]
    stability = stability.sort_values(["_rec_n", "_is_vc", "_lfdscl_norm"], ascending=False).drop(columns=["_rec_n","_is_vc","_lfdscl_norm"])
    stability.to_csv(out / "top10_pair_recurrence_CE_LFCE_LFDSCL.csv", index=False)
    stability.to_latex(out / "top10_pair_recurrence_CE_LFCE_LFDSCL.tex", index=False, float_format="%.3f")

    compact = stability.copy()
    for run in ["CE", "LF+CE", "LF+DSCL"]:
        compact[run] = compact[f"{run} rank"].apply(lambda x: "✓" if x != "" else "")
    compact = compact[["Pair","Category","CE","LF+CE","LF+DSCL","Top-10 recurrence","CE norm.","LF+CE norm.","LF+DSCL norm."]]
    compact.to_csv(out / "compact_top10_pair_recurrence_CE_LFCE_LFDSCL.csv", index=False)
    compact.to_latex(out / "compact_top10_pair_recurrence_CE_LFCE_LFDSCL.tex", index=False, float_format="%.3f", escape=False)
    print("Saved outputs to", out)

if __name__ == "__main__":
    main()
