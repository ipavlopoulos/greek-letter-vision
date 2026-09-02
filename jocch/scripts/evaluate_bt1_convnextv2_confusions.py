#!/usr/bin/env python3
"""
Evaluate the main ConvNeXt-V2 LF+DSCL checkpoint on the standard Hell-Char BT1 split.
Outputs reproduce the classification report, confusion matrix, pairwise scores,
and global pairwise tests used in the revised paper.
"""
from __future__ import annotations
import argparse, json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

from scripts.train_timm_backbone import TimmClassifier, build_loaders
from scripts.graphic_compensation_utils import (
    make_pairwise_scores, permutation_test_mean_difference, exposure_weighted_test,
    load_summary, resolve_checkpoint
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--summary", default="models/convnextv2_lf_dscl/convnextv2_tiny_lf_dscl_summary.json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", default="results/graphic_compensation/bt1")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=100000)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)
    checkpoint = resolve_checkpoint(args.summary, args.checkpoint)

    loader_args = Namespace(
        data_dir=Path(args.data_dir),
        batch_size=64,
        num_workers=2,
        seed=args.seed,
        test_size=0.2,
        val_size=0.1,
        image_size=args.image_size,
        otsu=False,
        no_standard_augmentation=False,
        use_lf=True,
        use_rectangular_erasure=False,
        use_rect_lacuna=False,
        lacuna_min=1,
        lacuna_max=4,
    )
    _, _, test_loader, label_encoder = build_loaders(loader_args)
    classes = label_encoder.classes_.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier("convnextv2_tiny", num_classes=len(classes), pretrained=False, in_chans=1, image_size=args.image_size).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())

    true_labels = label_encoder.inverse_transform(np.array(all_true))
    pred_labels = label_encoder.inverse_transform(np.array(all_pred))

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    report = classification_report(true_labels, pred_labels, labels=classes, target_names=classes, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(out / "classification_report_bt1_convnextv2_lf_dscl.csv")

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(out / "bt1_confusion_matrix_counts.csv")

    support = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, support, out=np.zeros_like(cm, dtype=float), where=support != 0)
    pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(out / "bt1_confusion_matrix_normalized_by_true.csv")

    directed_rows = []
    for i, true_class in enumerate(classes):
        true_support = cm[i, :].sum()
        for j, pred_class in enumerate(classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count:
                directed_rows.append({
                    "true": true_class, "pred": pred_class, "count": count,
                    "true_support": int(true_support),
                    "rate_within_true_class": count / true_support if true_support else 0.0,
                })
    pd.DataFrame(directed_rows).sort_values(["rate_within_true_class", "count"], ascending=False).to_csv(
        out / "bt1_directed_confusions.csv", index=False
    )

    pair_df = make_pairwise_scores(cm_df)
    pair_df.to_csv(out / "bt1_pairwise_confusions.csv", index=False)
    pair_df[pair_df["total_confusions"] > 0].to_csv(out / "bt1_pairwise_confusions_nonzero.csv", index=False)

    global_row = permutation_test_mean_difference(pair_df, n_perm=args.n_perm, seed=args.seed)
    pd.DataFrame([global_row]).to_csv(out / "bt1_global_pairwise_test.csv", index=False)

    exposure_row = exposure_weighted_test(pair_df, n_perm=args.n_perm, seed=args.seed)
    pd.DataFrame([exposure_row]).to_csv(out / "bt1_exposure_weighted_pairwise_test.csv", index=False)

    compact = {
        "checkpoint": str(checkpoint),
        "summary": str(args.summary),
        "image_size": args.image_size,
        "seed": args.seed,
        "test_size_n": int(len(true_labels)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    with (out / "bt1_eval_summary.json").open("w") as f:
        json.dump(compact, f, indent=2)

    # Figure used in the paper.
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(np.arange(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("ConvNeXt-V2 LF+DSCL confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "convnextv2_confusion_matrix.png", dpi=300)
    fig.savefig(out / "convnextv2_confusion_matrix.pdf")
    print(json.dumps(compact, indent=2))
    print("Saved outputs to", out)

if __name__ == "__main__":
    main()
