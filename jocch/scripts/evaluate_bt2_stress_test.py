#!/usr/bin/env python3
"""
Evaluate the main ConvNeXt-V2 LF+DSCL checkpoint on the matched BT2 subset.
"""
from __future__ import annotations
import argparse, json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from scripts.train_timm_backbone import TimmClassifier, build_loaders
from scripts.graphic_compensation_utils import make_pairwise_scores, permutation_test_mean_difference, load_summary, resolve_checkpoint

def preprocess_image_2d(image_path, size=(64, 64), otsu=False):
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    if otsu:
        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_np = 255 - img_np
    img_resized = cv2.resize(img_np, size, interpolation=cv2.INTER_AREA)
    return img_resized.astype(np.float32) / 255.0

class BT2MatchedDataset(Dataset):
    def __init__(self, df, path_col, label_col, transform, label_encoder, image_size=64):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.label_encoder = label_encoder
        self.image_size = image_size
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = (preprocess_image_2d(row[self.path_col], size=(self.image_size, self.image_size)) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = self.transform(img) if self.transform is not None else img
        y = int(self.label_encoder.transform([row[self.label_col]])[0])
        return img, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--summary", default="models/convnextv2_lf_dscl/convnextv2_tiny_lf_dscl_summary.json")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--bt2-csv", default="data/bt2_matched/bt2_matched_all_annotations_resnet_analog_metadata.csv")
    ap.add_argument("--output-dir", default="results/graphic_compensation/bt2")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=100000)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.summary, args.checkpoint)
    bt2_df = pd.read_csv(args.bt2_csv)
    path_col = "matched_crop_path" if "matched_crop_path" in bt2_df.columns else "crop_path"
    label_col = "letter"

    base_args = Namespace(
        data_dir=Path(args.data_dir), batch_size=64, num_workers=2, seed=args.seed,
        test_size=0.2, val_size=0.1, image_size=args.image_size, otsu=False,
        no_standard_augmentation=False, use_lf=True, use_rectangular_erasure=False,
        use_rect_lacuna=False, lacuna_min=1, lacuna_max=4
    )
    _, _, test_loader, label_encoder = build_loaders(base_args)
    classes = label_encoder.classes_.tolist()
    ds = test_loader.dataset
    while hasattr(ds, "dataset"): ds = ds.dataset
    eval_transform = ds.transform

    unknown = sorted(set(bt2_df[label_col]) - set(classes))
    if unknown: raise ValueError(f"BT2 labels not in model classes: {unknown}")
    missing = bt2_df[path_col].apply(lambda p: not Path(str(p)).exists()).sum()
    if missing: raise FileNotFoundError(f"{missing} BT2 crop paths are missing.")

    bt2_loader = DataLoader(BT2MatchedDataset(bt2_df, path_col, label_col, eval_transform, label_encoder, image_size=args.image_size), batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier("convnextv2_tiny", num_classes=len(classes), pretrained=False, in_chans=1, image_size=args.image_size).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state); model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in bt2_loader:
            pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
            all_pred.extend(pred.tolist()); all_true.extend(y.numpy().tolist())
    true_labels = label_encoder.inverse_transform(np.array(all_true))
    pred_labels = label_encoder.inverse_transform(np.array(all_pred))

    acc = accuracy_score(true_labels, pred_labels)
    macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    report = classification_report(true_labels, pred_labels, labels=classes, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "convnextv2_bt2_matched_classification_report.csv")
    cm_df = pd.DataFrame(confusion_matrix(true_labels, pred_labels, labels=classes), index=classes, columns=classes)
    cm_df.to_csv(out / "convnextv2_bt2_matched_confusion_matrix_counts.csv")
    pair_df = make_pairwise_scores(cm_df)
    pair_df.to_csv(out / "convnextv2_bt2_matched_all_pairwise_scores.csv", index=False)
    pd.DataFrame([permutation_test_mean_difference(pair_df, n_perm=args.n_perm, seed=args.seed)]).to_csv(out / "convnextv2_bt2_matched_global_pairwise_test.csv", index=False)

    with (out / "convnextv2_bt2_matched_eval_summary.json").open("w") as f:
        json.dump({"N": len(true_labels), "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted, "checkpoint": str(checkpoint), "bt2_csv": str(args.bt2_csv)}, f, indent=2)
    print("N:", len(true_labels), "Accuracy:", acc, "Macro-F1:", macro, "Weighted-F1:", weighted)
    print("Saved outputs to", out)

if __name__ == "__main__":
    main()
