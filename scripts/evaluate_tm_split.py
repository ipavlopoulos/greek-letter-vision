#!/usr/bin/env python3
"""
Evaluate the ConvNeXt-V2 LF+DSCL model trained under the TM-level split.
"""
from __future__ import annotations
import argparse, json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from scripts.train_timm_backbone import TimmClassifier, build_loaders
from scripts.graphic_compensation_utils import make_pairwise_scores, permutation_test_mean_difference

def resize_with_padding(img, image_size=64, fill=255):
    img = img.convert("L")
    w, h = img.size
    scale = min(image_size / w, image_size / h)
    new_w = max(1, int(round(w * scale))); new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), color=fill)
    canvas.paste(img, ((image_size-new_w)//2, (image_size-new_h)//2))
    return canvas

class TMTestDataset(Dataset):
    def __init__(self, df, transform, label_encoder, image_size=64):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = label_encoder
        self.image_size = image_size
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(str(row["path"])) if "path" in row and pd.notna(row["path"]) else Path("data/hellchar/cliplets") / str(row["filename"])
        if not img_path.exists():
            img_path = Path("data/hellchar/cliplets") / str(row["filename"])
        img = resize_with_padding(Image.open(img_path).convert("L"), image_size=self.image_size)
        img = self.transform(img)
        y = int(self.label_encoder.transform([str(row["letter"])])[0])
        return img, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tm-split-csv", default="data/splits/hellchar_tm_level_split_seed.csv")
    ap.add_argument("--summary", default="models/convnextv2_lf_dscl_tm_split/timm_summary.json")
    ap.add_argument("--checkpoint", default="models/convnextv2_lf_dscl_tm_split/best_timm_model.pth")
    ap.add_argument("--output-dir", default="results/graphic_compensation/tm_split")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=100000)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(args.summary) as f: summary = json.load(f)
    split_df = pd.read_csv(args.tm_split_csv)
    test_df = split_df[split_df["split"] == "test"].copy()

    base_args = Namespace(
        data_dir=Path(args.data_dir), batch_size=64, num_workers=2, seed=args.seed,
        test_size=0.2, val_size=0.1, image_size=args.image_size, otsu=False,
        no_standard_augmentation=False, use_lf=True, use_rectangular_erasure=False,
        use_rect_lacuna=False, lacuna_min=1, lacuna_max=4
    )
    _, _, orig_test_loader, label_encoder = build_loaders(base_args)
    classes = label_encoder.classes_.tolist()
    ds = orig_test_loader.dataset
    while hasattr(ds, "dataset"): ds = ds.dataset
    eval_transform = ds.transform

    test_loader = DataLoader(TMTestDataset(test_df, eval_transform, label_encoder, image_size=args.image_size), batch_size=64, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier("convnextv2_tiny", num_classes=len(classes), pretrained=False, in_chans=1, image_size=args.image_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
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
    macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    report = classification_report(true_labels, pred_labels, labels=classes, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "convnextv2_tm_split_classification_report.csv")
    cm_df = pd.DataFrame(confusion_matrix(true_labels, pred_labels, labels=classes), index=classes, columns=classes)
    cm_df.to_csv(out / "convnextv2_tm_split_confusion_matrix_counts.csv")
    pair_df = make_pairwise_scores(cm_df)
    pair_df.to_csv(out / "convnextv2_tm_split_pairwise_scores.csv", index=False)
    pd.DataFrame([permutation_test_mean_difference(pair_df, n_perm=args.n_perm, seed=args.seed)]).to_csv(out / "convnextv2_tm_split_global_pairwise_test.csv", index=False)

    with (out / "convnextv2_tm_split_eval_summary.json").open("w") as f:
        json.dump({"N": len(true_labels), "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted, "summary": str(args.summary), "checkpoint": str(args.checkpoint)}, f, indent=2)
    print("N:", len(true_labels), "Accuracy:", acc, "Macro-F1:", macro, "Weighted-F1:", weighted)
    print("Saved outputs to", out)

if __name__ == "__main__":
    main()
