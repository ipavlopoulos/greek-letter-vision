#!/usr/bin/env python3
"""
Train a timm backbone using an existing TM-level split CSV.
This monkey-patches scripts.train_timm_backbone.build_loaders, so all model,
loss, optimizer, scheduler and training code remains inherited from the original script.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import scripts.train_timm_backbone as ttb

_ORIGINAL_BUILD_LOADERS = ttb.build_loaders

def _get_transform_from_loader(loader):
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    if hasattr(ds, "transform"):
        return ds.transform
    raise AttributeError("Could not find transform in original loader dataset.")

def _resize_with_padding(img, image_size=64, fill=255):
    img = img.convert("L")
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Empty image encountered.")
    scale = min(image_size / w, image_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), color=fill)
    canvas.paste(img, ((image_size - new_w)//2, (image_size - new_h)//2))
    return canvas

class TMCSVDataset(Dataset):
    def __init__(self, df, transform, label_encoder, image_size=64):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = label_encoder
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if "path" in row and pd.notna(row["path"]):
            img_path = Path(str(row["path"]))
        else:
            img_path = Path("data/hellchar/cliplets") / str(row["filename"])
        if not img_path.exists():
            img_path = Path("data/hellchar/cliplets") / str(row["filename"])
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        img = Image.open(img_path).convert("L")
        img = _resize_with_padding(img, image_size=self.image_size, fill=255)
        img = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)

        label = str(row["letter"])
        y = int(self.label_encoder.transform([label])[0])
        return img, y

def build_loaders(args):
    orig_train_loader, orig_val_loader, orig_test_loader, label_encoder = _ORIGINAL_BUILD_LOADERS(args)
    train_transform = _get_transform_from_loader(orig_train_loader)
    eval_transform = _get_transform_from_loader(orig_test_loader)

    split_csv = Path(getattr(args, "tm_split_csv", "data/splits/hellchar_tm_level_split_seed.csv"))
    split_df = pd.read_csv(split_csv)
    required_cols = {"filename", "letter", "TM", "split"}
    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        raise ValueError(f"TM split CSV missing columns: {missing_cols}")

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_tms = set(train_df["TM"].astype(str))
    val_tms = set(val_df["TM"].astype(str))
    test_tms = set(test_df["TM"].astype(str))
    assert len(train_tms & val_tms) == 0, "TM leakage: train-val overlap"
    assert len(train_tms & test_tms) == 0, "TM leakage: train-test overlap"
    assert len(val_tms & test_tms) == 0, "TM leakage: val-test overlap"

    train_dataset = TMCSVDataset(train_df, train_transform, label_encoder, image_size=args.image_size)
    val_dataset = TMCSVDataset(val_df, eval_transform, label_encoder, image_size=args.image_size)
    test_dataset = TMCSVDataset(test_df, eval_transform, label_encoder, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("\n[TM-LEVEL SPLIT OVERRIDE]")
    print("Split CSV:", split_csv)
    print("Train examples:", len(train_df), "| TMs:", train_df["TM"].nunique())
    print("Val examples:", len(val_df), "| TMs:", val_df["TM"].nunique())
    print("Test examples:", len(test_df), "| TMs:", test_df["TM"].nunique())
    print("Train ∩ Val TMs:", len(train_tms & val_tms))
    print("Train ∩ Test TMs:", len(train_tms & test_tms))
    print("Val ∩ Test TMs:", len(val_tms & test_tms))
    print("Classes:", list(label_encoder.classes_))
    print("[Only the split has changed; all other training settings are inherited.]\n")
    return train_loader, val_loader, test_loader, label_encoder

# Monkey-patch original training script.
ttb.build_loaders = build_loaders

if __name__ == "__main__":
    # Ensure the original parser tolerates no --tm-split-csv argument.
    # Use the default data/splits/hellchar_tm_level_split_seed.csv.
    ttb.main()
