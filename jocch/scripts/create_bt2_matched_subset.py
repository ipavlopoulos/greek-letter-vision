#!/usr/bin/env python3
"""
Create the matched BT2 damaged-character subset used for the stress test.
Requires Hell-Date annotation JSON files and full papyrus images.
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image
from tqdm import tqdm

ALLOWED = {
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
    "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
}
GREEK_TO_NAME = {
    "Α":"Alpha","α":"Alpha","Β":"Beta","β":"Beta","Γ":"Gamma","γ":"Gamma",
    "Δ":"Delta","δ":"Delta","Ε":"Epsilon","ε":"Epsilon","Ζ":"Zeta","ζ":"Zeta",
    "Η":"Eta","η":"Eta","Θ":"Theta","θ":"Theta","Ι":"Iota","ι":"Iota",
    "Κ":"Kappa","κ":"Kappa","Λ":"Lambda","λ":"Lambda","Μ":"Mu","μ":"Mu",
    "Ν":"Nu","ν":"Nu","Ξ":"Xi","ξ":"Xi","Ο":"Omicron","ο":"Omicron",
    "Π":"Pi","π":"Pi","Ρ":"Rho","ρ":"Rho","Σ":"Sigma","σ":"Sigma","ς":"Sigma",
    "Τ":"Tau","τ":"Tau","Υ":"Upsilon","υ":"Upsilon","Φ":"Phi","φ":"Phi",
    "Χ":"Chi","χ":"Chi","Ψ":"Psi","ψ":"Psi","Ω":"Omega","ω":"Omega"
}
TARGET_COUNTS = {
    "Alpha":139,"Beta":67,"Chi":85,"Delta":113,"Epsilon":138,"Eta":124,
    "Gamma":105,"Iota":141,"Kappa":127,"Lambda":117,"Mu":126,"Nu":134,
    "Omega":126,"Omicron":136,"Phi":83,"Pi":127,"Psi":17,"Rho":133,
    "Sigma":138,"Tau":139,"Theta":86,"Upsilon":133,"Xi":47,"Zeta":22,
}

def normalize_label(label):
    if label is None: return None
    label = str(label).strip()
    if label in ALLOWED: return label
    if label in GREEK_TO_NAME: return GREEK_TO_NAME[label]
    for name in ALLOWED:
        if label.lower() == name.lower() or name.lower() in label.lower():
            return name
    return None

def get_image_filename(img_info):
    for key in ["file_name", "filename", "name", "path"]:
        if key in img_info and img_info[key]:
            return Path(str(img_info[key])).name
    return None

def parse_tm_from_filename(filename):
    m = re.search(r"TM(\d+)", str(filename))
    return int(m.group(1)) if m else None

def crop_bbox(img, bbox, pad=2):
    x, y, w, h = bbox
    x1 = max(0, int(round(x)) - pad); y1 = max(0, int(round(y)) - pad)
    x2 = min(img.width, int(round(x+w)) + pad); y2 = min(img.height, int(round(y+h)) + pad)
    return img.crop((x1, y1, x2, y2))

def crop_from_annotation_file(annotation_path, full_image_dir, crop_dir, source_name):
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    categories = ann["categories"]; images = ann["images"]; annotations = ann["annotations"]
    cat_id_to_name = {c["id"]: c.get("name") or c.get("label") or c.get("title") for c in categories}
    image_id_to_info = {img["id"]: img for img in images}
    rows = []
    skip = Counter()
    for a in tqdm(annotations, desc=f"Cropping BT2 from {annotation_path.name}"):
        tags = a.get("tags", {})
        base_types = tags.get("BaseType", [])
        if isinstance(base_types, str): base_types = [base_types]
        if "bt2" not in [str(x).lower() for x in base_types]:
            continue
        raw_label = cat_id_to_name.get(a.get("category_id"))
        letter = normalize_label(raw_label)
        if letter is None:
            skip["non_alphabetic_or_unmatched_label"] += 1; continue
        bbox = a.get("bbox")
        img_info = image_id_to_info.get(a.get("image_id"))
        if bbox is None or img_info is None:
            skip["missing_bbox_or_image_info"] += 1; continue
        img_filename = get_image_filename(img_info)
        if img_filename is None:
            skip["missing_image_filename"] += 1; continue
        img_path = full_image_dir / img_filename
        if not img_path.exists():
            skip["missing_full_image"] += 1; continue
        img = Image.open(img_path).convert("L")
        crop = crop_bbox(img, bbox, pad=2)
        crop_filename = f"{letter}_BT2_{source_name}_ann{a.get('id')}_img{a.get('image_id')}_{img_filename}"
        crop_path = crop_dir / crop_filename
        crop.save(crop_path)
        rows.append({
            "annotation_id": a.get("id"), "image_id": a.get("image_id"),
            "image_filename": img_filename, "full_image_path": str(img_path),
            "crop_path": str(crop_path), "letter": letter, "label_raw": raw_label,
            "bbox": bbox, "area": a.get("area"), "base_types": ",".join(map(str, base_types)),
            "TM": parse_tm_from_filename(img_filename), "source_annotation_file": annotation_path.name,
        })
    if skip:
        print("Skip reasons:", skip)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hell-date-dir", default="Hell-Date")
    ap.add_argument("--full-image-dir", default="dataset")
    ap.add_argument("--output-dir", default="data/bt2_matched")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    hell = Path(args.hell_date_dir); full = Path(args.full_image_dir); out = Path(args.output_dir)
    crop_all = out / "bt2_letter_crops_all_annotations"; crop_all.mkdir(parents=True, exist_ok=True)
    matched_crops = out / "crops"; matched_crops.mkdir(parents=True, exist_ok=True)

    rows = []
    for fname, source in [("annotations_training.json","train"), ("annotations_test.json","test")]:
        path = hell / fname
        if path.exists():
            rows.extend(crop_from_annotation_file(path, full, crop_all, source))
    bt2_all_df = pd.DataFrame(rows)
    all_csv = out / "bt2_letter_crops_all_annotations_metadata.csv"
    bt2_all_df.to_csv(all_csv, index=False)

    parts = []
    for letter, target in TARGET_COUNTS.items():
        part = bt2_all_df[bt2_all_df["letter"] == letter].copy()
        if len(part) < target:
            raise ValueError(f"Not enough BT2 crops for {letter}: available {len(part)}, target {target}")
        parts.append(part.sample(n=target, random_state=args.seed))
    matched = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    copied_paths = []
    for i, row in tqdm(matched.iterrows(), total=len(matched), desc="Copying matched BT2 crops"):
        src = Path(row["crop_path"])
        dst = matched_crops / f"{i:05d}_{src.name}"
        shutil.copy2(src, dst)
        copied_paths.append(str(dst))
    matched["matched_crop_path"] = copied_paths

    counts = pd.DataFrame({
        "target_count": pd.Series(TARGET_COUNTS),
        "matched_count": matched["letter"].value_counts().sort_index(),
        "available_bt2": bt2_all_df["letter"].value_counts().sort_index(),
    }).fillna(0).astype(int)

    # Legacy filename from the working notebook plus clean aliases.
    legacy_csv = out / "bt2_matched_all_annotations_resnet_analog_metadata.csv"
    legacy_counts = out / "bt2_matched_all_annotations_resnet_analog_counts.csv"
    matched.to_csv(legacy_csv, index=False); counts.to_csv(legacy_counts)
    matched.to_csv(out / "bt2_matched_metadata.csv", index=False); counts.to_csv(out / "bt2_matched_counts.csv")
    print("Full BT2 pool:", len(bt2_all_df))
    print("Matched BT2 subset:", len(matched))
    print("Saved:", all_csv)
    print("Saved:", legacy_csv)
    print("Saved:", legacy_counts)

if __name__ == "__main__":
    main()
