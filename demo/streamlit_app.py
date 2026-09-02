from __future__ import annotations

import base64, io, json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import timm
import torch
import torch.nn.functional as F
import umap
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.neighbors import NearestNeighbors

HF_MODEL_REPO = "pplatanou/greek-letter-convnextv2-jocch"
GITHUB_RAW_CLIPLETS = "https://raw.githubusercontent.com/ipavlopoulos/diachronic-greek-letterforms/main/data/hellchar/cliplets"
N_NEIGHBORS = 5

st.set_page_config(page_title="Greek Letter Embedding Explorer", page_icon="✍️", layout="wide")
st.title("Greek Letter Embedding Explorer")
st.caption("ConvNeXt-V2 Tiny + LF + DSCL — JOCCH interactive demonstrator")
st.markdown("""
Upload a **single cropped Ancient Greek handwritten character**. The explorer predicts its
class, extracts its learned representation, places it in the Hell-Char embedding map, and
retrieves its nearest reference characters.

**Input scope:** historical handwritten single-character crops.
""")

class TimmClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False, in_chans=1, image_size=64):
        super().__init__()
        kwargs = {"img_size": image_size} if model_name.startswith("vit") else {}
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes,
            in_chans=in_chans, **kwargs
        )

    def forward(self, x):
        return self.model(x)

    def get_embeddings(self, x):
        features = self.model.forward_features(x)
        try:
            z = self.model.forward_head(features, pre_logits=True)
        except TypeError:
            z = self.model.forward_head(features)
        if z.ndim > 2:
            z = torch.flatten(z, 1)
        return z

@st.cache_resource(show_spinner="Loading JOCCH model and reference embedding space…")
def load_resources():
    def hub(filename):
        return hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)

    model_path = hub("best_convnextv2_tiny_lf_dscl.pth")
    config_path = hub("artifacts/demo_config.json")
    metadata_path = hub("artifacts/hellchar_reference_metadata.csv")
    embeddings_path = hub("artifacts/hellchar_reference_embeddings.npy")
    umap_path = hub("artifacts/hellchar_reference_umap.npy")
    reducer_path = hub("artifacts/umap_reducer.joblib")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    classes = list(config["classes"])
    model_name = config.get("model", "convnextv2_tiny")
    image_size = int(config.get("image_size", 64))

    df = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path).astype(np.float32)
    coords = np.load(umap_path).astype(np.float32)
    df["UMAP1"] = coords[:, 0]
    df["UMAP2"] = coords[:, 1]

    try:
        reducer = joblib.load(reducer_path)
    except Exception:
        reducer = umap.UMAP(
            n_neighbors=20, min_dist=0.15, metric="cosine",
            n_components=2, random_state=42, low_memory=True
        )
        reducer.fit(embeddings)

    nn_index = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="cosine", algorithm="brute")
    nn_index.fit(embeddings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimmClassifier(
        model_name, num_classes=len(classes), pretrained=False,
        in_chans=1, image_size=image_size
    ).to(device)

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    return {
        "config": config, "classes": classes, "model_name": model_name,
        "image_size": image_size, "df": df, "embeddings": embeddings,
        "coords": coords, "reducer": reducer, "nn": nn_index,
        "device": device, "model": model
    }

R = load_resources()

@st.cache_data(show_spinner=False)
def fetch_reference_image_bytes(filename):
    url = f"{GITHUB_RAW_CLIPLETS}/{filename}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content

def fetch_reference_image(filename):
    return Image.open(io.BytesIO(fetch_reference_image_bytes(filename))).convert("L").copy()

def image_to_data_uri(img):
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def preprocess_uploaded_image(img_pil):
    img = img_pil.convert("L")
    arr = np.asarray(img)
    arr = 255 - arr
    arr = cv2.resize(arr, (R["image_size"], R["image_size"]), interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0)
    return (x - 0.5) / 0.5

def analyse(img):
    x = preprocess_uploaded_image(img).unsqueeze(0).to(R["device"])
    with torch.inference_mode():
        z = R["model"].get_embeddings(x)
        z = F.normalize(z, p=2, dim=1)
        probs = torch.softmax(R["model"](x), dim=1)

    z_np = z.cpu().numpy().astype(np.float32)
    probs_np = probs.cpu().numpy()[0]
    pred_idx = int(np.argmax(probs_np))
    query_xy = R["reducer"].transform(z_np)[0]

    distances, indices = R["nn"].kneighbors(z_np, n_neighbors=N_NEIGHBORS)
    neighbors = R["df"].iloc[indices[0]].copy().reset_index(drop=True)
    neighbors["cosine_similarity"] = 1.0 - distances[0]
    top_idx = np.argsort(-probs_np)[:5]

    return {
        "pred_letter": R["classes"][pred_idx],
        "pred_conf": float(probs_np[pred_idx]),
        "query_xy": query_xy,
        "neighbors": neighbors,
        "top5": [(R["classes"][int(i)], float(probs_np[int(i)])) for i in top_idx],
    }

@st.cache_data(show_spinner=False)
def representative_prototype_indices():
    labels = R["df"]["letter"].astype(str).to_numpy()
    out = []
    for letter in R["classes"]:
        idx = np.where(labels == letter)[0]
        if len(idx) == 0:
            continue
        z = R["embeddings"][idx]
        centroid = z.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        best = idx[int(np.argmax(z @ centroid))]
        out.append(int(best))
    return out

def make_umap_plot(result, uploaded_img, show_prototypes):
    df = R["df"]
    fig = px.scatter(
        df, x="UMAP1", y="UMAP2", color="letter",
        hover_data=["filename", "letter", "pred_letter"],
        opacity=0.34, title="Hell-Char learned embedding space"
    )

    xr = float(df["UMAP1"].max() - df["UMAP1"].min())
    yr = float(df["UMAP2"].max() - df["UMAP2"].min())
    sx, sy = max(xr * 0.035, 0.15), max(yr * 0.035, 0.15)

    if show_prototypes:
        for idx in representative_prototype_indices():
            row = df.iloc[idx]
            try:
                proto = fetch_reference_image(str(row["filename"]))
                fig.add_layout_image(dict(
                    source=image_to_data_uri(proto),
                    x=float(row["UMAP1"]), y=float(row["UMAP2"]),
                    xref="x", yref="y", sizex=sx, sizey=sy,
                    xanchor="center", yanchor="middle",
                    sizing="contain", opacity=0.95, layer="above"
                ))
            except Exception:
                pass

    qx, qy = result["query_xy"]
    fig.add_trace(go.Scatter(
        x=[float(qx)], y=[float(qy)], mode="markers",
        marker=dict(size=34, symbol="star", color="#ff7f0e",
                    line=dict(width=2, color="black")),
        name="Uploaded image"
    ))
    fig.add_layout_image(dict(
        source=image_to_data_uri(uploaded_img.convert("L")),
        x=float(qx), y=float(qy), xref="x", yref="y",
        sizex=sx * 1.8, sizey=sy * 1.8,
        xanchor="center", yanchor="middle",
        sizing="contain", opacity=1.0, layer="above"
    ))
    fig.add_annotation(
        x=float(qx), y=float(qy + sy * 1.25),
        text=f"Uploaded → {result['pred_letter']}",
        showarrow=True, arrowhead=2,
        bgcolor="rgba(255,255,255,0.88)"
    )
    fig.update_layout(height=760, legend_title_text="True letter")
    return fig

uploaded = st.file_uploader(
    "Upload a single-character crop",
    type=["png", "jpg", "jpeg", "webp"]
)
show_prototypes = st.checkbox(
    "Show one representative real Hell-Char image per letter class on the UMAP",
    value=True
)

if uploaded is None:
    st.info("Upload an image to start the analysis.")
else:
    img = Image.open(uploaded).convert("L")
    with st.spinner("Embedding and comparing the uploaded character…"):
        result = analyse(img)

    left, right = st.columns([1, 2])
    with left:
        st.image(img, caption="Uploaded character", width=180)
        st.subheader(f"Prediction: {result['pred_letter']}")
        st.metric("Classifier confidence", f"{result['pred_conf']:.4f}")
        top_df = pd.DataFrame(result["top5"], columns=["Letter", "Probability"])
        st.dataframe(top_df, hide_index=True, use_container_width=True)

    with right:
        st.plotly_chart(
            make_umap_plot(result, img, show_prototypes),
            use_container_width=True,
            config={"displaylogo": False}
        )

    st.subheader("Five nearest Hell-Char reference characters")
    cols = st.columns(N_NEIGHBORS)
    for col, (_, row) in zip(cols, result["neighbors"].iterrows()):
        with col:
            try:
                nimg = fetch_reference_image(str(row["filename"]))
                st.image(nimg, use_container_width=True)
            except Exception:
                st.warning("Preview unavailable")
            st.markdown(f"**{row['letter']}**")
            st.caption(f"cosine similarity = {row['cosine_similarity']:.3f}")

st.divider()
st.markdown("""
**Methodological note.** UMAP is used only as a two-dimensional exploratory visualization.
Nearest-neighbour similarity is calculated separately in the original normalized ConvNeXt
embedding space using cosine similarity.

**Model assets:** [pplatanou/greek-letter-convnextv2-jocch](https://huggingface.co/pplatanou/greek-letter-convnextv2-jocch)  
**Source/data framework:** [ipavlopoulos/diachronic-greek-letterforms](https://github.com/ipavlopoulos/diachronic-greek-letterforms)
""")
