# JOCCH Extension — Graphic Compensation in Ancient Greek Documentary Hands

This directory contains the study-specific training, evaluation, and robustness
scripts accompanying the forthcoming article:

**Paraskevi Platanou, Lavinia Ferretti, Isabelle Marthot-Santaniello,
Giuseppe De Gregorio, Spiros Barbakos, Maria Konstantinidou,
Asimina Paparrigopoulou, and John Pavlopoulos.**

**“Graphic Compensation in Ancient Greek Documentary Hands:
A Computational Paleographic Analysis from Handwritten Character Recognition.”**

*ACM Journal on Computing and Cultural Heritage (JOCCH), to appear, 2026.*

---

## Overview

This study builds on the Ancient Greek character-recognition framework developed
in the companion ICDAR 2026 work
*Learning Diachronic Representations of Ancient Greek Letterforms*.

The two studies share:

- the **Hell-Char** 24-class character benchmark;
- the common preprocessing and character-classification framework;
- **LF** augmentation;
- **DSCL** training;
- the underlying representation-learning infrastructure.

The JOCCH study, however, addresses a different research question. Rather than
focusing primarily on diachronic representation learning, it investigates
**structured character confusions as computational evidence for graphic
compensation in Ancient Greek documentary hands**.

Its principal recognition model is:

**ConvNeXt-V2 Tiny + LF + DSCL**

The final model achieves:

- **Accuracy:** 0.864
- **Macro-F1:** 0.860

on the held-out Hell-Char test set.

---

## Shared data and infrastructure

The JOCCH experiments do **not** duplicate the Hell-Char data.

They reuse the shared dataset available at:

```text
../data/hellchar/
```

and the shared implementation of LF, DSCL, datasets, and training utilities
provided by the main repository.

The generic ConvNeXt/ViT training implementation is available at:

```text
../scripts/train_timm_backbone.py
```

The scripts in this directory contain the **JOCCH-specific experimental
configurations and analyses**.

---

## Final model and reproducibility artifacts

The final ConvNeXt-V2 Tiny + LF + DSCL checkpoint is hosted separately on the
Hugging Face Model Hub because of its size:

**https://huggingface.co/pplatanou/greek-letter-convnextv2-jocch**

The Hugging Face repository contains:

```text
best_convnextv2_tiny_lf_dscl.pth

artifacts/
  demo_config.json
  hellchar_reference_embeddings.npy
  hellchar_reference_metadata.csv
  hellchar_reference_umap.npy
  umap_reducer.joblib
```

The embedding and UMAP artifacts support the interactive demonstrator described
below.

---

## Interactive demonstrator

A live embedding explorer is available at:

**https://diachronic-greek-letterforms.streamlit.app/**

The demo allows a user to upload a single Ancient Greek handwritten character
crop and:

- obtain the ConvNeXt-V2 Tiny + LF + DSCL prediction;
- extract its learned representation;
- project it into the fixed Hell-Char UMAP visualization;
- inspect representative Hell-Char letterforms;
- retrieve the five nearest Hell-Char reference characters.

Nearest neighbours are ranked by **cosine similarity in the original normalized
ConvNeXt embedding space**.

UMAP is used **only as a two-dimensional exploratory visualization** and not as
the space in which nearest-neighbour similarity is computed.

The deployment code is maintained under:

```text
../demo/
```

---

## Directory contents

The JOCCH-specific scripts are organized as follows:

| Script | Purpose |
|---|---|
| `run_train_main_convnextv2_lfdscl.sh` | Main ConvNeXt-V2 Tiny + LF + DSCL training configuration used for the JOCCH model |
| `evaluate_bt1_convnextv2_confusions.py` | Main Hell-Char evaluation and pairwise character-confusion analysis |
| `create_tm_level_split.py` | Constructs the TM-disjoint train/validation/test split |
| `train_timm_backbone_tm_split.py` | Trains the ConvNeXt model under the TM-level split |
| `evaluate_tm_split.py` | Evaluates recognition and confusion behaviour under TM-level separation |
| `create_bt2_matched_subset.py` | Constructs the matched BT2 degradation subset |
| `evaluate_bt2_stress_test.py` | Evaluates robustness under naturally degraded BT2 character images |
| `evaluate_objective_stability.py` | Assesses whether major confusion patterns remain stable across training objectives |
| `graphic_compensation_utils.py` | Shared utilities used by the JOCCH analyses |

---

## Main experimental setting

The principal JOCCH experiment uses:

```text
Backbone:        ConvNeXt-V2 Tiny
Input size:      64 × 64
Pretraining:     ImageNet pretrained
LF:              enabled
LF lacunae:      1–4
DSCL:            enabled
λSCL:            0.1
Seed:            42
```

The final held-out Hell-Char performance is:

```text
Accuracy:        0.864387
Macro-F1:        0.859942
Weighted-F1:     0.864123
```

The final model checkpoint is:

```text
best_convnextv2_tiny_lf_dscl.pth
```

and is distributed through the Hugging Face repository linked above.

---

## Main analysis components

### 1. Hell-Char / BT1 confusion analysis

The principal analysis examines **symmetric pairwise character confusions**
rather than relying only on aggregate classification accuracy.

For a pair of classes \(x\) and \(y\), the normalized bidirectional confusion
score is defined as:

```text
(M[x,y] + M[y,x]) / (N[x] + N[y])
```

where:

- `M[x,y]` is the number of examples of class `x` predicted as `y`;
- `M[y,x]` is the reverse confusion;
- `N[x]` and `N[y]` are the class supports.

This analysis is used to identify recurring visual relationships between letter
classes.

---

### 2. TM-level split

To test whether the observed confusion structure depends on material from the
same Trismegistos document appearing across data partitions, the study includes
a **TM-level split** in which no TM identifier is shared between train,
validation, and test sets.

This provides an additional robustness check against document-level leakage.

---

### 3. BT2 degradation stress test

The BT2 experiment evaluates the final ConvNeXt-V2 model on naturally degraded
character crops drawn from Hell-Date.

The model is evaluated **without retraining or fine-tuning** on BT2.

The purpose of this analysis is not to establish BT2 as a second standard test
set, but to examine whether structured character-confusion tendencies persist
under substantially more degraded visual conditions.

---

### 4. Objective-stability analysis

The study also examines whether the principal pairwise confusion patterns remain
visible across alternative training objectives.

This analysis distinguishes effects that are highly specific to a single
objective from broader visual relationships repeatedly learned by the models.

---

## Interpretation

The JOCCH study does not treat recognition errors solely as failures of the
classifier.

Instead, recurrent and asymmetric confusions are examined as **structured
computational observations about the visual relationships between Ancient Greek
letterforms**.

These patterns are interpreted together with:

- overall recognition performance;
- class support;
- directional confusion rates;
- TM-level robustness;
- objective stability;
- degradation stress testing.

The resulting evidence is used to support the paleographic discussion of
**graphic compensation**, while avoiding the claim that model confusion alone
proves a historical or scribal relationship.

---

## Relationship to the ICDAR study

The companion ICDAR 2026 study uses ResNet18 as its principal architecture and
focuses on diachronic representation learning across Hell-Char, PaLit-Char, and
Med-Char.

The JOCCH study reuses the common Hell-Char/LF/DSCL framework but:

1. adopts **ConvNeXt-V2 Tiny + LF + DSCL** as the principal recognition model;
2. shifts the main analytical target from diachronic generalization to
   **structured character confusions**;
3. introduces additional TM-level, objective-stability, and BT2 degradation
   analyses;
4. connects the learned confusion structure to a paleographic question about
   graphic compensation.

For the full ICDAR reproduction workflow, see the
[root README](../README.md).

---

## Citation

If you use the JOCCH model, confusion-analysis scripts, embedding artifacts, or
interactive demonstrator, please cite:

```bibtex
@article{platanou2026graphic,
  author  = {Platanou, Paraskevi and
             Ferretti, Lavinia and
             Marthot-Santaniello, Isabelle and
             De Gregorio, Giuseppe and
             Barbakos, Spiros and
             Konstantinidou, Maria and
             Paparrigopoulou, Asimina and
             Pavlopoulos, John},
  title   = {Graphic Compensation in Ancient Greek Documentary Hands:
             A Computational Paleographic Analysis from Handwritten Character Recognition},
  journal = {ACM Journal on Computing and Cultural Heritage},
  year    = {2026},
  note    = {To appear}
}
```

For the shared representation-learning framework and diachronic experiments,
please also see:

```bibtex
@inproceedings{pavlopoulos2026diachronic,
  title     = {Learning Diachronic Representations of Ancient Greek Letterforms},
  author    = {Pavlopoulos, John and Barbakos, Spyros and Ferretti, Lavinia and
               Voulgarakis, Dionysis and Paparrigopoulou, Asimina and
               Konstantinidou, Maria and De Gregorio, Giuseppe and
               Marthot-Santaniello, Isabelle and Platanou, Paraskevi and
               Essler, Holger},
  booktitle = {International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2026}
}
```

---

## License

The code and datasets distributed through the main repository are released under
**CC BY 4.0**; see [`../LICENSE`](../LICENSE).

The final ConvNeXt-V2 checkpoint and associated embedding artifacts are
distributed separately through the
[Hugging Face Model Hub](https://huggingface.co/pplatanou/greek-letter-convnextv2-jocch).

The datasets build on Hell-Date and on material from securely dated papyri and
manuscripts. Users should also cite the originating datasets and resources where
applicable.
