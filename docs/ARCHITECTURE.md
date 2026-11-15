# Architecture: Hierarchical Open-Set Crop Health Network

## 1. Why the v1 service is being replaced

Audit of Govi Isuru v1 (`ai-service/`, November 2026):

| v1 component | Finding |
|---|---|
| Rice CNN (MobileNetV2, 8 classes) | 42.7 % test accuracy, macro-F1 0.42; Leaf Blast recall 9 %, Sheath Blight 7 % |
| Tea CNN | Splits are broken: val has no Blister Blight or Healthy images, and test has no Healthy images, so the reported 99 % val accuracy is invalid |
| Chili CNN | Val accuracy peaked at 97.5 % and ended at 88.9 %; data from a single source |
| Crop suitability | Random Forest trained on 450 rule-generated synthetic rows |
| Yield | Data has ±10 % random noise injected, and random variation is added to predictions at serving time |
| Inference | All 3 CNNs plus 5 TTA passes per request on full TensorFlow (~1.5 GB RAM); non-leaf check is an HSV heuristic |

## 2. Model

```
                   TRAINING (Colab)                                   SERVING (ONNX INT8, CPU)
 ┌─────────────────────────────────────────┐        ┌────────────────────────────────────────────────┐
 │ Teacher: frozen DINOv2 (+ CLIP) features │        │ Student CNN (~4–6 M params) – ONE forward pass │
 │ fusion + cosine heads                    │──KD──▶ │   ├─ gate head: rice | tea | chili |           │
 └─────────────────────────────────────────┘        │   │             other_plant | non_plant         │
                                                    │   ├─ per-crop disease heads (cosine/AM-Softmax) │
 Losses: logit KD + hierarchical CE                 │   ├─ embedding → prototype library (Mahalanobis)│
         + AM-Softmax + outlier exposure            │   └─ CAM heatmap from the same pass             │
                                                    └──────────────┬─────────────────────────────────┘
                                                                   ▼
                     decision: calibrated confidence + energy score + prototype distance
       NOT_A_LEAF · WRONG_CROP · KNOWN DISEASE · UNKNOWN_DISEASE (→ officer review) · LOW_CONFIDENCE
```

Design decisions and the evidence behind them (full references in RESEARCH.md):

- **Frozen foundation-model teacher.** Features from a frozen self-supervised model transfer to field photos
  much better than fine-tuned CNNs, and fine-tuning the backbone hurts that transfer [1]. Combining diverse
  backbones adds more [4].
- **Logit distillation into a small CNN.** Simple logit KD beats feature-alignment losses [5]. KD plus INT8 keeps
  accuracy at about 1–2 MB [6].
- **Compact class clusters plus outlier exposure** for unknown-disease rejection: AM-Softmax [7], max-logit or energy
  scores [8], and combining general with domain knowledge [9].
- **Prototype library** so an officer-verified new disease can be added without retraining [11]. Periodic retraining
  uses replay so old classes aren't forgotten [10].
- **Hierarchy** (crop gate first, then the crop's disease head) gives wrong-crop and non-leaf detection
  as first-class outputs.

## 3. New-disease workflow

1. The prediction is `UNKNOWN_DISEASE`: a leaf of a known crop that is far from every known prototype.
2. The service stores the embedding plus image reference. The backend owns the storage; the AI service only returns the embedding id.
3. The service clusters unknowns (HDBSCAN on embeddings); the backend shows clusters to officers.
4. An officer verifies a cluster and gives it a name. Its prototype is added to the library, so it is recognised immediately.
5. The next scheduled retrain turns it into a trained class, with replay of old data.

## 4. Evaluation protocol (paper)

| Aspect | Metric |
|---|---|
| Known diseases | macro-F1, balanced accuracy, per-class recall; in-domain **and leave-one-dataset-out** |
| Unknown diseases | `unknown_eval` classes (never trained on): AUROC, FPR@95 %TPR, OSCR |
| Gate | rejection rate for non-plant images, other plants and the wrong crop |
| Trust | ECE after temperature scaling; risk at 80 % coverage; AURC |
| Cost | parameters, MB on disk, peak RAM, p50/p95 latency on 1 vCPU |
| Ablations | KD, AM-Softmax, outlier exposure, prototype library, hierarchy |

All results are reported over 3 seeds, with McNemar or Wilcoxon tests where models are compared.

## 5. Service (phase 6)

- Runtime dependencies: fastapi, uvicorn, onnxruntime, numpy, pillow, pydantic (no torch, TensorFlow or OpenCV).
- Weights and a `manifest.json` (classes, thresholds, sha256) are pulled from the HF Hub at image build time.
- Versioned API (`/v1/disease/predict`, `/v1/yield/*`, `/v1/suitability`, `/health`, `/ready`, `/metadata`),
  API-key auth, and a v1-compatible `/predict` shim for the backend during migration.
- Targets: image size ≤ 350 MB, RAM ≤ 300 MB, p95 latency ≤ 300 ms on 1 vCPU.

## 6. Tabular models (phase 7, real data only)

- **Yield:** DCS district × season paddy statistics (data.gov.lk CSVs 1952–2012, plus later DCS tables), joined
  with NASA POWER / CHIRPS weather. LightGBM, trained on earlier years and tested on later ones, with quantile intervals.
- **Suitability:** a knowledge-based scoring model following FAO land evaluation. Crop requirement ranges come from
  FAO ECOCROP (GAEZ v4), with soil data from SoilGrids and climate from NASA POWER. Every score comes with its reasons.
