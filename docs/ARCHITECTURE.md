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

- **Yield (done):** official DCS district × season statistics, 1978/79 Maha – 2026 (statistics.gov.lk Excel series;
  data.gov.lk's certificate had expired). Rolling-origin backtest 2010–2026 (795 district-seasons):
  5-year climatology MAE 436 kg/ha (12.0 %), gradient-boosted model on sowing-time features 446 (12.2 %),
  last-year baseline 493 (13.5 %). Without in-season weather nothing beats climatology, so the service returns
  the 5-year mean with an empirically calibrated 80 % interval (79.5 % backtest coverage). Next step: in-season
  weather (NASA POWER / CHIRPS) for mid-season updates.
- **Suitability (done):** the published EcoCrop limiting-factor model (trapezoid score per factor, minimum
  over factors) with the FAO ECOCROP requirements of *Oryza sativa*, *Camellia sinensis* and *Capsicum annuum*.
  District climate defaults (mean temperature, annual / Maha / Yala rainfall, elevation) come from ERA5 1995-2024
  climatology via Open-Meteo (NASA POWER's coarse grid was rejected: it put Nuwara Eliya at 22.7 C and Hambantota
  at 2,225 mm/yr). CHIRPS rainfall (0.05 deg) is the planned upgrade for dry-zone accuracy.
  Every answer names its limiting factor and the optimal ranges. No training and no synthetic data.
