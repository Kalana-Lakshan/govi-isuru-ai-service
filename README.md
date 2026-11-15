# govi-isuru-ai-service

AI service for **Govi Isuru**: crop-health recognition for **rice, tea and chili**, plus yield and
crop-suitability models. The service is built to run on a small CPU container (target: under 300 MB RAM,
one forward pass per photo) and is designed to support a research paper.

What the image model does in one forward pass:

| Output | Meaning |
|---|---|
| `NOT_A_PLANT` | photo is not a plant |
| `UNSUPPORTED_PLANT` | a plant, but not rice / tea / chili |
| `WRONG_CROP` | e.g. a chili leaf sent as paddy; the response names the detected crop |
| `KNOWN_DISEASE` | class, calibrated confidence, top-3, CAM heatmap |
| `UNKNOWN_DISEASE` | a leaf of a known crop that doesn't match any known disease → embedding returned so an agricultural officer can verify and name it; once registered it is recognised immediately |
| `LOW_CONFIDENCE` | abstain and suggest consulting an officer |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design, [docs/DATASETS.md](docs/DATASETS.md) for
the data, and [docs/RESEARCH.md](docs/RESEARCH.md) for the literature it is based on.

## Repository layout

```
configs/        taxonomy.yaml (unified label space) · datasets.yaml (every image source, licence, citation)
src/govi_ai/    shared package: taxonomy, data pipeline, metrics, open-set scores, models, training
scripts/        build_dataset.py · pack_dataset.py · eval_legacy.py · extract_features.py · run_probes.py
                train_student.py · eval_student.py · build_bundle.py · fetch_paddy_stats.py
notebooks/      Colab notebooks, one per phase
service/        FastAPI + ONNX Runtime serving image (phase 6)
tests/          unit + end-to-end tests (pytest)
docs/           architecture, datasets, research notes
```

No datasets and no weights are committed. Datasets are rebuilt from `configs/datasets.yaml`;
weights will be published to the Hugging Face Hub and pulled when the service image is built.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Data: registry, harmonisation, dedupe, leakage-safe splits, legacy baseline | **code ready, run on Colab** |
| 2 | Teacher: frozen foundation-model features (DINOv2 / CLIP), linear & fusion probes | **code ready** (notebook 02) |
| 3 | Student: hierarchical open-set CNN, knowledge distillation | **code ready** (notebook 03) |
| 4 | Calibration, prototype library, decision logic | **code ready** (`build_bundle.py`) |
| 5 | ONNX INT8 export + parity check | **code ready** (`build_bundle.py`) |
| 6 | FastAPI service, Docker, OpenAPI contract | **code ready** (`service/`) |
| 7 | Yield (real DCS data) + suitability (FAO ECOCROP / GAEZ knowledge model) | DCS data fetcher ready |
| 8 | Evaluations, ablations, paper | |

## Quick start

Local (tests and small builds):

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install -e ".[data,dev]"
pytest
python scripts/build_dataset.py --list
python scripts/build_dataset.py --data-root data --sources uci_rice tea_sickness --limit 50
```

Full build and baseline: open `notebooks/01_build_dataset.ipynb` in Google Colab.

## Data rules

- **Real photographs only.** No synthetic or generated data anywhere, including the tabular models.
- Never use a source's *augmented* release (it leaks near-duplicates across splits).
- Every near-duplicate cluster (found with flip/rotation-invariant pHash) is one group. Splits are made by group,
  and clusters with conflicting labels are dropped.
- Sources with unclear provenance (`use_in_paper: false`, e.g. the v1 legacy sets) may help train the deployed model,
  but they never count towards a reported number.
