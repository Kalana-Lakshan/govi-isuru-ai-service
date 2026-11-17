# Serving bundle v1.1

The trained model the service ships with. Built on Colab from run `mnv4_s_split` (MobileNetV4-S distilled from
DINOv2 + CLIP), then composed with `scripts/compose_bundle.py`:

* base: bundle v1 (Mahalanobis distance decides unknown diseases);
* tea: softmax-confidence unknown score from bundle v2 (unknown tea diseases caught 15 % -> 46 % on test);
* rice: LOW_CONFIDENCE threshold for answering 70 % of validation photos (7.2 % error on validation, 5.6 % on test).

| File | Contents |
|---|---|
| `model.onnx` | the network, FP32 (5.2 MB). INT8 (1.6 MB) changed ~6 % of disease answers on real test photos, so it is not shipped |
| `decision.json`, `prototypes_*.npz` | calibration: temperatures, thresholds, class prototypes (see `meta.composition`) |
| `classes.json` | disease names (en/si/ta), type, advice (from `configs/disease_info.yaml`) |
| `yield/`, `suitability/` | DCS paddy forecasts and history; FAO EcoCrop requirements and ERA5 district climate |
| `quality.json` | photo-quality thresholds |
| `bundle_report.json` | parity, latency and calibration report |

Test results: `RESULTS.md` on the project Drive. Replace this folder only with a bundle built from a newer run.
