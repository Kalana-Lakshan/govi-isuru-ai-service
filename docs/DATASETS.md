# Datasets

The single source of truth is [`configs/datasets.yaml`](../configs/datasets.yaml): every source's
licence, citation, label mapping and paper eligibility. This page explains the choices.

## Rice

| Source | Images | Classes | Origin | Notes |
|---|---|---|---|---|
| Paddy Doctor | 10,407 | 10 | IN, field | Main source; adds BLS, panicle blight, dead heart, downy mildew, tungro |
| RLBF (Hasan 2023) | 1,701 (raw) | 8 | BD, field | **Is the v1 Govi Isuru rice dataset** |
| Sethy 2020 | 5,932 | 4 | IN, field | |
| Rice disease Bangladesh | 2,769 (raw) | 8 → 5 used | BD, field | `Insect`, `Rice`, `Rice Stripes` dropped (ambiguous) |
| Dhan-Shomadhan | 1,106 | 5 | BD | Label typos merged in label_map |
| Shakeel (Zenodo) | 1,914 | 3 | PK | Perfectly balanced; dedupe will tell if augmented |
| UCI rice | 120 | 3 | IN | `leaf_smut` → unknown-disease evaluation |

## Tea

| Source | Images | Classes | Origin | Notes |
|---|---|---|---|---|
| CS-D Assam (Mendeley original) | ~9k | 6 | IN, field | **Blister blight** source; the HF mirror looks augmented, so not used |
| teaLeafBD | 5,278 | 7 | BD, field | |
| Balasundaram | 5,867 | 6 | IN | CC BY-NC; low-resolution files |
| Bormon 2025 | 3,960 | 4 → 3 used | BD | `Tea red scab` → unknown eval |
| Chowdhury 2025 | 2,500 | 4 → 3 used | BD | `Blight` dropped (mixed) |
| Tea sickness (Kimutai) | 885 | 8 | KE | anthracnose, bird's eye spot, white spot → unknown eval |

## Chili

| Source | Images | Classes | Origin | Notes |
|---|---|---|---|---|
| COLD | 532 (raw) | 5 | IN, field | |
| Kim 2025 | 4,102 | 3 | KR, greenhouse | BBWV2 and flood stress → unknown eval |
| 3 Mendeley sets | — | — | BD / IN | Disabled until downloaded and folder names checked |
| v1 legacy chili | 3,264 | 4 | unknown | `use_in_paper: false` |

## Gate negatives

- **other_plant:** PlantVillage (minus bell pepper, which is the same species as chili), mango, banana, coffee,
  groundnut, eggplant, betel, sugarcane, maize, and rice-field weeds (hard negatives). Each is capped per class.
- **non_plant:** COCO val2017 with every image containing plants, vegetables, fruit or vases removed, plus the PlantVillage
  `Background_without_leaves` class.

## Known gaps / to-do

- No Sri Lankan field photos are available. This is stated as a limitation; a small self-collected external test set
  would strengthen the paper a lot.
- Licences marked "verify" must be checked before submission.
- Sinhala and Tamil class names must be filled in by an agronomist (`configs/taxonomy.yaml`).
- Red rust and algal leaf spot (both *Cephaleuros*) are separate classes until an expert reviews them.
