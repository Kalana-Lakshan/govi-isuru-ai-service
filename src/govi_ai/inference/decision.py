"""Turn raw model outputs into one of six answers (numpy only).

    NOT_A_PLANT        the gate says non-plant
    UNSUPPORTED_PLANT  a plant, but not rice / tea / chili
    WRONG_CROP         a supported crop other than the one the farmer selected (detected crop returned)
    UNKNOWN_DISEASE    a leaf of the crop, but far from every known disease prototype -> officer review
    LOW_CONFIDENCE     a known-looking leaf whose class is uncertain -> suggest an officer
    KNOWN_DISEASE      class + calibrated confidence + top-3

Thresholds and prototypes come from ``scripts/calibrate.py`` (fitted on validation data only).
The prototype library can grow at runtime: ``add_prototype`` registers an officer-verified new
disease from a few embeddings, and it is recognised immediately, without retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NON_PLANT, OTHER_PLANT = "non_plant", "other_plant"


def _softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class CropCalibration:
    classes: list[str]
    temperature: float
    means: np.ndarray           # K×D class prototypes (trained classes first, then officer-added)
    precision: np.ndarray       # D×D shared inverse covariance
    unknown_threshold: float    # Mahalanobis score below this -> UNKNOWN_DISEASE
    low_conf_threshold: float   # calibrated max prob below this -> LOW_CONFIDENCE
    head_weights: np.ndarray | None = None  # K×D normalised cosine weights (for CAM)
    # Which score decides UNKNOWN_DISEASE for this crop (chosen per crop at calibration time):
    #   "mahalanobis" (default, threshold = unknown_threshold), "msp" or "fused" (z-scored sum of both).
    # `unknown_threshold` always stays the Mahalanobis threshold: it decides officer-prototype matches.
    novelty: dict = field(default_factory=lambda: {"score": "mahalanobis"})
    proto_labels: list[str] = field(default_factory=list)  # one label per row of `means`
    proto_origin: list[str] = field(default_factory=list)  # "trained" | "officer"

    def __post_init__(self) -> None:
        if not self.proto_labels:
            self.proto_labels = list(self.classes)
            self.proto_origin = ["trained"] * len(self.classes)

    def distances(self, emb: np.ndarray) -> np.ndarray:
        d = self.means - emb[None, :]
        return np.einsum("kd,de,ke->k", d, self.precision, d)

    def novelty_score(self, maha_score: float, msp: float) -> tuple[float, float]:
        """(score, threshold) for the configured novelty score; low score = unknown."""
        return novelty_value(self.novelty, maha_score, msp, self.unknown_threshold)


def novelty_value(cfg: dict, maha_score, msp, maha_threshold: float):
    """Vectorised helper shared by serving and calibration. Returns (score, threshold)."""
    kind = cfg.get("score", "mahalanobis")
    if kind == "mahalanobis":
        return maha_score, maha_threshold
    if kind == "msp":
        return msp, cfg["threshold"]
    if kind == "fused":
        z = (np.asarray(maha_score) - cfg["maha_mu"]) / cfg["maha_sd"] + (np.asarray(msp) - cfg["msp_mu"]) / cfg["msp_sd"]
        return z, cfg["threshold"]
    raise ValueError(f"unknown novelty score {kind!r}")


@dataclass
class DecisionBundle:
    gate_classes: list[str]
    gate_temperature: float
    gate_min_conf: float
    crops: dict[str, CropCalibration]
    version: str = "dev"
    meta: dict = field(default_factory=dict)

    # -------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        cfg = {"version": self.version, "gate_classes": self.gate_classes, "gate_temperature": self.gate_temperature,
               "gate_min_conf": self.gate_min_conf, "meta": self.meta, "crops": {}}
        for name, c in self.crops.items():
            cfg["crops"][name] = {"classes": c.classes, "temperature": c.temperature,
                                  "unknown_threshold": c.unknown_threshold, "low_conf_threshold": c.low_conf_threshold,
                                  "proto_labels": c.proto_labels, "proto_origin": c.proto_origin,
                                  "novelty": c.novelty}
            arrays = {"means": c.means.astype(np.float32), "precision": c.precision.astype(np.float32)}
            if c.head_weights is not None:
                arrays["head_weights"] = c.head_weights.astype(np.float32)
            np.savez(d / f"prototypes_{name}.npz", **arrays)
        (d / "decision.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "DecisionBundle":
        d = Path(directory)
        cfg = json.loads((d / "decision.json").read_text(encoding="utf-8"))
        crops = {}
        for name, c in cfg["crops"].items():
            z = np.load(d / f"prototypes_{name}.npz")
            crops[name] = CropCalibration(
                classes=c["classes"], temperature=c["temperature"], means=z["means"].astype(np.float64),
                precision=z["precision"].astype(np.float64), unknown_threshold=c["unknown_threshold"],
                low_conf_threshold=c["low_conf_threshold"],
                head_weights=z["head_weights"] if "head_weights" in z else None,
                proto_labels=c.get("proto_labels", []), proto_origin=c.get("proto_origin", []),
                novelty=c.get("novelty", {"score": "mahalanobis"}))
        return cls(cfg["gate_classes"], cfg["gate_temperature"], cfg["gate_min_conf"], crops,
                   cfg.get("version", "dev"), cfg.get("meta", {}))

    # -------------------------------------------------------------- prototype library
    def add_prototype(self, crop: str, label: str, embeddings: np.ndarray) -> None:
        """Register an officer-verified disease (a few embeddings of confirmed cases)."""
        c = self.crops[crop]
        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        c.means = np.vstack([c.means, emb.mean(0, keepdims=True)])
        c.proto_labels.append(label)
        c.proto_origin.append("officer")

    # -------------------------------------------------------------- decision
    def decide(self, gate_logits: np.ndarray, disease_logits: dict[str, np.ndarray], emb: np.ndarray,
               requested_crop: str | None = None) -> dict:
        p_gate = _softmax(gate_logits, self.gate_temperature)
        g = int(p_gate.argmax())
        detected = self.gate_classes[g]
        gate_info = {"probabilities": {c: round(float(p), 4) for c, p in zip(self.gate_classes, p_gate)}}

        if detected == NON_PLANT:
            return {"status": "NOT_A_PLANT", "confidence": float(p_gate[g]), "gate": gate_info}
        if detected == OTHER_PLANT:
            return {"status": "UNSUPPORTED_PLANT", "confidence": float(p_gate[g]), "gate": gate_info}
        if p_gate[g] < self.gate_min_conf:
            return {"status": "LOW_CONFIDENCE", "reason": "crop_uncertain", "detected_crop": detected,
                    "confidence": float(p_gate[g]), "gate": gate_info}
        if requested_crop and requested_crop != detected:
            return {"status": "WRONG_CROP", "requested_crop": requested_crop, "detected_crop": detected,
                    "confidence": float(p_gate[g]), "gate": gate_info}

        cal = self.crops.get(detected)
        if cal is None:  # crop head without calibration data in this bundle
            return {"status": "LOW_CONFIDENCE", "reason": "crop_not_calibrated", "detected_crop": detected,
                    "confidence": float(p_gate[g]), "gate": gate_info}
        probs = _softmax(disease_logits[detected], cal.temperature)
        dist = cal.distances(np.asarray(emb, dtype=np.float64))
        nearest = int(dist.argmin())
        maha_score = float(-dist[nearest])
        top = np.argsort(-probs)[:3]
        top3 = [{"class": cal.classes[i], "probability": round(float(probs[i]), 4)} for i in top]
        score, threshold = cal.novelty_score(maha_score, float(probs[top[0]]))
        base = {"crop": detected, "gate": gate_info, "top3": top3,
                "novelty": {"score": cal.novelty.get("score", "mahalanobis"), "value": float(score),
                            "threshold": float(threshold), "mahalanobis_score": maha_score,
                            "nearest_prototype": cal.proto_labels[nearest]}}

        if cal.proto_origin[nearest] == "officer" and maha_score >= cal.unknown_threshold:
            # An officer-registered disease the network head has never been trained on.
            return {"status": "KNOWN_DISEASE", "class": cal.proto_labels[nearest], "source": "officer_prototype",
                    "confidence": None, **base}
        if score < threshold:
            return {"status": "UNKNOWN_DISEASE", **base}
        if probs[top[0]] < cal.low_conf_threshold:
            return {"status": "LOW_CONFIDENCE", "reason": "disease_uncertain", **base}
        return {"status": "KNOWN_DISEASE", "class": cal.classes[top[0]], "source": "model",
                "confidence": round(float(probs[top[0]]), 4), **base}
