"""Crop suitability: FAO ECOCROP requirements + the EcoCrop limiting-factor model (numpy-free).

Replaces the v1 Random Forest that was trained on rule-generated synthetic rows.

Model (Hijmans et al., DIVA-GIS / dismo ``ecocrop``; Ramirez-Villegas et al. 2013, Food Security):
for every numeric factor the score is a trapezoid — 0 outside the absolute [min, max] range, 1 inside
the optimal [opt_min, opt_max] range, linear in between — and the overall score is the *minimum* over
factors (Liebig's law of the minimum), so the answer always names the limiting factor.
Categorical factors (drainage, texture) score 1 if in the optimal set, 0.5 if only in the absolute
set, 0 otherwise.

Data: crop parameters from the FAO ECOCROP database (via OpenCLIM/ecocrop's EcoCrop_DB.csv);
district climate defaults from NASA POWER climatology at district reference points.

Assumptions (stated in every response):
  * rainfall is compared over the growing cycle for annual crops (rice, chili: the chosen season's
    rainfall) and annually for perennials (tea);
  * irrigation available => the rainfall factor is not limiting (scored 1), and for crops that need
    waterlogged soil (lowland rice) drainage is not limiting either (bunded, flooded paddies);
  * suitability classes follow the FAO land-evaluation convention S1/S2/S3/N with cut-offs
    0.8 / 0.6 / 0.4 on the EcoCrop score.
"""

from __future__ import annotations

from dataclasses import dataclass

NUMERIC = {
    "temperature_c": ("TMIN", "TOPMN", "TOPMX", "TMAX"),
    "rainfall_mm": ("RMIN", "ROPMN", "ROPMX", "RMAX"),
    "soil_ph": ("PHMIN", "PHOPMN", "PHOPMX", "PHMAX"),
}
CLASSES = [(0.8, "S1", "highly suitable"), (0.6, "S2", "moderately suitable"),
           (0.4, "S3", "marginally suitable"), (0.0, "N", "not suitable")]
# Our crops -> ECOCROP scientific names; perennial crops use annual rainfall.
CROPS = {"rice": ("Oryza sativa", False), "tea": ("Camellia sinensis", True), "chili": ("Capsicum annuum", False)}
DRAINAGE = {  # app vocabulary -> ECOCROP drainage wording
    "poor": "poorly", "moderate": "well", "good": "well", "excessive": "excessive",
}
TEXTURE = {"clay": "heavy", "loam": "medium", "sandy": "light", "organic": "organic",
           "sandy_loam": "light", "clay_loam": "medium"}


def trapezoid(x: float, lo: float, opt_lo: float, opt_hi: float, hi: float) -> float:
    if x <= lo or x >= hi:
        return 0.0
    if opt_lo <= x <= opt_hi:
        return 1.0
    if x < opt_lo:
        return (x - lo) / (opt_lo - lo) if opt_lo > lo else 1.0
    return (hi - x) / (hi - opt_hi) if hi > opt_hi else 1.0


def _matches(value: str, spec: str) -> bool:
    return bool(spec) and (value in spec.lower() or "wide" in spec.lower())


@dataclass
class CropRequirements:
    crop: str
    scientific_name: str
    perennial: bool
    params: dict[str, float]
    drainage_opt: str = ""
    drainage_abs: str = ""
    texture_opt: str = ""
    texture_abs: str = ""
    max_altitude_m: float | None = None

    @classmethod
    def from_ecocrop_row(cls, crop: str, row: dict) -> "CropRequirements":
        sci, perennial = CROPS[crop]
        keys = {k for v in NUMERIC.values() for k in v}
        return cls(crop=crop, scientific_name=sci, perennial=perennial,
                   params={k: float(row[k]) for k in keys},
                   drainage_opt=str(row.get("DRA") or ""), drainage_abs=str(row.get("DRAR") or ""),
                   texture_opt=str(row.get("TEXT") or ""), texture_abs=str(row.get("TEXTR") or ""),
                   max_altitude_m=float(row["ALTMX"]) if row.get("ALTMX") not in (None, "") else None)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def score_crop(req: CropRequirements, *, temperature_c: float, rainfall_mm: float, soil_ph: float,
               irrigated: bool = False, drainage: str | None = None, texture: str | None = None,
               altitude_m: float | None = None) -> dict:
    p = req.params
    factors: dict[str, float] = {
        "temperature_c": trapezoid(temperature_c, *(p[k] for k in NUMERIC["temperature_c"])),
        "rainfall_mm": 1.0 if irrigated else trapezoid(rainfall_mm, *(p[k] for k in NUMERIC["rainfall_mm"])),
        "soil_ph": trapezoid(soil_ph, *(p[k] for k in NUMERIC["soil_ph"])),
    }
    if drainage:
        d = DRAINAGE.get(drainage, drainage)
        if irrigated and "poorly" in req.drainage_opt.lower():
            # Crops that need waterlogging (lowland rice): bunded, irrigated fields create it.
            factors["drainage"] = 1.0
        else:
            factors["drainage"] = 1.0 if _matches(d, req.drainage_opt) else 0.5 if _matches(d, req.drainage_abs) else 0.0
    if texture:
        t = TEXTURE.get(texture, texture)
        factors["texture"] = 1.0 if _matches(t, req.texture_opt) else 0.5 if _matches(t, req.texture_abs) else 0.0
    if altitude_m is not None and req.max_altitude_m:
        factors["altitude_m"] = 1.0 if altitude_m <= req.max_altitude_m else 0.0
    limiting = min(factors, key=factors.get)
    score = factors[limiting]
    cls = next((c, n) for t, c, n in CLASSES if score >= t)
    return {"crop": req.crop, "score": round(score, 3), "class": cls[0], "class_name": cls[1],
            "limiting_factor": limiting if score < 1 else None,
            "factors": {k: round(v, 3) for k, v in factors.items()},
            "optimal_ranges": {
                "temperature_c": [p["TOPMN"], p["TOPMX"]],
                "rainfall_mm": [p["ROPMN"], p["ROPMX"]],
                "soil_ph": [p["PHOPMN"], p["PHOPMX"]]}}


def recommend(reqs: dict[str, CropRequirements], *, season_rainfall_mm: float, annual_rainfall_mm: float,
              temperature_c: float, soil_ph: float, irrigated: bool = False, drainage: str | None = None,
              texture: str | None = None, altitude_m: float | None = None) -> list[dict]:
    """Score every crop and rank them (best first)."""
    out = []
    for name, req in reqs.items():
        rain = annual_rainfall_mm if req.perennial else season_rainfall_mm
        r = score_crop(req, temperature_c=temperature_c, rainfall_mm=rain, soil_ph=soil_ph,
                       irrigated=irrigated and not req.perennial, drainage=drainage, texture=texture,
                       altitude_m=altitude_m)
        r["rainfall_basis"] = "annual" if req.perennial else "season"
        out.append(r)
    return sorted(out, key=lambda r: -r["score"])
