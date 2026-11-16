"""ONNX export and INT8 quantisation of the student (used by build_bundle.py and make_dev_bundle.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def exportable(model, crops: list[str]):
    """Wrap the student so ONNX gets a flat tuple of named outputs."""
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, image):
            out = self.m(image)
            return (out["gate"], *[out["disease"][c] for c in crops], out["emb"], out["fmap"])

    return Wrapper().eval()


def output_names(crops: list[str]) -> list[str]:
    return ["gate", *[f"disease_{c}" for c in crops], "emb", "fmap"]


def export_onnx(model, crops: list[str], img_size: int, path: Path) -> list[str]:
    import torch

    names = output_names(crops)
    torch.onnx.export(exportable(model, crops), torch.zeros(1, 3, img_size, img_size), str(path),
                      input_names=["image"], output_names=names, opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "batch"}, **{n: {0: "batch"} for n in names}})
    return names


def quantize_int8(fp32: Path, int8: Path, calib: list[np.ndarray]) -> None:
    """Static INT8 (QDQ, per-channel weights) calibrated on real validation images."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": x} for x in calib])

        def get_next(self):
            return next(self.it, None)

    pre = fp32.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
