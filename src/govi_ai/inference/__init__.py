"""Runtime inference: preprocessing, ONNX engine and decision logic.

Dependencies here must stay within the slim serving image: numpy, pillow, onnxruntime.
No torch, no tensorflow, no pandas, no scikit-learn.
"""
