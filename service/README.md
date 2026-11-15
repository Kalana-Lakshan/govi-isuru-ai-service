# service/

The deployable FastAPI + ONNX Runtime image (phase 6). It will contain only inference code and
depend only on `govi-ai[service]` (no torch or TensorFlow). Its API contract will be published as OpenAPI
for the Govi Isuru backend. See ../docs/ARCHITECTURE.md §5.
