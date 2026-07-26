Place model weights here:
  yolov8n.pt          - auto-downloaded by Ultralytics on first run if missing
  navigation_best.pt  - produced by training/train.py (custom hazard model)
  best.tflite         - produced by training/export_tflite.py (Phase-2 Android)

MiDaS_small is downloaded automatically via torch.hub into your torch cache
(~/.cache/torch/hub) the first time depth_estimator.py runs — no manual file
needed for it.
