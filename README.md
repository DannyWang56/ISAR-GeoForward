## Contents
- `data/Calipso/image/`: 500 512 × 512 RGB images named `1.png`–`500.png`.
- `data/Calipso/cameras.json`: matching RLOS and camera parameters for all
  500 images, in the same order.
- `data/Calipso/nas_full_points.npy`: 8,192 world-frame reference points.
- `code/`: `point_heads.py` (point-cloud heads), `renderer.py` (ISAR forward
  renderer), `metrics.py` (reconstruction metrics), and `verify_images.py`
  (image and filename checks).
