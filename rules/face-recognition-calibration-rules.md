# Face Recognition Calibration Rules

When working with `face_recognition` (dlib ResNet, 128-d embeddings):

## Confidence (→ `face-recognition-confidence` skill)
- **Use piecewise mapping.** `d ≤ 0.30 → 1.0`, `d ≥ 0.60 → 0.0`, linear between.
- **NEVER** use textbook `1 - d / tolerance`. Compresses strong matches into the mid-range.
- **Distance is not similarity.** Lower = closer.

## Enrollment (→ `face-recognition-enrollment` skill)
- Target **intra-class distance mean 0.25–0.40**. `<0.20` = overfit; `>0.45` = loose cloud.
- **Face coverage 60–75%** of frame height.
- **Blur threshold = 40** (Laplacian variance). NOT 80 — pale/fair skin scores 40–80 on sharp photos. 80 causes false rejections.
- 5–7 photos per person with pose + lighting variety.
- **Pre-pickle the enrollment** to `enrolled.pkl`. NEVER re-enroll from JPEG files in a latency-sensitive path (saves 8–10 s per startup).
- **Bad enrollment → "weak confidence".** Check enrollment BEFORE retuning thresholds. Custom threshold tuning is a workaround for poor enrollment.

## Persistence (→ `face-recognition-persistence` skill)
- Detectors miss 10–20% of frames. **Hold last state for ~0.8 s worth of misses** on the producer side.
- Persistence is the producer-side layer; actuator-side debounce is a SEPARATE layer — they compose.

## Install traps
- Python 3.14: pin `setuptools==75.8.0` for `face_recognition_models` / `pkg_resources`.
