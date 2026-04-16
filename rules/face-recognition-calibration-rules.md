# Face Recognition Calibration Rules

When working with `face_recognition` (dlib ResNet, 128-d embeddings):

## Confidence (→ `face-recognition-confidence` skill)
- **Use piecewise mapping.** `d ≤ 0.30 → 1.0`, `d ≥ 0.60 → 0.0`, linear between.
- **Avoid** textbook `1 - d / tolerance`. Compresses strong matches into the mid-range.
- **Distance is not similarity.** Lower = closer.

## Enrollment quality (→ `face-recognition-enrollment` skill)
- Target **intra-class distance mean 0.25–0.40**. `<0.20` = overfit; `>0.45` = loose cloud.
- **Face coverage 60–75%** of frame height.
- **Laplacian blur ≥ 150**.
- 5–7 photos per person with pose + lighting variety.
- **Bad enrollment manifests as "weak confidence".** Check enrollment before retuning thresholds.

## Persistence (→ `face-recognition-persistence` skill)
- Detectors miss 10–20% of frames. **Hold last state for ~0.8 s worth of misses** on the producer side.
- Persistence is the producer-side layer; actuator-side debounce is separate — they compose.

## Install traps
- Python 3.14: pin `setuptools==75.8.0` for `face_recognition_models` / `pkg_resources`.

Full references: `skills/face-recognition-confidence/SKILL.md`, `skills/face-recognition-enrollment/SKILL.md`, `skills/face-recognition-persistence/SKILL.md`.
