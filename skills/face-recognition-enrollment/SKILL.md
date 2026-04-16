---
name: face-recognition-enrollment
description: Capture and validate high-quality face enrollments for dlib face_recognition. Covers pose diversity, face-coverage framing, blur rejection, and intra-class distance diagnostics. Use when building or refreshing a face enrollment dataset, diagnosing "recognition looks weak even with my photos", or deciding whether to retune thresholds vs retake photos.
---

# Face Recognition Enrollment

Enrollment quality is the #1 determinant of face recognition reliability. Most
"the model is bad" complaints trace to an enrolled cluster that's either too
tight (overfit to one pose/light) or too loose (sparse cloud). This skill
encodes how to take and validate good enrollment photos.

## What a good enrollment looks like (empirical)

Measured on `face_recognition` (dlib ResNet, 128-d embeddings):

| Metric | Target | Bad signal |
|---|---|---|
| Intra-class distance mean | **0.25 – 0.40** | `<0.20` = overfit single pose; `>0.45` = cloud too sparse |
| Intra-class distance max | `<0.50` | `>0.55` = one or more outlier photos to drop |
| Face coverage (face_h / frame_h) | **60 – 75%** | `<40%` = too far, loses detail after 0.25× downscale; `>80%` = too close, edge crops |
| Laplacian blur variance | **≥ 150** | `<100` = out of focus; at-threshold (80–100) = camera hunting focus |
| Photos per person | **5 – 7** | `<3` = brittle; `>10` = diminishing returns, outliers creep in |

## 6-pose capture sequence (recommended)

Take photos with **real variation** across these axes:

1. Frontal, neutral expression
2. Frontal, slight smile
3. Three-quarter left (head turned ~30° to the subject's left; eyes on camera)
4. Three-quarter right (~30° to the subject's right)
5. Frontal under brighter lighting
6. Frontal under dimmer lighting

Same subject, same camera, different angles and lighting. This gives the
embedding model a cloud that covers the typical live-camera variation.

## Framing rules (during capture)

- Face should fill **~70% of the frame height**. Typical webcam distance: 40–60 cm.
- **Eyes on camera**, not on the screen edge.
- **Sharp focus on the eyes** — they're the highest-information region for ResNet.
- No glasses unless worn consistently at runtime.
- No hat, no heavy beard change between photos.

## Pre-publish validation

```python
import face_recognition, numpy as np, glob
encs = []
for p in sorted(glob.glob("faces/<person>/*.jpg")):
    img = face_recognition.load_image_file(p)
    e = face_recognition.face_encodings(img)
    if e: encs.append(e[0])
A = np.stack(encs)
dists = [np.linalg.norm(A[i]-A[j])
         for i in range(len(A)) for j in range(i+1, len(A))]
mean, mx = float(np.mean(dists)), float(max(dists))
assert 0.25 <= mean <= 0.40, f"bad spread: mean={mean:.3f}"
assert mx < 0.50, f"outlier photo detected: max={mx:.3f}"
```

If validation fails: identify the outlier photo(s) by distance-to-centroid and
drop/replace them. Retake the bottom-quality photo with better pose/lighting.

## Per-file rejection at capture time

Enforce validation at the moment of capture — better than cleaning up after:

- Reject frames where **no face is detected** (subject not in view).
- Reject frames where **face_h < 400 px** on a 720p capture (coverage too low).
- Reject frames where **Laplacian variance of the face crop < threshold** (too blurry).
  - **Threshold = 40** (not 80). Pale/fair skin produces low Laplacian variance
    (40–80) on perfectly sharp photos because there are fewer skin-texture edges.
    A threshold of 80 causes false rejections on pale subjects. 40 catches
    genuinely blurry frames without penalizing skin tone.
- Re-pose and try again.

## Averaging encodings

Per person, compute the mean of all accepted encodings:

```python
enrolled[name] = np.mean(np.stack(vecs), axis=0)
```

## Pre-pickle the enrollment

```python
import pickle
pickle.dump(enrolled, open("faces/enrolled.pkl", "wb"))
```

On subsequent runs (and in sub-agent workers), load the pickle:

```python
enrolled = pickle.load(open("faces/enrolled.pkl", "rb"))
```

This saves 8–10 seconds of JPEG loading + face_encodings per startup. Critical
for sub-agent architectures where each worker would otherwise re-enroll from
scratch. **Never re-enroll from JPEG files in a latency-sensitive path.**

## When enrollment is good, don't tune the formula

Related skill: `face-recognition-confidence` ships with thresholds
(`strong=0.30`, `reject=0.60`) calibrated to the library's documented
distribution. If your runtime distances land in 0.25–0.40 on strong matches,
those defaults work as designed. **If you're tempted to raise `strong` to 0.40
or 0.45, check enrollment first.** Custom threshold tuning is usually a
workaround for a poor enrollment — fix the root cause.

## Reference implementation

[`scripts/enroll.py`](../../scripts/enroll.py) — `validate_capture()`, `compute_enrollment()`, `save_enrollment()`, `load_enrollment()`. Import directly.

## Python 3.14 install trap (same note as the confidence skill)

`face_recognition_models` still depends on `pkg_resources`, which setuptools
removed in 82+. Pin `setuptools==75.8.0` before `pip install face_recognition`.
