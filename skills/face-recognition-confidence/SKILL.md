---
name: face-recognition-confidence
description: Compute perceptually-correct confidence from dlib face_recognition distances using piecewise mapping (d at most 0.3 maps to 1.0, d at least 0.6 maps to 0.0, linear between). Includes enrollment averaging and the setuptools==75.8.0 pin. Use when mapping face_recognition distance to a user-facing confidence score or diagnosing weak recognition results.
---

# Face Recognition Confidence Calibration

Use this skill any time you are mapping a `face_recognition` distance to a user-facing
confidence score. The textbook formula looks right on paper and looks broken on stage.

## Typical distance ranges (dlib ResNet, 128-d embeddings)

| Match quality | Distance |
|---|---|
| Strong match | 0.30 – 0.40 |
| Borderline   | 0.40 – 0.55 |
| Reject       | > 0.60 |

Library default `tolerance = 0.6`.

## The formula we use

```python
def confidence(distance: float) -> float:
    if distance <= 0.30:
        return 1.0
    if distance >= 0.60:
        return 0.0
    return (0.60 - distance) / 0.30
```

A strong match at d=0.38 gives 0.73 — feels right on a meter. The naive
`1 - distance/tolerance` at the same distance gives 0.37 and the demo looks broken.

## Enrollment

- 3–7 well-lit photos per person.
- Compute 128-d encoding for each; **average** them into a single enrolled vector.
- Pickle the resulting dict `{name: np.ndarray(128)}` so the stage machine doesn't
  recompute on every boot.
- **Validate after encoding**: confirm each encoding array has shape `(128,)` and
  contains no `NaN` values (`np.isnan(encoding).any()` should be `False`). If
  `face_recognition.face_encodings()` returns an empty list for a photo, that image
  had no detectable face — discard it and substitute another.
- **Validate after averaging**: check that the stored vector norm is in the range
  0.9 – 1.1 (`np.linalg.norm(avg_encoding)`). A vector far outside this range
  indicates a bad source encoding slipped through.

## Python 3.14 install trap

`face_recognition_models` still uses `pkg_resources`, which setuptools removed in 82+.
On Python 3.14, pin:

```
setuptools==75.8.0
```

Do this in the project's requirements file before `pip install face_recognition`.
Otherwise the import will crash with `ModuleNotFoundError: No module named 'pkg_resources'`.

## How to act

1. Prefer `scripts/confidence.py` over re-deriving the mapping.
2. If asked about distances, remember that lower = closer (not a similarity score).
3. If asked why recognition "looks weak", check which formula is producing the UI number.
4. After integrating the formula, run a quick sanity check: call `confidence(0.38)` and verify the result is approximately `0.73`. If it is not, the mapping is misconfigured or the wrong formula is in use.

See the rule file `face-recognition-calibration-rules` for a quick reminder card.
