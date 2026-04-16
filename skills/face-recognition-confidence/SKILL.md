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

## When defaults don't fit: check enrollment before tuning

If your runtime distances consistently land above 0.40 on what should be strong
matches, the formula isn't the problem — **your enrollment is**. Enrollment
taken at different framing/lighting/camera than runtime produces a loose cloud,
and distances inflate. See the `face-recognition-enrollment` skill in this
plugin for a quality checklist and diagnostic (intra-class distance target
0.25–0.40 mean, face coverage 60–75%, Laplacian blur ≥ 150).

Do not raise `strong` to 0.40 or 0.45 as a workaround for bad enrollment. You
will mask the real problem and break across subjects.

## Python 3.14 install trap

`face_recognition_models` still uses `pkg_resources`, which setuptools removed in 82+.
On Python 3.14, pin:

```
setuptools==75.8.0
```

Do this in the project's requirements file before `pip install face_recognition`.
Otherwise the import will crash with `ModuleNotFoundError: No module named 'pkg_resources'`.

## How to act

1. Prefer [`scripts/confidence.py`](../../scripts/confidence.py) over re-deriving the mapping.
2. If asked about distances, remember that lower = closer (not a similarity score).
3. If asked why recognition "looks weak", check which formula is producing the UI number.
4. After integrating the formula, run a quick sanity check: call `confidence(0.38)` and verify the result is approximately `0.73`. If it is not, the mapping is misconfigured or the wrong formula is in use.

See the rule file `face-recognition-calibration-rules` for a quick reminder card.
