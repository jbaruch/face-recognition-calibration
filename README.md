# face-recognition-calibration

A [Tessl](https://tessl.io) tile that encodes a **perceptually-correct confidence
formula** for the `face_recognition` (dlib) library, plus enrollment best practices
and the Python 3.14 install trap.

## What this tile provides

| Kind | Name | Purpose |
|---|---|---|
| Skill | `face-recognition-confidence` | Piecewise distance → confidence mapping, enrollment workflow, validation checks. |
| Rule  | `face-recognition-calibration-rules` | Concise in-context reminder card. |
| Script | `scripts/confidence.py` | Reference implementation of the mapping and enrollment helpers. |

## Why it exists

The textbook `1 - distance / tolerance` mapping is technically correct and empirically
useless — a strong match at `d = 0.38` shows up as `0.37` on a UI meter and the demo
looks broken. This tile uses a piecewise mapping that reflects how operators actually
read the numbers:

```
d <= 0.30  ->  1.0   (strong match)
d >= 0.60  ->  0.0   (reject)
else       ->  (0.60 - d) / 0.30
```

Same `d = 0.38` now scores `0.73`. The meter matches the vibe.

It also documents the Python 3.14 install trap: `face_recognition_models` depends on
`pkg_resources`, which setuptools removed in 82+. Pin `setuptools==75.8.0`.

## Install

```bash
tessl install jbaruch/face-recognition-calibration
```

Or from this repo:

```bash
tessl install github:jbaruch/face-recognition-calibration
```

## Usage (quick)

```python
from scripts.confidence import confidence, best_match, load_enrollment

enrolled = load_enrollment("faces.pkl")
name, distance, conf = best_match(live_encoding, enrolled)
print(f"{name} d={distance:.2f} conf={conf:.2f}")
```

See `skills/face-recognition-confidence/SKILL.md` for the full guidance and
`rules/face-recognition-calibration-rules.md` for the short version.

## License

MIT — see `LICENSE`.
