---
name: face-recognition-persistence
description: Producer-side face persistence that absorbs transient detection dropouts. Keeps the last observed confidence across N consecutive no-face frames instead of flipping to "nobody here" on every single missed frame. Use when you see the bar/state flickering even though the subject is plainly in frame, or when detection + recognition hands off to a downstream actuator that wants steady state.
---

# Face Recognition Persistence

Haar/HOG/CNN face detectors routinely miss 10–20% of frames even with a subject
clearly in view. Each miss propagates to any downstream state ("no face →
state=None → bulb dark → face returns → bulb on"). The result is visible
flicker that has nothing to do with the subject moving.

This skill encodes the **producer-side** persistence buffer that sits BETWEEN
the detector and whatever downstream consumer cares about "is a face present
right now". It complements — does not replace — any actuator-side stability
filter (see `rate-limited-iot-debounce` / `iot-actuator-patterns`).

## The pattern

- Track a `miss_streak` counter of consecutive no-face frames.
- On detection, reset the streak and record the latest result.
- On no-detection:
  - If `miss_streak < FACE_PERSIST_MISSES`: keep the last observed state.
  - If `miss_streak ≥ FACE_PERSIST_MISSES`: commit "no face".

```python
FACE_PERSIST_MISSES = 8   # ~0.9s at frame_skip=3 over 30fps

last_seen_conf = 0.0
miss_streak = 0

# per-frame:
if encs:
    miss_streak = 0
    last_seen_conf = compute_confidence(...)
    effective_conf = last_seen_conf
else:
    miss_streak += 1
    if miss_streak < FACE_PERSIST_MISSES:
        effective_conf = last_seen_conf      # persist
    else:
        effective_conf = 0.0
        last_seen_conf = 0.0

downstream.set(effective_conf)
```

## Picking `FACE_PERSIST_MISSES`

Pick the number of consecutive misses that implies "actually gone" rather
than "brief occlusion":

| Inference rate | Recommended persist | Implied wall-time |
|---|---|---|
| 30 fps no skip | 24 | 0.8s |
| 30 fps, skip=3 (effective 10 Hz) | 8 | 0.8s |
| 10 fps inference | 8 | 0.8s |
| 5 fps inference | 4 | 0.8s |

Aim for **~0.8s of persistence** — long enough to ride over detection dropouts
and brief occlusions, short enough that someone actually leaving produces a
timely "gone" signal.

## Why this is NOT the debounce-controller's job

The actuator-side debounce in `iot-actuator-patterns/debounce-controller` has a
stability filter that requires the target to hold for N consecutive ticks. But
that filter sees whatever the producer emits — if the producer flips between
`last_state` and `None` on every frame, the stability filter still has to
choose between two values and will never commit.

Persistence runs **upstream** of stability: the producer emits a steady
"last-known-state" across dropouts, and the actuator debounce suppresses the
remaining real transitions. Two filters, two layers.

## How to act

- If you see flicker downstream of face detection, add the persistence buffer
  at the point where you convert "face detected" into whatever state feeds the
  actuator.
- Use `FACE_PERSIST_MISSES` tuned to ~0.8s of your effective inference rate.
- Keep the downstream stability filter (actuator debounce) — they compose.

## Related skills

- `face-recognition-confidence` — map distance to piecewise confidence.
- `face-recognition-enrollment` — make sure the low-level detection is actually
  strong. No persistence buffer saves you from a poor enrollment.
- `iot-actuator-patterns/debounce-controller` — actuator-side stability on top.
