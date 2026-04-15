# Face Recognition Calibration Rules

When mapping `face_recognition` distances to UI confidence:

- **Use piecewise mapping, not linear-to-tolerance.**
  - `d <= 0.30` → `1.0`
  - `d >= 0.60` → `0.0`
  - else → `(0.60 - d) / 0.30`
- **Avoid** the textbook `1 - distance / tolerance`. It compresses strong matches into the mid-range and makes the UI look broken.
- **Enrollment:** 3–7 photos per person, average the 128-d embeddings, pickle the result.
- **Python 3.14 trap:** `face_recognition_models` depends on `pkg_resources` (removed in setuptools 82+). Pin `setuptools==75.8.0`.
- **Distance is not similarity.** Lower = closer. Don't threshold as if higher were better.

Full context in `skills/face-recognition-confidence/SKILL.md`. Reference implementation in `scripts/confidence.py`.
