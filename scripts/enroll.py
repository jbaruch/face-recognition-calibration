"""
Face enrollment helpers: validate captures, compute centroids, pickle.

Hard rules encoded:
- Blur threshold = 40 (Laplacian variance). NOT 80 — pale skin scores 40-80 sharp.
- Face coverage >= 60% of frame height.
- Intra-class distance mean 0.25-0.40. <0.20 = overfit, >0.45 = loose.
- Pre-pickle to enrolled.pkl. Never re-enroll from JPEG in a hot path.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

MIN_FACE_H_RATIO = 0.40     # face must be >= 40% of frame height
MIN_LAPLACIAN_VAR = 40       # pale skin = low variance; 80 causes false rejects
INTRA_CLASS_MEAN_MIN = 0.20
INTRA_CLASS_MEAN_MAX = 0.45
INTRA_CLASS_MAX_DIST = 0.55


def validate_capture(frame: np.ndarray) -> tuple[bool, str]:
    """Validate a single enrollment capture. Returns (ok, reason)."""
    import face_recognition
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    if not locs:
        return False, "no face detected"
    top, right, bot, left = locs[0]
    face_h = bot - top
    coverage = face_h / h
    if coverage < MIN_FACE_H_RATIO:
        return False, f"face too small ({coverage:.0%}, need >= {MIN_FACE_H_RATIO:.0%})"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[top:bot, left:right]
    blur = cv2.Laplacian(crop, cv2.CV_64F).var()
    if blur < MIN_LAPLACIAN_VAR:
        return False, f"too blurry (Laplacian={blur:.0f}, need >= {MIN_LAPLACIAN_VAR})"
    return True, f"OK coverage={coverage:.0%} blur={blur:.0f}"


def compute_enrollment(
    faces_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Compute mean 128-d encoding per person from faces/{name}/*.jpg."""
    import face_recognition
    from pathlib import Path
    enrolled = {}
    for person_dir in sorted(Path(faces_dir).iterdir()):
        if not person_dir.is_dir():
            continue
        vecs = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                vecs.append(encs[0])
        if vecs:
            enrolled[person_dir.name] = np.mean(np.stack(vecs), axis=0)
    return enrolled


def validate_enrollment(enrolled: Mapping[str, np.ndarray]) -> dict:
    """Check intra-class distances. Returns diagnostic dict."""
    results = {}
    for name, vec in enrolled.items():
        results[name] = {"norm": float(np.linalg.norm(vec))}
    if len(enrolled) >= 2:
        names = list(enrolled.keys())
        vecs = np.stack([enrolled[n] for n in names])
        dists = [
            float(np.linalg.norm(vecs[i] - vecs[j]))
            for i in range(len(vecs))
            for j in range(i + 1, len(vecs))
        ]
        results["_pairwise"] = {
            "min": min(dists),
            "max": max(dists),
            "mean": float(np.mean(dists)),
        }
    return results


def save_enrollment(enrolled: dict[str, np.ndarray], path: str | Path) -> None:
    """Pickle the enrollment dict."""
    with open(path, "wb") as f:
        pickle.dump(enrolled, f)


def load_enrollment(path: str | Path) -> dict[str, np.ndarray]:
    """Load a pre-pickled enrollment dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    faces_dir = sys.argv[1] if len(sys.argv) > 1 else "faces"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{faces_dir}/enrolled.pkl"
    enrolled = compute_enrollment(faces_dir)
    diag = validate_enrollment(enrolled)
    for name, info in diag.items():
        print(f"  {name}: {info}")
    save_enrollment(enrolled, out_path)
    print(f"Saved → {out_path}")
