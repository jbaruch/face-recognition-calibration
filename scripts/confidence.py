"""
Piecewise confidence mapping for dlib face_recognition distances.

The library default tolerance is 0.6. The naive `1 - d/tol` mapping is technically
correct and empirically useless — a strong match at d=0.38 shows up as 0.37 on a UI
meter. The piecewise mapping below reflects how operators actually read the numbers.

    d <= 0.30  -> 1.0
    d >= 0.60  -> 0.0
    else       -> (0.60 - d) / 0.30
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping

import numpy as np


STRONG = 0.30
REJECT = 0.60
SPAN = REJECT - STRONG


def confidence(distance: float) -> float:
    """Map a face_recognition distance to a [0.0, 1.0] UI confidence score."""
    if distance <= STRONG:
        return 1.0
    if distance >= REJECT:
        return 0.0
    return (REJECT - distance) / SPAN


def enroll(encodings_per_person: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Average 3–7 encodings per person into a single enrolled vector.

    Input: {name: [encoding_1, encoding_2, ...]}
    Output: {name: mean_encoding}
    """
    return {
        name: np.mean(np.stack(vecs), axis=0)
        for name, vecs in encodings_per_person.items()
        if vecs
    }


def save_enrollment(enrolled: dict[str, np.ndarray], path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(enrolled, f)


def load_enrollment(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        return pickle.load(f)


def best_match(
    encoding: np.ndarray,
    enrolled: Mapping[str, np.ndarray],
) -> tuple[str | None, float, float]:
    """
    Return (name, distance, confidence) for the closest enrolled identity.

    If `enrolled` is empty, returns (None, inf, 0.0).
    """
    if not enrolled:
        return None, float("inf"), 0.0
    names = list(enrolled.keys())
    stack = np.stack([enrolled[n] for n in names])
    dists = np.linalg.norm(stack - encoding, axis=1)
    idx = int(np.argmin(dists))
    d = float(dists[idx])
    return names[idx], d, confidence(d)


if __name__ == "__main__":
    # Quick sanity check for the mapping shape.
    for d in (0.20, 0.30, 0.38, 0.45, 0.55, 0.60, 0.75):
        print(f"d={d:.2f}  conf={confidence(d):.3f}")
