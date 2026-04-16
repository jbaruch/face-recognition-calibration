"""
Producer-side face persistence buffer.

Hard rule: detectors miss 10-20% of frames. Hold last state for ~0.8s worth
of misses on the producer side instead of flipping to "nobody here" every
time a single frame drops.

This is the PRODUCER-side layer. The ACTUATOR-side debounce (iot-actuator-patterns)
is a separate layer — they compose.
"""

from __future__ import annotations


class FacePersistence:
    """Buffer that absorbs transient detection dropouts.

    Usage:
        persist = FacePersistence(max_misses=8)

        # per-frame:
        if face_detected:
            conf = persist.update(confidence_value)
        else:
            conf = persist.miss()
    """

    def __init__(self, max_misses: int = 8) -> None:
        self.max_misses = max_misses
        self._last_conf: float = 0.0
        self._miss_streak: int = 0

    def update(self, confidence: float) -> float:
        """Face detected this frame. Reset streak, record confidence."""
        self._miss_streak = 0
        self._last_conf = confidence
        return confidence

    def miss(self) -> float:
        """No face detected this frame. Return persisted or zero."""
        self._miss_streak += 1
        if self._miss_streak < self.max_misses:
            return self._last_conf
        self._last_conf = 0.0
        return 0.0

    @property
    def is_persisting(self) -> bool:
        return 0 < self._miss_streak < self.max_misses

    @property
    def miss_streak(self) -> int:
        return self._miss_streak


if __name__ == "__main__":
    # Quick demo: simulate 20 frames with 30% dropout
    import random
    p = FacePersistence(max_misses=4)
    for i in range(20):
        if random.random() > 0.3:
            c = p.update(0.85)
            print(f"frame {i:2d}: face  → conf={c:.2f}")
        else:
            c = p.miss()
            tag = "(persisted)" if p.is_persisting else "(gone)"
            print(f"frame {i:2d}: miss  → conf={c:.2f} {tag}")
