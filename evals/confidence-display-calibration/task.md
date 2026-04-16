# Building a Face Recognition Confidence Display

## Problem Description

A retail analytics company uses face recognition to greet returning VIP customers at store entrances. Their existing system shows a live confidence percentage on a staff tablet, but customer success has flagged a persistent complaint: the display always reads somewhere between 30% and 50% even for customers the system clearly recognises, making staff skeptical of the readout and ignoring it entirely.

The engineering team has narrowed it down to the confidence scoring logic. The current formula derives a percentage score from the raw dlib face_recognition distance. A senior engineer suspects the formula is mathematically defensible but perceptually broken — producing numbers that feel weak even on strong matches.

Your job is to write a Python module called `confidence_module.py` that replaces the existing confidence calculation. The module should expose a `compute_confidence(distance: float) -> float` function. You should also write a short verification script `verify_confidence.py` that demonstrates how the function behaves across a range of representative distance values and prints the results.

## Output Specification

Produce the following files in the workspace:
- `confidence_module.py` — the confidence computation module
- `verify_confidence.py` — a standalone script that imports `compute_confidence` from `confidence_module`, runs it on at least 5 representative distance values spanning the full distance range, and prints each distance alongside its computed confidence score
- `results.txt` — the printed output from running `verify_confidence.py` (capture it by running the script and saving stdout)

The `results.txt` file must be produced by actually running `verify_confidence.py`.
