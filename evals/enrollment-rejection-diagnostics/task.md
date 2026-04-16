# Face Enrollment Pipeline for an Access Control System

## Problem Description

A co-working space operator is rolling out a face recognition door-access system. A facilities engineer will photograph each member at sign-up using a standard webcam. Those photos are then used to build a recognition enrollment that the door controller loads at startup.

The operator has had two failure modes in early testing: some members kept having their enrollment photos rejected as "too blurry" even though the photos looked sharp to the eye. Others enrolled fine but the recognition system produced inconsistent distances for the same person on different days, suggesting the enrollment embeddings were too tightly clustered around a single lighting condition.

You have been asked to write a Python script `enrollment_pipeline.py` that:
1. Simulates the capture-time validation logic that would be applied to each photo frame before accepting it into the enrollment set.
2. Runs pre-publish validation on a completed enrollment to check whether the embedding cloud has good spread.
3. Saves and loads the final enrollment using an efficient format suitable for production startup.

Because real webcam hardware is not available, the script should work entirely with synthetic data (generated numpy arrays representing face encodings and synthetic frames) rather than calling live camera APIs. The script must be runnable standalone.

Also write a `design_notes.md` file (1–2 pages) explaining the key design decisions: the blur rejection threshold you chose and why, the target range for intra-class distances, and how the enrollment is stored for production use.

## Output Specification

- `enrollment_pipeline.py` — runnable Python script using synthetic data that demonstrates: frame-level blur/coverage validation, pre-publish intra-class distance validation, and saving/loading the enrollment
- `design_notes.md` — design rationale covering blur threshold choice, intra-class distance targets, and enrollment storage strategy
- `pipeline_output.txt` — stdout captured from running `enrollment_pipeline.py`
