# Eliminating Flicker in a Smart Meeting Room Presence System

## Problem Description

A facilities team runs a smart meeting room system that dims the lights and locks the booking display whenever the room is detected as empty. The system uses a webcam with a face detector running at roughly 10 frames per second. The lights and lock state are controlled by an IoT actuator that listens for a presence signal.

The team has raised a support ticket: "The lights flicker constantly even when someone is sitting still in the room. The system seems to flip between occupied/empty multiple times a minute, which is distracting and makes the booking display unreadable."

An engineer has confirmed the detector itself is working — it just drops roughly 1 in 8 frames even with a subject clearly in frame, which is normal for this class of detector. The current pipeline connects the detector directly to the actuator.

Your task is to write a Python module `presence_buffer.py` that eliminates the flicker without modifying the detector or actuator code. The buffer should smooth out these short detection dropouts so the actuator sees stable presence state rather than frame-by-frame noise.

Also write `demo_simulation.py`: a self-contained simulation that instantiates your buffer, runs 60 synthetic frames with roughly 15% random dropout, and prints a frame-by-frame log showing the raw detector output alongside the effective presence signal the actuator would receive. Capture the simulation output to `simulation_output.txt`.

Additionally write `architecture_notes.md` explaining where this buffer fits in the overall pipeline relative to any actuator-side stability filtering, and how you chose the persistence duration.

## Output Specification

- `presence_buffer.py` — the buffer module
- `demo_simulation.py` — runnable simulation (no external dependencies beyond standard library and the buffer module)
- `simulation_output.txt` — stdout captured from running `demo_simulation.py`
- `architecture_notes.md` — architecture notes on buffer placement and persistence duration choice
