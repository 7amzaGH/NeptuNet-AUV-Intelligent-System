# NeptuNet C++ Embedded Runtime Skeleton

This folder contains a lightweight C++ runtime skeleton for the **NeptuNet** framework.

The goal is to demonstrate how the NeptuNet perception-level coordination logic can be expressed in an embedded-friendly C++ structure.

This component does **not** perform neural network inference, underwater robot control, SLAM, underwater physics simulation, or full onboard AUV deployment.

It is intentionally limited to the **runtime state-machine logic** connecting the three NeptuNet perception levels.

---

## Purpose

The C++ runtime skeleton reproduces the high-level NeptuNet coordination logic:

```text
Level 1 is always active
        ↓
If the pipeline is detected, Level 2 becomes active
        ↓
If the bubble suspicion score exceeds the threshold, Level 3 is activated
        ↓
The system reports the current perception-level state
```

This is useful for showing how the framework logic could later be integrated into embedded software, robotic middleware, or onboard inspection systems.

---

## Runtime States

| State | Meaning |
|---|---|
| `PIPELINE_SEARCH` | Level 1 is active, but no valid pipeline is detected |
| `NORMAL_MONITORING` | Pipeline is detected and bubble suspicion remains low |
| `SUSPICION_UNCONFIRMED` | Bubble suspicion activates Level 3, but no plume is confirmed |
| `LEAK_CONFIRMED` | Bubble suspicion activates Level 3 and a gas plume is confirmed |

---

## Folder Structure

```text
cpp_runtime/
│
├── README.md
├── CMakeLists.txt
│
├── include/
│   └── neptunet_state_machine.hpp
│
├── src/
│   ├── neptunet_state_machine.cpp
│   └── main.cpp
│
└── examples/
    └── scenario_04_confirmed_leak.csv
```

---

## Build

From inside `cpp_runtime/`:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

## Run

Run the example confirmed leak scenario:

```bash
./neptunet_runtime ../examples/scenario_04_confirmed_leak.csv
```

Run with a custom suspicion threshold:

```bash
./neptunet_runtime ../examples/scenario_04_confirmed_leak.csv 0.75
```

---

## Input Format

The runtime reads simplified CSV scenario files.

Each row represents one inspection window, approximately one second.

```csv
window,pipeline_detected,center_offset_px,orientation_deg,direction,bubble_activity,suspicion_score,plume_detected,plume_centroid_x,plume_centroid_y,probable_source_x,probable_source_y,propagation_direction
```

The fields simulate outputs from the three NeptuNet perception levels:

| CSV Field | Meaning |
|---|---|
| `pipeline_detected` | Level 1 pipeline detection output |
| `center_offset_px` | Pipeline center offset in image space |
| `orientation_deg` | Pipeline orientation angle |
| `direction` | Directional cue: `LEFT`, `STRAIGHT`, or `RIGHT` |
| `bubble_activity` | Level 2 bubble activity flag |
| `suspicion_score` | Bubble-based leak suspicion score |
| `plume_detected` | Level 3 plume confirmation result |
| `plume_centroid_x`, `plume_centroid_y` | Image-plane plume centroid |
| `probable_source_x`, `probable_source_y` | Image-plane probable leak source |
| `propagation_direction` | Estimated plume propagation direction |

---

## Example Output

```text
[Window 05] System State: LEAK_CONFIRMED
  Level 1: ACTIVE  | pipeline detected | center_offset=-13.50 px | orientation=5.30 deg | direction=RIGHT
  Level 2: ACTIVE  | bubble_activity=true | suspicion_score=0.78
  Level 3: ACTIVE  | plume_detected=true | centroid=(325,270) | source=(318,440) | direction=UP
```

---

## Scope Note

This C++ runtime is a **deployment-oriented skeleton**, not a complete embedded implementation.

It does not claim:

- real-time onboard AUV deployment,
- closed-loop control,
- underwater navigation,
- SLAM,
- acoustic localization,
- real neural network inference,
- physical underwater validation.

Its purpose is to show how the NeptuNet perception coordination logic can be written as a clean C++ state machine.

---

## Relation to the Python Simulation

The Python notebook is used for explanation, visualization, and CSV generation.

The C++ runtime is used to demonstrate a lightweight embedded-style implementation of the same perception-level logic.

```text
Python notebook
        ↓
Research explanation and visualization

C++ runtime skeleton
        ↓
Embedded-style state-machine prototype
```

Both components support the same NeptuNet framework idea while remaining scientifically honest about the validated scope.