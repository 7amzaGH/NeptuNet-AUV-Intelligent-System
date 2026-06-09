# NeptuNet Integration Demo

This folder contains the lightweight framework-level orchestration demo for **NeptuNet**.

The goal of this demo is to show how the three NeptuNet perception levels can be coordinated through a simple resource-aware decision logic.

This integration layer does **not** perform real image inference, underwater robot control, SLAM, underwater physics simulation, or full onboard AUV deployment.

It is intentionally limited to **perception-level coordination**.

---

## Purpose

NeptuNet is organized as a hierarchical underwater inspection perception framework:

| Level | Module | Runtime Role |
|---|---|---|
| Level 1 | Pipeline Geometric Perception | Continuous pipeline context |
| Level 2 | Bubble-Based Early Warning | Lightweight suspicion monitoring |
| Level 3 | Leak Confirmation | Conditional plume confirmation |

The integration demo shows how these levels interact using simulated module outputs stored in JSON scenario files.

---

## Orchestration Logic

The orchestrator follows the main NeptuNet resource-aware logic:

```text
Level 1 is always active
        ↓
If the pipeline is detected, Level 2 monitors bubble activity
        ↓
If the bubble suspicion score exceeds the threshold, Level 3 is activated
        ↓
The system reports the current perception-level state
```

The possible system states are:

| State | Meaning |
|---|---|
| `PIPELINE_SEARCH` | Level 1 is active, but no valid pipeline is detected |
| `NORMAL_MONITORING` | Pipeline is detected and bubble suspicion remains low |
| `SUSPICION_UNCONFIRMED` | Bubble suspicion triggers Level 3, but no plume is confirmed |
| `LEAK_CONFIRMED` | Bubble suspicion triggers Level 3 and a gas plume is confirmed |

---

## Files

```text
integration/
└── neptunet_orchestrator.py
```

The main script is:

```text
neptunet_orchestrator.py
```

It reads simulated scenario files from:

```text
scenarios/
```

and optionally saves structured decision outputs to:

```text
outputs/
```

---

## Scenario Inputs

Each scenario file contains 10 inspection windows, representing approximately 10 seconds of inspection.

Example scenario files:

```text
scenarios/scenario_01_normal_inspection.json
scenarios/scenario_02_false_anomaly_return.json
scenarios/scenario_03_pipeline_reacquisition.json
scenarios/scenario_04_confirmed_leak.json
```

Each inspection window contains simulated outputs for:

```text
pipeline → Level 1 output
bubble   → Level 2 output
leak     → Level 3 output
```

Example window:

```json
{
  "frame_id": 1,
  "pipeline": {
    "detected": true,
    "center_offset_px": -5.2,
    "orientation_deg": 1.2,
    "direction": "STRAIGHT"
  },
  "bubble": {
    "activity_detected": true,
    "suspicion_score": 0.08,
    "dominant_features": []
  },
  "leak": {
    "plume_detected": false
  }
}
```

---

## Running the Orchestrator

From the repository root:

```bash
python integration/neptunet_orchestrator.py \
  --scenario scenarios/scenario_01_normal_inspection.json
```

Run the confirmed leak scenario:

```bash
python integration/neptunet_orchestrator.py \
  --scenario scenarios/scenario_04_confirmed_leak.json
```

Save the orchestrator output:

```bash
python integration/neptunet_orchestrator.py \
  --scenario scenarios/scenario_04_confirmed_leak.json \
  --output outputs/scenario_04_confirmed_leak.json
```

---

## Example Output

```text
[Window 05] System State: LEAK_CONFIRMED
[Level 1] ACTIVE  | Pipeline detected | center_offset=-13.5 px | orientation=5.3° | direction=RIGHT
[Level 2] ACTIVE  | Bubble monitoring | activity=True | suspicion_score=0.78
[Level 3] ACTIVE  | Leak confirmation triggered
[Level 3] Plume confirmed | centroid=[325, 270] | source=[318, 440] | direction=UP
```

---

## Notebook Demo

A full notebook demo is provided in:

```text
notebooks/01_neptunet_framework_simulation.ipynb
```

The notebook:

- loads the four scenario JSON files,
- runs the NeptuNet orchestrator,
- converts decisions into tables,
- visualizes suspicion scores,
- visualizes module activation timelines,
- saves CSV outputs for documentation.

Open the notebook:

```bash
jupyter notebook notebooks/01_neptunet_framework_simulation.ipynb
```

---

## Important Scope Note

This integration demo is a **framework-level simulation**, not a full robotic system.

It does not claim:

- closed-loop AUV control,
- underwater SLAM,
- acoustic localization,
- hydrodynamic simulation,
- real-time onboard deployment,
- full mission autonomy.

Its purpose is to transparently demonstrate how the NeptuNet perception modules can be coordinated at the framework level.

---

## Relation to Module Repositories

The real perception implementations are maintained in the module repositories:

| Module | Repository |
|---|---|
| Pipeline Geometric Perception | [Underwater-Pipeline-Geometric-Perception](https://github.com/7amzaGH/Underwater-Pipeline-Geometric-Perception) |
| Bubble-Based Early Warning | [TUBLEX-Bubble-Plume-Analysis](https://github.com/7amzaGH/TUBLEX-Bubble-Plume-Analysis) |
| Leak Confirmation | [Underwater-Gas-Leak-Geometric-Perception](https://github.com/7amzaGH/Underwater-Gas-Leak-Geometric-Perception) |

This folder only demonstrates the coordination logic connecting these modules inside the NeptuNet framework.

---

## Summary

The integration demo gives NeptuNet a lightweight technical framework layer while keeping the project scientifically honest.

It shows:

```text
Scenario inputs
        ↓
Perception-level orchestration
        ↓
System state decisions
        ↓
Notebook visualization and outputs
```

This makes the main NeptuNet repository more than a documentation hub, while avoiding unsupported claims of complete underwater robotic autonomy.