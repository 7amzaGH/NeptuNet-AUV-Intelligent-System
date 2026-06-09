#!/usr/bin/env python3
"""
NeptuNet Framework Orchestrator Demo

This script simulates the high-level decision logic of NeptuNet.

It does NOT perform real image inference, AUV control, SLAM, underwater physics,
or onboard robotic deployment.

It demonstrates how module outputs can be coordinated:

Level 1: Pipeline geometric perception
Level 2: Bubble-based early warning
Level 3: Conditional leak confirmation

Each JSON scenario entry represents one inspection window, approximately 1 second.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Data classes
# ============================================================

@dataclass
class PipelineState:
    active: bool
    detected: bool
    center_offset_px: Optional[float] = None
    orientation_deg: Optional[float] = None
    direction: Optional[str] = None


@dataclass
class BubbleState:
    active: bool
    activity_detected: bool
    suspicion_score: float
    dominant_features: List[str]


@dataclass
class LeakState:
    active: bool
    plume_detected: bool
    plume_centroid: Optional[List[int]] = None
    probable_source: Optional[List[int]] = None
    propagation_direction: Optional[str] = None


@dataclass
class NeptuNetDecision:
    frame_id: int
    system_state: str
    pipeline: PipelineState
    bubble: BubbleState
    leak: LeakState


# ============================================================
# Simulated module adapters
# ============================================================

class Level1PipelinePerception:
    """Simulated Level 1 output adapter."""

    def run(self, window: Dict[str, Any]) -> PipelineState:
        data = window.get("pipeline", {})

        return PipelineState(
            active=True,
            detected=bool(data.get("detected", False)),
            center_offset_px=data.get("center_offset_px"),
            orientation_deg=data.get("orientation_deg"),
            direction=data.get("direction"),
        )


class Level2BubbleMonitoring:
    """Simulated Level 2 output adapter."""

    def run(self, window: Dict[str, Any], pipeline_detected: bool) -> BubbleState:
        if not pipeline_detected:
            return BubbleState(
                active=False,
                activity_detected=False,
                suspicion_score=0.0,
                dominant_features=[],
            )

        data = window.get("bubble", {})

        return BubbleState(
            active=True,
            activity_detected=bool(data.get("activity_detected", False)),
            suspicion_score=float(data.get("suspicion_score", 0.0)),
            dominant_features=list(data.get("dominant_features", [])),
        )


class Level3LeakConfirmation:
    """Simulated Level 3 output adapter."""

    def run(self, window: Dict[str, Any], activated: bool) -> LeakState:
        if not activated:
            return LeakState(
                active=False,
                plume_detected=False,
            )

        data = window.get("leak", {})

        return LeakState(
            active=True,
            plume_detected=bool(data.get("plume_detected", False)),
            plume_centroid=data.get("plume_centroid"),
            probable_source=data.get("probable_source"),
            propagation_direction=data.get("propagation_direction"),
        )


# ============================================================
# Orchestrator
# ============================================================

class NeptuNetOrchestrator:
    """
    Framework-level NeptuNet orchestration logic.

    Rules:
        1. Level 1 is always active.
        2. Level 2 activates only when the pipeline is detected.
        3. Level 3 activates only when bubble suspicion crosses threshold.
    """

    def __init__(self, suspicion_threshold: float = 0.70) -> None:
        self.suspicion_threshold = suspicion_threshold

        self.level1 = Level1PipelinePerception()
        self.level2 = Level2BubbleMonitoring()
        self.level3 = Level3LeakConfirmation()

    def process_window(self, window: Dict[str, Any]) -> NeptuNetDecision:
        frame_id = int(window.get("frame_id", 0))

        pipeline_state = self.level1.run(window)

        bubble_state = self.level2.run(
            window=window,
            pipeline_detected=pipeline_state.detected,
        )

        level3_should_activate = (
            pipeline_state.detected
            and bubble_state.active
            and bubble_state.suspicion_score >= self.suspicion_threshold
        )

        leak_state = self.level3.run(
            window=window,
            activated=level3_should_activate,
        )

        system_state = self._resolve_system_state(
            pipeline_state=pipeline_state,
            bubble_state=bubble_state,
            leak_state=leak_state,
        )

        return NeptuNetDecision(
            frame_id=frame_id,
            system_state=system_state,
            pipeline=pipeline_state,
            bubble=bubble_state,
            leak=leak_state,
        )

    def run_scenario(self, scenario: List[Dict[str, Any]]) -> List[NeptuNetDecision]:
        decisions = []

        for window in scenario:
            decision = self.process_window(window)
            decisions.append(decision)
            self.print_decision(decision)

        return decisions

    def _resolve_system_state(
        self,
        pipeline_state: PipelineState,
        bubble_state: BubbleState,
        leak_state: LeakState,
    ) -> str:
        if not pipeline_state.detected:
            return "PIPELINE_SEARCH"

        if leak_state.active and leak_state.plume_detected:
            return "LEAK_CONFIRMED"

        if leak_state.active and not leak_state.plume_detected:
            return "SUSPICION_UNCONFIRMED"

        if bubble_state.active and bubble_state.suspicion_score >= self.suspicion_threshold:
            return "SUSPICION"

        return "NORMAL_MONITORING"

    def print_decision(self, decision: NeptuNetDecision) -> None:
        print("=" * 80)
        print(f"[Window {decision.frame_id:02d}] System State: {decision.system_state}")

        p = decision.pipeline
        if p.detected:
            print(
                "[Level 1] ACTIVE  | Pipeline detected "
                f"| center_offset={p.center_offset_px} px "
                f"| orientation={p.orientation_deg}° "
                f"| direction={p.direction}"
            )
        else:
            print("[Level 1] ACTIVE  | Pipeline not detected")

        b = decision.bubble
        if b.active:
            print(
                "[Level 2] ACTIVE  | Bubble monitoring "
                f"| activity={b.activity_detected} "
                f"| suspicion_score={b.suspicion_score:.2f}"
            )

            if b.dominant_features:
                print(f"[Level 2] Features | {', '.join(b.dominant_features)}")
        else:
            print("[Level 2] INACTIVE | Waiting for valid pipeline context")

        l = decision.leak
        if l.active:
            print("[Level 3] ACTIVE  | Leak confirmation triggered")

            if l.plume_detected:
                print(
                    "[Level 3] Plume confirmed "
                    f"| centroid={l.plume_centroid} "
                    f"| source={l.probable_source} "
                    f"| direction={l.propagation_direction}"
                )
            else:
                print("[Level 3] No gas plume confirmed")
        else:
            print("[Level 3] INACTIVE | Suspicion threshold not reached")

        print("=" * 80)
        print()


# ============================================================
# Utilities
# ============================================================

def load_scenario(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        scenario = json.load(file)

    if not isinstance(scenario, list):
        raise ValueError("Scenario JSON must contain a list of inspection windows.")

    return scenario


def save_decisions(decisions: List[NeptuNetDecision], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = [asdict(decision) for decision in decisions]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the NeptuNet framework-level orchestration simulation."
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Path to the JSON scenario file.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Bubble suspicion threshold for activating Level 3.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save simulation decisions as JSON.",
    )

    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    scenario = load_scenario(scenario_path)

    orchestrator = NeptuNetOrchestrator(
        suspicion_threshold=args.threshold
    )

    decisions = orchestrator.run_scenario(scenario)

    if args.output:
        output_path = Path(args.output)
        save_decisions(decisions, output_path)
        print(f"Saved simulation output to: {output_path}")


if __name__ == "__main__":
    main()