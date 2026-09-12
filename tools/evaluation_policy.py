"""Deployable term-major observation history for CPU policy evaluators.

This matches MJLab CircularBuffer: oldest frame first within each observation
term, and the first frame after reset repeats across all history slots.
"""

from __future__ import annotations

import numpy as np


TERM_DIMS = (3, 3, 14, 14, 14, 3, 4, 6)


class ObservationHistory:
    def __init__(self, length: int, term_dims: tuple[int, ...] = TERM_DIMS):
        if length < 1 or term_dims != TERM_DIMS:
            raise ValueError("Unsupported history length or MicroDuck observation layout")
        self.length = length
        self.term_dims = term_dims
        self.reset()

    def reset(self) -> None:
        self.frames: np.ndarray | None = None
        self.time: float | None = None

    def observe(self, frame: np.ndarray, simulation_time: float) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (sum(self.term_dims),):
            raise ValueError(f"Expected one 61D frame, received {frame.shape}")
        if self.frames is None:
            self.frames = np.repeat(frame[None, :], self.length, axis=0)
        elif self.time != simulation_time:
            if self.time is not None and simulation_time < self.time:
                raise ValueError("Simulation time reset without resetting policy history")
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = frame
        # Reading a trace and inferring at the same control instant must not
        # append twice or replace the previous action with the current action.
        self.time = simulation_time
        offset = 0
        pieces = []
        for width in self.term_dims:
            pieces.append(self.frames[:, offset : offset + width].reshape(-1))
            offset += width
        return np.concatenate(pieces)


def configure_policy_history(controller) -> dict:
    """Install history only when explicitly declared by the ONNX metadata."""
    session = controller.ort_session
    dimension = session.get_inputs()[0].shape[-1]
    metadata = session.get_modelmeta().custom_metadata_map
    length = int(metadata.get("ducklab_actor_history_length", "1"))
    term_dims = tuple(int(value) for value in metadata.get(
        "ducklab_actor_term_dims", ",".join(map(str, TERM_DIMS))
    ).split(","))
    if length == 1:
        if dimension not in (61, 67):
            raise ValueError(f"Unsupported policy input {dimension}; temporal policies must declare history metadata")
        controller.reset_observation_history = lambda: None
    else:
        if dimension != 61 * length or term_dims != TERM_DIMS:
            raise ValueError("Policy history metadata disagrees with the exported input layout")
        for key, expected in (
            ("ducklab_actor_history_order", "term-major-oldest-to-newest"),
            ("ducklab_actor_history_reset", "repeat-first-frame"),
            ("ducklab_control_frequency_hz", "50"),
        ):
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"Unsupported {key}: {metadata[key]}")
        history = ObservationHistory(length, term_dims)
        base_observations = controller.get_observations
        controller.get_observations = lambda: history.observe(
            base_observations(), float(controller.data.time)
        )
        controller.reset_observation_history = history.reset
    return {
        "input_dimension": dimension,
        "history_length": length,
        "term_dimensions": list(term_dims),
        "history_layout": "term-major-oldest-to-newest",
        "reset_fill": "repeat-first-observation",
    }
