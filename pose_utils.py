"""
Utility functions for pose processing: angles, smoothing, and landmark helpers.
No heavy dependencies; uses only Python stdlib and typing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

Point = Tuple[float, float]


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """
    Calculate the angle ABC (at point B) between BA and BC in degrees [0, 180].

    Args:
        a: (x, y) of point A
        b: (x, y) of point B (vertex)
        c: (x, y) of point C
    Returns:
        Angle in degrees, clamped to [0, 180].
    """
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    dot = bax * bcx + bay * bcy
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    # Clamp cosine to [-1,1] to avoid NaNs from floating error.
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    # Normalize: angle at a joint typically capped to 180
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg
    return max(0.0, min(180.0, angle_deg))


@dataclass
class EmaSmoother:
    """Simple exponential moving average smoother for a scalar signal."""
    alpha: float = 0.2
    _value: Optional[float] = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


# Mediapipe Pose landmark indices for convenience
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16


def mp_landmark_xy(landmarks, idx: int, image_width: int, image_height: int) -> Optional[Point]:
    """
    Convert a mediapipe landmark with normalized coords to pixel (x,y).
    Returns None if landmark visibility is low or out of bounds.
    """
    try:
        lm = landmarks[idx]
    except (IndexError, TypeError):
        return None

    x = lm.x * image_width
    y = lm.y * image_height
    # Optional: filter by visibility; MP returns visibility in [0,1]
    if hasattr(lm, "visibility") and lm.visibility is not None and lm.visibility < 0.5:
        return None
    if math.isnan(x) or math.isnan(y):
        return None
    return (x, y)


def elbow_angle_from_landmarks(landmarks, image_width: int, image_height: int, side: str = "left") -> Optional[float]:
    """
    Compute elbow angle for the given side using pose landmarks.

    Args:
        landmarks: results.pose_landmarks.landmark
        image_width, image_height: frame dimensions
        side: "left" or "right"
    Returns:
        Angle in degrees or None if landmarks not confident
    """
    if side.lower().startswith("l"):
        sh_idx, el_idx, wr_idx = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    else:
        sh_idx, el_idx, wr_idx = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

    shoulder = mp_landmark_xy(landmarks, sh_idx, image_width, image_height)
    elbow = mp_landmark_xy(landmarks, el_idx, image_width, image_height)
    wrist = mp_landmark_xy(landmarks, wr_idx, image_width, image_height)

    if shoulder is None or elbow is None or wrist is None:
        return None
    return calculate_angle(shoulder, elbow, wrist)


def moving_min_max(values: Iterable[float]) -> Tuple[float, float]:
    vmin = float("inf")
    vmax = float("-inf")
    for v in values:
        if v < vmin:
            vmin = v
        if v > vmax:
            vmax = v
    return (vmin if vmin != float("inf") else 0.0, vmax if vmax != float("-inf") else 0.0)
