"""Pose Coach main application.

Features:
  - Real-time elbow angle for biceps curls using MediaPipe Pose.
  - Rep counting based on angle thresholds (flexion/extension).
  - Smoothing angles, tempo calculation, best depth tracking.
  - On-demand CSV logging (privacy: no raw video saved automatically).
  - Optional annotated video recording only when toggled.
  - Keyboard controls:
        q: quit
        h: help overlay toggle
        r: toggle recording (annotated)
        n: new set (reset rep counter & stats, keep same logger file)
        s: snapshot PNG
        c: switch side (left/right)

Windows PowerShell run example:
    python pose_coach.py --side left --low 45 --high 160
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import cv2  # type: ignore
import mediapipe as mp  # type: ignore

from session_logger import SessionLogger
from pose_utils import (
    EmaSmoother,
    elbow_angle_from_landmarks,
)


@dataclass
class RepState:
    reps: int = 0
    phase: str = "start"  # start, lowering, raising
    last_phase_change_time: float = time.time()
    min_angle_current_rep: float = 999.0
    max_angle_current_rep: float = 0.0
    best_min_angle: float = 999.0  # track deepest flexion (smallest angle)
    last_rep_duration: float = 0.0


def detect_rep(angle: float, low_thresh: float, high_thresh: float, rs: RepState) -> Optional[bool]:
    """Update rep state machine; return True when a rep completes."""
    completed = None
    # Update min/max for current rep
    rs.min_angle_current_rep = min(rs.min_angle_current_rep, angle)
    rs.max_angle_current_rep = max(rs.max_angle_current_rep, angle)

    if rs.phase == "start":
        if angle > high_thresh * 0.95:  # almost extended
            rs.phase = "lowering"  # going down (eccentric)
            rs.last_phase_change_time = time.time()
    elif rs.phase == "lowering":
        if angle <= low_thresh:  # reached deep flexion
            rs.phase = "raising"
            rs.last_phase_change_time = time.time()
            rs.best_min_angle = min(rs.best_min_angle, rs.min_angle_current_rep)
    elif rs.phase == "raising":
        if angle >= high_thresh:  # returned to extension => rep complete
            rs.reps += 1
            rep_time = time.time() - rs.last_phase_change_time
            rs.last_rep_duration = rep_time
            completed = True
            # reset for next rep
            rs.phase = "lowering"
            rs.last_phase_change_time = time.time()
            rs.min_angle_current_rep = 999.0
            rs.max_angle_current_rep = 0.0
    return completed


def format_help_lines() -> list[str]:
    return [
        "Controls:",
        "  q: quit", "  h: toggle help", "  r: toggle recording", "  n: new set", "  s: snapshot", "  c: switch side",
    ]


def draw_overlay(frame, angle: Optional[float], rs: RepState, side: str, fps: float, recording: bool, show_help: bool):
    h, w = frame.shape[:2]
    y = 20
    cv2.putText(frame, f"Side: {side}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    y += 20
    cv2.putText(frame, f"Reps: {rs.reps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 20
    if angle is not None:
        cv2.putText(frame, f"Angle: {angle:5.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y += 20
    cv2.putText(frame, f"Phase: {rs.phase}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    y += 20
    cv2.putText(frame, f"Last rep s: {rs.last_rep_duration:4.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
    y += 20
    cv2.putText(frame, f"Best min angle: {rs.best_min_angle if rs.best_min_angle<999 else 0:4.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 1)
    y += 20
    cv2.putText(frame, f"FPS: {fps:4.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
    y += 20
    if recording:
        cv2.putText(frame, "REC", (w - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if show_help:
        hy = h - 20 * (len(format_help_lines()) + 1)
        for line in format_help_lines():
            cv2.putText(frame, line, (10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            hy += 18


def main():  # noqa: C901 complexity acceptable for a script
    ap = argparse.ArgumentParser(description="Pose Coach - Biceps Curl Counter")
    ap.add_argument("--side", default="left", choices=["left", "right"], help="Arm side to track")
    ap.add_argument("--low", type=float, default=45.0, help="Low angle threshold (deep flexion)")
    ap.add_argument("--high", type=float, default=160.0, help="High angle threshold (full extension)")
    ap.add_argument("--device", type=int, default=0, help="Webcam device index")
    ap.add_argument("--record_dir", default="recordings", help="Directory for optional annotated recordings")
    ap.add_argument("--width", type=int, default=960, help="Capture width")
    ap.add_argument("--height", type=int, default=540, help="Capture height")
    ap.add_argument("--show_help", action="store_true", help="Start with help overlay visible")
    args = ap.parse_args()

    rs = RepState()
    logger = SessionLogger(side=args.side, thresholds=f"LOW={args.low},HIGH={args.high}")
    logger.start()

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    angle_smoother = EmaSmoother(alpha=0.25)
    angles_window: Deque[float] = deque(maxlen=50)

    last_time = time.time()
    fps = 0.0
    recording = False
    writer = None
    Path(args.record_dir).mkdir(exist_ok=True)
    show_help = args.show_help

    snapshot_dir = Path("snapshots")
    snapshot_dir.mkdir(exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame from webcam; exiting.")
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(image_rgb)
            landmarks = getattr(results.pose_landmarks, "landmark", None)

            angle = None
            if landmarks:
                angle_raw = elbow_angle_from_landmarks(landmarks, frame.shape[1], frame.shape[0], side=args.side)
                if angle_raw is not None:
                    angle = angle_smoother.update(angle_raw)
                    angles_window.append(angle)
                    completed = detect_rep(angle, args.low, args.high, rs)
                    if completed:
                        logger.log_rep(rs.reps, rs.best_min_angle, rs.max_angle_current_rep, rs.last_rep_duration, side=args.side)

            # Draw pose skeleton
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            # FPS update
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            draw_overlay(frame, angle, rs, args.side, fps, recording, show_help)

            # Handle recording
            if recording and writer is not None:
                writer.write(frame)

            cv2.imshow("PoseCoach", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("r"):
                recording = not recording
                if recording:
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    fname = f"annotated_{int(time.time())}.avi"
                    writer = cv2.VideoWriter(str(Path(args.record_dir) / fname), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    print(f"Recording started: {fname}")
                else:
                    if writer:
                        writer.release()
                        writer = None
                        print("Recording stopped.")
            elif key == ord("n"):
                print("New set reset.")
                rs.phase = "start"
                rs.min_angle_current_rep = 999.0
                rs.max_angle_current_rep = 0.0
                rs.best_min_angle = 999.0
                rs.last_rep_duration = 0.0
                rs.reps = 0
            elif key == ord("s"):
                fname = snapshot_dir / f"snap_{int(time.time())}.png"
                cv2.imwrite(str(fname), frame)
                print(f"Saved snapshot {fname}")
            elif key == ord("c"):
                args.side = "right" if args.side == "left" else "left"
                logger.side = args.side
                print(f"Switched side to {args.side}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.close(summary_notes="Session ended.")
        print(f"Session log written: {logger.filepath}")


if __name__ == "__main__":
    main()
