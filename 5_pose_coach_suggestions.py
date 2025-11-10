# Pose Coach - Pose Estimation (Suggestions Only)
# Summary:
#   Real-time rep counter and basic form feedback for biceps curls.
# Why it matters:
#   Immediate feedback helps users improve form and avoid injury.
#
# Suggested components:
#   - Model: MediaPipe pose or YOLO-pose; show skeleton overlay.
#   - Angles: compute knee/hip or elbow/shoulder angles; threshold-based rep detection.
#   - UI: current rep count, set counter, best angle, and a timer.
#   - Coaching: simple messages like "go deeper" or "keep back straight".
#   - Logging: CSV per session (reps, tempo, avg angles, rest time).
#   - Privacy: do not store raw video by default; allow saving annotated clips on demand.
#
# Stretch ideas:
#   - Audio cues (metronome, "rep complete").
#   - Simple leaderboard with best streaks.
#
# Notes:
#   Provide a demo video and default thresholds for quick testing.
