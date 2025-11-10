# Pose Coach

Real-time rep counter and basic form feedback for biceps curls using MediaPipe Pose.

## Features
- Live elbow angle and phase detection (left/right arm)
- Rep counting with thresholds (default: low=45°, high=160°)
- Smoothed angles and last rep tempo
- CSV logging per session (privacy-first, no raw video saved by default)
- Optional annotated video recording on demand
- Snapshots and quick help overlay

## Install (Windows PowerShell)
```powershell
# From the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you see an error activating the venv, allow local scripts once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run
```powershell
# Left arm (default)
python pose_coach.py

# Right arm and custom thresholds
python pose_coach.py --side right --low 50 --high 170
```

## Controls
- q: quit
- h: toggle help
- r: toggle annotated recording (saved in `recordings/`)
- n: new set (resets counters)
- s: snapshot (saved in `snapshots/`)
- c: switch side (left/right)

## Logs and Privacy
- A CSV is written to `logs/` for each run with per-rep metrics and a summary.
- No raw video is stored unless you press `r` to start recording.

## Troubleshooting
- If the camera fails to open, try `--device 1` or check Windows camera permissions.
- For low light or occlusion, angles may be noisy—keep the tracked arm clearly visible.
- If MediaPipe fails to install, ensure you use Python 3.9–3.11 and upgrade pip: `python -m pip install --upgrade pip`.

## Notes
- Default thresholds target a typical curl; adjust for your range of motion.
- Extend easily to other joints by changing which landmarks define the angle.