from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class RepEntry:
    timestamp: str
    rep_index: int
    min_angle: float
    max_angle: float
    rep_time_s: float
    side: str
    notes: str = ""


@dataclass
class SessionLogger:
    output_dir: str = "logs"
    side: str = "left"
    thresholds: str = "ANGLE_LOW=45,ANGLE_HIGH=160"
    filepath: Optional[str] = None
    _writer: Optional[csv.writer] = field(default=None, init=False, repr=False)
    _file_handle: Optional[object] = field(default=None, init=False, repr=False)
    entries: List[RepEntry] = field(default_factory=list)

    def start(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.output_dir, f"session_{ts}.csv")
        self._file_handle = open(self.filepath, mode="w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file_handle)
        self._writer.writerow(["session_started", ts])
        self._writer.writerow(["side", self.side])
        self._writer.writerow(["thresholds", self.thresholds])
        self._writer.writerow([])
        self._writer.writerow(["timestamp", "rep_index", "min_angle", "max_angle", "rep_time_s", "side", "notes"])

    def log_rep(self, rep_index: int, min_angle: float, max_angle: float, rep_time_s: float, side: Optional[str] = None, notes: str = "") -> None:
        if self._writer is None:
            self.start()
        side_val = side or self.side
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = RepEntry(ts, rep_index, float(min_angle), float(max_angle), float(rep_time_s), side_val, notes)
        self.entries.append(entry)
        assert self._writer is not None
        self._writer.writerow([entry.timestamp, entry.rep_index, f"{entry.min_angle:.1f}", f"{entry.max_angle:.1f}", f"{entry.rep_time_s:.3f}", entry.side, entry.notes])

    def close(self, summary_notes: str = "") -> None:
        if self._writer is None:
            return
        # Summary
        total_reps = len(self.entries)
        avg_time = sum(e.rep_time_s for e in self.entries) / total_reps if total_reps else 0.0
        best_depth = min((e.min_angle for e in self.entries), default=0.0)
        self._writer.writerow([])
        self._writer.writerow(["summary_total_reps", total_reps])
        self._writer.writerow(["summary_avg_rep_time_s", f"{avg_time:.3f}"])
        self._writer.writerow(["summary_best_min_angle", f"{best_depth:.1f}"])
        if summary_notes:
            self._writer.writerow(["summary_notes", summary_notes])

        # Cleanup
        self._file_handle.close()  # type: ignore[arg-type]
        self._file_handle = None
        self._writer = None
