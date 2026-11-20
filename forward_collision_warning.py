#!/usr/bin/env python3
"""
Forward Collision Warning (ADAS) -- time-to-collision from box growth
======================================================================

A dashcam view from the driver's seat. YOLOv8n vehicle detection
(OpenCV DNN, no torch / ultralytics at run time) plus a plain centroid
tracker follows every vehicle ahead. For each tracked vehicle the
system estimates time-to-collision (TTC) from the rate its bounding
box grows:

* a vehicle's image size grows as ``1 / distance``, so with
  ``u = 1 / sqrt(box area)`` the distance is proportional to ``u``
  and, for a constant closing speed, ``u`` falls linearly with time:
  ``TTC = u / |du/dt|`` -- the slope comes from a least-squares fit
  of ``u`` over the last ~3 s of the track (median-smoothed over 3
  rounds), so a single flickering box cannot fake a warning, and an
  alert needs the fitted TTC below its threshold for 6 straight
  rounds (0.6 s);
* only vehicles in the forward path (centre x in 0.30--0.70 of the
  frame -- the ego lane) can raise an alert -- a car in the next lane
  is shown as MONITORED, never as a collision risk -- and a vehicle
  sweeping sideways across the frame faster than 80 px/s is PASSING
  the ego car (its box grows fast too, but it is beside the ego car,
  not ahead of it) and likewise cannot alert;
* states: TTC > 4 s SAFE (green), 2--4 s CAUTION (amber),
  TTC < 2 s COLLISION RISK (red banner + timestamped event log).

The board shows vehicles tracked, vehicles in path, the closest TTC
and how long each state has held. The demo clip is a real dashcam
recording; the ego car's own bonnet is excluded from detection (the
bottom 22% of the frame is not searched -- YOLO otherwise reads the
bonnet as a car).

Weights: ``yolov8n.onnx`` next to this script (12.8 MB, ships in this
folder). If it is missing, the script tries to export it from
``yolov8n.pt`` with Ultralytics (downloading the .pt if needed) and
otherwise exits with instructions.

Simulation mode (``--simulate``) renders a dashcam view from the ego
car: a road with lane markings and three cars ahead. One car in the
ego lane is closing fast (its drawn size grows) and correctly raises
COLLISION RISK; one car in the next lane holds a steady gap (SAFE);
one car pulls away (its drawn size shrinks, SAFE). The simulation's
vehicle source finds the cars with a colour mask + contours -- the
TTC, state and alert logic are the exact same code as real mode, and
the detector never sees the simulation's ground truth. Simulation is
an internal logic check; the delivered demo is real footage (see
README).

Honest framing: TTC from box growth is an ESTIMATE that assumes a
constant closing speed and a flat road; it is not a surveyed range
measurement, and it cannot see a vehicle that YOLO misses (small,
occluded or night-time vehicles).

Examples
--------
    python forward_collision_warning.py                 (opens a file picker)
    python forward_collision_warning.py --source pick   (same picker, explicit)
    python forward_collision_warning.py --source drive.mp4
    python forward_collision_warning.py --source 0      (webcam)
    python forward_collision_warning.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the
timestamped event log to ``output_events.log`` and the final counters
to ``output_summary.txt`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 150                     # stats panel at the bottom
VIDEO_W, VIDEO_H = W, H - TITLE_H - PANEL_H
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (90, 200, 90)
AMBER = (60, 170, 255)
RED = (60, 60, 235)
YELLOW = (0, 210, 235)
BLUE = (220, 160, 60)

# ---------------------------------------------------------------------------
# Detection + TTC tuning
# ---------------------------------------------------------------------------
DET_W, DET_H = 1280, 720          # detection runs on this frame size
DETECT_EVERY = 3                  # run YOLO every 3rd frame (10 fps)
CONF = 0.35
NMS_THRESH = 0.30
# A looser NMS merges YOLO's duplicate boxes on one car (a tight box
# plus a loose box on the same rear end survive at 0.45 and become two
# tracks that swap sizes -- fake TTC jumps). Genuinely separate cars
# in neighbouring lanes barely overlap, so 0.30 is safe.
VEHICLE_CLASSES = {2, 3, 5, 7}    # car, motorcycle, bus, truck (COCO)
# The ego bonnet fills the bottom of a dashcam frame and YOLO reads it
# as a car (a 983 px wide "car" at the bottom of the demo clip) -- the
# bottom 22% of the frame is never searched.
DETECT_Y1 = int(DET_H * 0.78)
MATCH_DIST = 80                   # px, centroid match radius -- cars move
                                  # ~20 px between detection rounds; a
                                  # wider radius lets one track steal a
                                  # neighbouring car and fake a huge TTC jump
MAX_MISSED = 15                   # detection rounds before a track drops
PT_URL = ("https://github.com/ultralytics/assets/releases/"
          "download/v8.2.0/yolov8n.pt")

# Forward path: a vehicle whose centre x falls in this band is "ahead"
# and can raise an alert; anything else is MONITORED only. The band is
# the EGO LANE only (measured on the demo clip: the car ahead sits at
# cx 0.57--0.66): cars in the next lane to the right (cx 0.70+) are
# being overtaken, not collided with, however fast their boxes grow.
PATH_X0, PATH_X1 = 0.30, 0.70
# A vehicle sweeping sideways across the frame faster than this is
# PASSING the ego car in the next lane (its box grows fast too, but it
# is beside the ego car, not ahead of it): it is shown as MONITORED
# and can never raise an alert. Measured on the demo clip: the car
# ahead drifts ~35 px/s, the overtaken SUV sweeps ~160 px/s.
PASSING_VX = 80.0                # px/s of lateral centroid movement
# TTC state thresholds (seconds).
CAUTION_TTC = 4.0
RISK_TTC = 2.0
# TTC fit: need at least this many size samples spanning this long.
# The span is 1 s so a single YOLO box jump (a car re-detected a
# size class larger in one round) cannot fake a collision.
FIT_MIN_POINTS = 6
FIT_MIN_SPAN_S = 1.0
FIT_MIN_CORR = 0.4                # slope must correlate with time
# A vehicle only enters CAUTION / COLLISION RISK after its fitted TTC
# has sat below the threshold for this many consecutive rounds. Six
# rounds is 0.6 s at the real-mode 10 Hz detection rate: a YOLO box
# burst (one car re-detected a size class larger for half a second)
# cannot fake a collision, while a genuinely closing car holds a small
# TTC for seconds. Standing down needs no confirmation.
CONFIRM_ROUNDS = 6
ALERT_HOLD_S = 1.0                # keep the banner this long after the
                                  # path clears, so it never flickers


# ---------------------------------------------------------------------------
# Vehicle sources -- real mode (YOLOv8 via OpenCV DNN) and simulation
# (colour mask). Both return boxes in detection-frame coordinates.
# ---------------------------------------------------------------------------
def ensure_weights(path):
    """Return a usable yolov8n.onnx path, exporting it if needed."""
    if os.path.exists(path):
        return path
    print(f"Weights not found at {path}; trying Ultralytics export ...")
    try:
        from ultralytics import YOLO
        pt = os.path.join(os.path.dirname(path) or ".", "yolov8n.pt")
        if not os.path.exists(pt):
            print("Downloading yolov8n.pt ...")
            urllib.request.urlretrieve(PT_URL, pt)
        print("Exporting yolov8n.pt -> yolov8n.onnx (one-time) ...")
        YOLO(pt).export(format="onnx", imgsz=640)
        exported = os.path.join(os.path.dirname(pt) or ".", "yolov8n.onnx")
        if os.path.exists(exported) and exported != path:
            os.replace(exported, path)
        return path
    except Exception as exc:  # noqa: BLE001 - report and stop cleanly
        print(f"Could not obtain weights: {exc}")
        print("Download yolov8n.onnx and place it next to this script, or")
        print("install ultralytics (`pip install ultralytics`) and re-run.")
        sys.exit(1)


class YoloVehicleDetector:
    """YOLOv8n vehicle detection through OpenCV DNN."""

    def __init__(self, weights_path):
        self.net = cv2.dnn.readNetFromONNX(ensure_weights(weights_path))

    def detect(self, frame):
        """frame is a DET_W x DET_H image; returns [x, y, w, h, conf]."""
        search = frame[:DETECT_Y1]
        blob = cv2.dnn.blobFromImage(search, 1 / 255.0, (640, 640),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # [8400, 84]: cx cy w h + 80 scores
        scores = out[:, 4:].max(axis=1)
        classes = out[:, 4:].argmax(axis=1)
        keep = (scores > CONF) & np.isin(classes, list(VEHICLE_CLASSES))
        sh, sw = search.shape[:2]
        boxes = []
        for (cx, cy, bw, bh), s in zip(out[keep, :4], scores[keep]):
            boxes.append([float((cx - bw / 2) * sw / 640),
                          float((cy - bh / 2) * sh / 640),
                          float(bw * sw / 640), float(bh * sh / 640),
                          float(s)])
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes([b[:4] for b in boxes],
                               [b[4] for b in boxes], CONF, NMS_THRESH)
        return [boxes[int(i)] for i in idx.flatten()] if len(idx) else []


def sim_vehicle_boxes(frame):
    """Colour-blob vehicle source for the rendered dashcam view."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 120, 80]),
                       np.array([115, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([float(x), float(y), float(w), float(h), 1.0])
    return boxes


# ---------------------------------------------------------------------------
# TTC tracker -- centroid tracking + per-vehicle time-to-collision
# ---------------------------------------------------------------------------
def _fit_ttc(history):
    """TTC from [(t, sqrt(area)), ...]; None if unusable, inf if not closing.

    Distance is proportional to ``u = 1 / sqrt(area)``, and for a vehicle
    closing at a constant speed ``u`` falls LINEARLY with time -- so the
    fit is done on ``u``, not on the size itself (fitting the size
    directly lags badly, because size grows hyperbolically as a car
    approaches). TTC = u / |du/dt|.
    """
    if len(history) < FIT_MIN_POINTS:
        return None
    t0 = history[0][0]
    ts = [t - t0 for t, _ in history]
    if ts[-1] - ts[0] < FIT_MIN_SPAN_S:
        return None
    us = [1.0 / s for _, s in history if s > 0]
    if len(us) < FIT_MIN_POINTS:
        return None
    n = len(ts)
    mt = sum(ts) / n
    mu = sum(us) / n
    num = sum((t - mt) * (u - mu) for t, u in zip(ts, us))
    den = sum((t - mt) ** 2 for t in ts)
    if den <= 0:
        return None
    slope = num / den
    var_u = sum((u - mu) ** 2 for u in us)
    if var_u <= 1e-12:
        return math.inf                   # rock-steady gap: not closing
    corr = num / math.sqrt(den * var_u)
    if slope >= 0:
        # u flat or rising = the gap is steady or opening. A solid fit
        # means the vehicle is genuinely not closing.
        return math.inf if abs(corr) >= FIT_MIN_CORR else None
    if abs(corr) < FIT_MIN_CORR:
        return None                       # flickering box, keep measuring
    return us[-1] / (-slope)              # TTC = u / |du/dt|


class CollisionTracker:
    """Tracks vehicles, estimates TTC for each one in the forward path."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frames = 0
        self.video_frames = 0
        self.events = []
        self.detections = 0
        self.vehicles_seen = 0
        self.caution_events = 0
        self.risk_events = 0
        self.min_ttc = None
        self.min_ttc_id = None
        self.first_caution_t = None
        self.first_risk_t = None
        self.risk_seconds = 0.0
        self.caution_seconds = 0.0
        self.alert = "SAFE"
        self._alert_since = None
        self._last_t = None
        self.in_path_now = 0
        self.closest_ttc = None
        self.closest_id = None

    def _log(self, t, text):
        self.events.append((t, text))

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah, _ = a
        bx, by, bw, bh, _ = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _centre(box):
        x, y, w, h, _ = box
        return (x + w / 2, y + h / 2)

    @staticmethod
    def _in_path(box):
        x, y, w, h, _ = box
        cx = (x + w / 2) / DET_W
        return PATH_X0 <= cx <= PATH_X1

    def reset(self, t):
        """Drop all tracks: the source looped, so the scene restarted."""
        n = len(self.tracks)
        self.tracks.clear()
        self._log(t, f"source looped -- {n} in-flight tracks reset "
                     f"(TTC histories restart from zero)")

    def update(self, boxes, t):
        """One detection round. Returns the alert state + live stats."""
        self.frames += 1
        self.detections += len(boxes)
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        used = set()
        for box in boxes:
            cx, cy = self._centre(box)
            best, best_d = None, MATCH_DIST
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if d >= best_d:
                    continue
                # A detection must also OVERLAP the track's last box:
                # without this, a track on a small distant car jumps
                # onto a big close car that appears beside it (the
                # size jump then reads as a fake collision -- found on
                # the first demo clip, where a track leapt from a
                # 39 px car onto a 411 px SUV in one round).
                if self._iou(box, tr["box"]) <= 0.05:
                    continue
                best, best_d = tid, d
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.vehicles_seen += 1
                self.tracks[tid] = {"id": tid, "cx": cx, "cy": cy,
                                    "box": box, "missed": 0,
                                    "history": [], "raw_sizes": [],
                                    "cx_history": [],
                                    "ttc": None,
                                    "state": "MEASURING",
                                    "below_risk": 0, "below_caution": 0,
                                    "in_path": False, "passing": False}
                best = tid
            used.add(best)
            tr = self.tracks[best]
            tr.update({"cx": cx, "cy": cy, "box": box, "missed": 0})
            tr["cx_history"].append((t, cx))
            tr["cx_history"] = [(tt, cc) for tt, cc in tr["cx_history"]
                                if t - tt <= 1.0]
            tr["passing"] = False
            ch = tr["cx_history"]
            if len(ch) >= 5 and ch[-1][0] - ch[0][0] >= 0.5:
                t0 = ch[0][0]
                xs = [tt - t0 for tt, _ in ch]
                cs = [cc for _, cc in ch]
                n = len(xs)
                mx = sum(xs) / n
                mc = sum(cs) / n
                den = sum((x - mx) ** 2 for x in xs)
                if den > 0:
                    vx = sum((x - mx) * (c - mc)
                             for x, c in zip(xs, cs)) / den
                    tr["passing"] = abs(vx) > PASSING_VX
            tr["in_path"] = self._in_path(box) and not tr["passing"]
            # A box clipped by the frame edge is a vehicle too close to
            # measure (it fills the windscreen): freeze the size history
            # so the last good TTC holds instead of collapsing.
            clipped = (box[0] <= 2 or box[1] <= 2
                       or box[0] + box[2] >= DET_W - 2
                       or box[1] + box[3] >= DET_H - 2)
            if not clipped:
                # Median-smooth the size over the last 3 rounds: a YOLO
                # box can jump a size class in one round, and a single
                # jump must not read as a collision.
                tr["raw_sizes"].append(math.sqrt(box[2] * box[3]))
                tr["raw_sizes"] = tr["raw_sizes"][-3:]
                med = sorted(tr["raw_sizes"])[len(tr["raw_sizes"]) // 2]
                tr["history"].append((t, med))
                # Keep the last 3 s of size samples, whatever the update
                # rate (real mode updates at 10 Hz, simulation at 30 Hz):
                # a 3 s window dilutes a half-second YOLO box burst.
                tr["history"] = [(tt, ss) for tt, ss in tr["history"]
                                 if t - tt <= 3.0]
            old_state = tr["state"]
            tr["ttc"] = _fit_ttc(tr["history"])
            if tr["ttc"] is None:
                raw_state = "MEASURING"
            elif not tr["in_path"]:
                raw_state = "MONITORED"
            elif tr["ttc"] == math.inf:
                raw_state = "SAFE"
            elif tr["ttc"] < RISK_TTC:
                raw_state = "COLLISION RISK"
            elif tr["ttc"] < CAUTION_TTC:
                raw_state = "CAUTION"
            else:
                raw_state = "SAFE"
            # Confirm an alert: the fitted TTC must sit below the
            # threshold for CONFIRM_ROUNDS consecutive rounds. Standing
            # down needs no confirmation.
            if raw_state == "COLLISION RISK":
                tr["below_risk"] += 1
                tr["below_caution"] += 1
            elif raw_state == "CAUTION":
                tr["below_risk"] = 0
                tr["below_caution"] += 1
            else:
                tr["below_risk"] = 0
                tr["below_caution"] = 0
            if tr["below_risk"] >= CONFIRM_ROUNDS:
                new_state = "COLLISION RISK"
            elif tr["below_caution"] >= CONFIRM_ROUNDS:
                new_state = "CAUTION"
            elif raw_state in ("COLLISION RISK", "CAUTION"):
                # The fitted TTC is below a threshold but has not held
                # it for 6 rounds yet: show CHECKING, not SAFE, so a
                # "TTC 1.2 s" label never sits next to the word SAFE.
                new_state = "CHECKING"
            else:
                new_state = raw_state
            tr["state"] = new_state
            if new_state != old_state:
                if new_state == "CAUTION":
                    self.caution_events += 1
                    if self.first_caution_t is None:
                        self.first_caution_t = t
                    self._log(t, f"vehicle #{tr['id']} CAUTION -- "
                                 f"TTC {tr['ttc']:.1f} s")
                elif new_state == "COLLISION RISK":
                    self.risk_events += 1
                    if self.first_risk_t is None:
                        self.first_risk_t = t
                    self._log(t, f"vehicle #{tr['id']} COLLISION RISK -- "
                                 f"TTC {tr['ttc']:.1f} s")
                elif old_state in ("CAUTION", "COLLISION RISK"):
                    self._log(t, f"vehicle #{tr['id']} back to "
                                 f"{new_state}"
                                 + (f" (TTC {tr['ttc']:.1f} s)"
                                    if tr["ttc"] is not None else ""))
            if (tr["ttc"] is not None and tr["in_path"]
                    and tr["ttc"] != math.inf):
                if self.min_ttc is None or tr["ttc"] < self.min_ttc:
                    self.min_ttc = tr["ttc"]
                    self.min_ttc_id = tr["id"]
        for tid in list(self.tracks):
            if tid in used:
                continue
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > MAX_MISSED:
                del self.tracks[tid]
        # -- live stats ------------------------------------------------------
        in_path = [tr for tr in self.tracks.values()
                   if tr["in_path"] and tr["ttc"] is not None
                   and tr["ttc"] != math.inf]
        self.in_path_now = len(in_path)
        if in_path:
            closest = min(in_path, key=lambda tr: tr["ttc"])
            self.closest_ttc = closest["ttc"]
            self.closest_id = closest["id"]
        else:
            self.closest_ttc = None
            self.closest_id = None
        risk_now = [tr for tr in in_path if tr["state"] == "COLLISION RISK"]
        caution_now = [tr for tr in in_path if tr["state"] == "CAUTION"]
        self.risk_seconds += len(risk_now) * dt
        self.caution_seconds += len(caution_now) * dt
        # -- alert: worst in-path state, held 1 s after the path clears
        if risk_now:
            self._alert_since = t
            new_alert = "COLLISION RISK"
        elif caution_now:
            self._alert_since = t
            new_alert = "CAUTION"
        elif (self._alert_since is not None
              and t - self._alert_since < ALERT_HOLD_S):
            new_alert = self.alert
        else:
            new_alert = "SAFE"
            self._alert_since = None
        if new_alert != self.alert:
            self.alert = new_alert
            if new_alert == "SAFE":
                self._log(t, "path clear -- alert back to SAFE")
        return {"alert": self.alert, "t": t}

    def summary(self):
        lines = [
            "FORWARD COLLISION WARNING -- RUN SUMMARY",
            f"frames processed: {self.video_frames}",
            f"tracker updates: {self.frames}",
            f"vehicle detections (summed): {self.detections}",
            f"vehicles tracked: {self.vehicles_seen}",
            f"CAUTION events (TTC < {CAUTION_TTC:.0f} s): "
            f"{self.caution_events}",
            f"COLLISION RISK events (TTC < {RISK_TTC:.0f} s): "
            f"{self.risk_events}",
            ("closest TTC: "
             + (f"{self.min_ttc:.1f} s (vehicle #{self.min_ttc_id})"
                if self.min_ttc is not None else "never measured")),
            ("first CAUTION at: "
             + (f"{self.first_caution_t:.1f} s"
                if self.first_caution_t is not None else "never")),
            ("first COLLISION RISK at: "
             + (f"{self.first_risk_t:.1f} s"
                if self.first_risk_t is not None else "never")),
            f"seconds in CAUTION: {self.caution_seconds:.1f} s",
            f"seconds in COLLISION RISK: {self.risk_seconds:.1f} s",
            f"final alert: {self.alert}",
            f"events logged: {len(self.events)}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Simulation renderer -- a dashcam view; one car closes fast, one holds
# its gap, one pulls away. The tracker only ever sees these pixels.
# ---------------------------------------------------------------------------
class DashcamSim:
    """Renders a dashcam view with three cars ahead."""

    def __init__(self, w, h, fps=30):
        self.w, self.h = w, h
        self.fps = fps
        self.frame_idx = 0
        # (x centre frac, y base frac, start size, closing time T,
        #  label). The closing car approaches at a constant speed, so
        # its size follows size0 / (1 - t/T) and its true TTC is T - t:
        # CAUTION at ~10 s, COLLISION RISK at ~12 s. Once it is huge it
        # slides past the ego car and out of frame (the tracker then
        # drops it, like the real clip's SUV passing by).
        self.cars = [
            {"cx": 0.50, "base_y": 0.62, "size0": 0.50, "T": 14.0,
             "kind": "closing", "label": "CLOSING"},
            {"cx": 0.82, "base_y": 0.55, "size0": 0.85, "T": None,
             "kind": "steady", "label": "STEADY"},
            {"cx": 0.24, "base_y": 0.50, "size0": 1.05, "T": None,
             "kind": "leaving", "label": "LEAVING"},
        ]

    def _draw_road(self, img):
        cv2.rectangle(img, (0, 0), (self.w, self.h), (52, 54, 58), -1)
        # Road: trapezoid from the horizon to the bottom corners.
        road = np.array([(int(self.w * 0.30), int(self.h * 0.30)),
                         (int(self.w * 0.70), int(self.h * 0.30)),
                         (self.w, self.h), (0, self.h)], dtype=np.int32)
        cv2.fillPoly(img, [road], (68, 68, 72))
        # Lane markings (dashed, converging to the vanishing point).
        vx, vy = self.w // 2, int(self.h * 0.30)
        for side in (-1, 1):
            for i in range(6):
                y0 = vy + i * int(self.h * 0.115)
                y1 = y0 + int(self.h * 0.07)
                x0 = vx + side * int((y0 - vy) * 0.55)
                x1 = vx + side * int((y1 - vy) * 0.55)
                cv2.line(img, (x0, y0), (x1, y1), (210, 210, 200), 4)
        # Centre line.
        cv2.line(img, (vx, vy), (vx, self.h), (200, 200, 190), 3)
        # Sky + horizon trees.
        cv2.rectangle(img, (0, 0), (self.w, vy), (120, 140, 165), -1)
        for x in range(0, self.w, 90):
            cv2.circle(img, (x + 20, vy - 8), 22, (60, 110, 60), -1)
        # Ego bonnet along the bottom (like the real dashcam clip).
        cv2.rectangle(img, (0, int(self.h * 0.88)), (self.w, self.h),
                      (210, 210, 212), -1)

    @staticmethod
    def _draw_car(img, cx, base_y, size):
        w = int(150 * size)
        h = int(84 * size)
        x0, y0 = int(cx - w / 2), int(base_y - h)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 130, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (200, 100, 20), 2)
        # Windshield band + tail lights so the blob reads as one car.
        cv2.rectangle(img, (x0 + int(w * 0.12), y0 + int(h * 0.14)),
                      (x0 + int(w * 0.88), y0 + int(h * 0.42)),
                      (40, 40, 46), -1)
        cv2.rectangle(img, (x0 + 4, y0 + h - int(h * 0.16)),
                      (x0 + int(w * 0.28), y0 + h - 4), (30, 30, 220), -1)
        cv2.rectangle(img, (x0 + int(w * 0.72), y0 + h - int(h * 0.16)),
                      (x0 + w - 4, y0 + h - 4), (30, 30, 220), -1)

    def frame(self):
        idx = self.frame_idx
        self.frame_idx += 1
        t = idx / self.fps
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_road(img)
        for car in self.cars:
            if car["kind"] == "closing":
                # Constant-speed approach: size = size0 / (1 - t/T).
                if t >= car["T"]:
                    continue            # reached the ego car: gone
                size = car["size0"] / (1 - t / car["T"])
                cx, base_y = car["cx"], car["base_y"]
                if size >= 4.0:
                    # Too big to measure any more: slide past the ego
                    # car and out of frame to the right over ~1.5 s, so
                    # the RISK banner reads clearly before the track
                    # drops (a car that fills the windscreen and then
                    # passes is what a near-miss looks like).
                    if car.get("exit_t") is None:
                        car["exit_t"] = t
                    dt_exit = t - car["exit_t"]
                    # Slide down and slightly right, out through the
                    # bottom of the frame over ~1.7 s. The lateral
                    # drift stays under the passing-vehicle threshold,
                    # so the RISK banner holds while the car passes.
                    cx = car["cx"] + 0.05 * dt_exit
                    base_y = car["base_y"] + 0.50 * dt_exit
                    size = 4.0
                    if base_y > 1.45:
                        continue
            elif car["kind"] == "steady":
                size = car["size0"]
                cx, base_y = car["cx"], car["base_y"]
            else:  # leaving: pulls away, shrinking slowly
                size = max(0.25, car["size0"] - 0.012 * t)
                cx, base_y = car["cx"], car["base_y"]
            self._draw_car(img, int(cx * self.w),
                           int(base_y * self.h), size)
        return img


# ---------------------------------------------------------------------------
# Rendering -- draws tracker output onto the 1280x720 canvas
# ---------------------------------------------------------------------------
def draw_canvas(det_frame, tracker, res, elapsed, mode_label, boxes):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    cv2.rectangle(canvas, (0, 0), (W, TITLE_H), PANEL_BG, -1)
    cv2.putText(canvas, "FORWARD COLLISION WARNING  --  ADAS TTC monitor",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(canvas, mode_label, (W - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    vh, vw = det_frame.shape[:2]
    scale = min(VIDEO_W / vw, VIDEO_H / vh)
    nw, nh = int(vw * scale), int(vh * scale)
    resized = cv2.resize(det_frame, (nw, nh))
    ox = (VIDEO_W - nw) // 2
    oy = TITLE_H + (VIDEO_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    sx, sy = nw / vw, nh / vh

    def _mx(x):
        return ox + int(x * sx)

    def _my(y):
        return oy + int(y * sy)

    # Forward-path band (the two vertical guides).
    x0 = _mx(PATH_X0 * vw)
    x1 = _mx(PATH_X1 * vw)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, oy), (x1, oy + nh), (0, 60, 80), -1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    cv2.line(canvas, (x0, oy), (x0, oy + nh), YELLOW, 2)
    cv2.line(canvas, (x1, oy), (x1, oy + nh), YELLOW, 2)
    cv2.putText(canvas, "FORWARD PATH", (x0 + 8, oy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

    # Vehicle boxes with their live TTC + state.
    for tid, tr in tracker.tracks.items():
        x, y, w, h, _ = tr["box"]
        state = tr["state"]
        colour = {"SAFE": GREEN, "CAUTION": AMBER,
                  "COLLISION RISK": RED, "MONITORED": GREY,
                  "MEASURING": GREY, "CHECKING": AMBER}[state]
        cv2.rectangle(canvas, (_mx(x), _my(y)), (_mx(x + w), _my(y + h)),
                      colour, 2 if state != "COLLISION RISK" else 3)
        ttc = tr["ttc"]
        if ttc is None:
            tag = f"#{tid} measuring..."
        elif ttc == math.inf:
            tag = f"#{tid} gap steady"
        else:
            tag = f"#{tid} TTC {ttc:.1f}s"
        tag += f"  {state}"
        if tr.get("passing"):
            tag += " (passing)"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx = _mx(x)
        # Clamp the label inside the canvas: a box at the right edge
        # used to print its label off the frame.
        if lx + tw + 8 > W - 4:
            lx = max(4, W - 4 - tw - 8)
        ly = _my(y) - 8
        if ly - th < oy + 4:            # box near the top: label below it
            ly = _my(y + h) + th + 10
        cv2.rectangle(canvas, (lx, ly - th - 4),
                      (lx + tw + 8, ly + 4), (24, 26, 32), -1)
        cv2.putText(canvas, tag, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # Bottom panel: board + alert banner.
    py = TITLE_H + VIDEO_H
    cv2.rectangle(canvas, (0, py), (W, H), PANEL_BG, -1)
    alert = tracker.alert
    acol = {"SAFE": GREEN, "CAUTION": AMBER,
            "COLLISION RISK": RED}[alert]
    cv2.rectangle(canvas, (0, py), (W, py + 6), acol, -1)
    banner = f"STATUS: {alert}"
    if alert == "COLLISION RISK":
        banner += "  !!! BRAKE -- VEHICLE AHEAD !!!"
    elif alert == "CAUTION":
        banner += "  -- closing on the vehicle ahead"
    cv2.putText(canvas, banner, (16, py + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, acol, 2)
    closest = (f"closest TTC: {tracker.closest_ttc:.1f} s "
               f"(#{tracker.closest_id})"
               if tracker.closest_ttc is not None else "closest TTC: --")
    cv2.putText(canvas, f"t = {elapsed:.1f} s", (16, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas, f"in path: {tracker.in_path_now}",
                (300, py + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas, closest, (520, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"vehicles: {tracker.vehicles_seen}   CAUTION: "
                f"{tracker.caution_events}   COLLISION RISK: "
                f"{tracker.risk_events}   min TTC: "
                + (f"{tracker.min_ttc:.1f} s"
                   if tracker.min_ttc is not None else "--"),
                (16, py + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
    return canvas


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


def open_source(source):
    if source == "pick":
        path = pick_file()
        if not path:
            print("No file chosen.")
            sys.exit(0)
        source = path
    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered dashcam view instead of a camera/video")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true",
                    help="loop a short source until --max-seconds is reached")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--weights", default=None,
                    help="path to yolov8n.onnx (default: next to the script)")
    args = ap.parse_args()

    fps = 30
    here = os.path.dirname(os.path.abspath(__file__))
    detector = None if args.simulate else YoloVehicleDetector(
        args.weights or os.path.join(here, "yolov8n.onnx"))
    tracker = CollisionTracker()
    sim = DashcamSim(DET_W, DET_H, fps) if args.simulate else None
    cap = None if args.simulate else open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else fps
    if not src_fps or src_fps <= 0:
        src_fps = fps
    step = max(1, round(src_fps / fps))  # sample the source at ~30 fps

    out_path = os.path.abspath(args.out)
    base, _ = os.path.splitext(out_path)
    events_path = base + "_events.log"
    summary_path = base + "_summary.txt"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mode_label = "SIMULATION (logic check)" if args.simulate else "REAL MODE"

    n_out = 0
    max_frames = int(args.max_seconds * fps)
    src_idx = 0
    boxes = []
    while n_out < max_frames:
        t = n_out / fps
        if args.simulate:
            raw = sim.frame()
            det_frame = raw
            boxes = sim_vehicle_boxes(det_frame)
            res = tracker.update(boxes, t)
        else:
            ok, frame = cap.read()
            src_idx += 1
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    src_idx = 1
                    tracker.reset(t)
                if not ok:
                    break
            if (src_idx - 1) % step != 0:
                continue
            det_frame = cv2.resize(frame, (DET_W, DET_H))
            if n_out % DETECT_EVERY == 0:
                boxes = detector.detect(det_frame)
                res = tracker.update(boxes, t)
            else:
                res = {"alert": tracker.alert, "t": t}
        canvas = draw_canvas(det_frame, tracker, res, t, mode_label, boxes)
        writer.write(canvas)
        n_out += 1
        if not args.no_display:
            try:
                cv2.imshow("Forward Collision Warning", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            except cv2.error:
                pass

    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    tracker.video_frames = n_out
    with open(events_path, "w") as f:
        for t, text in tracker.events:
            f.write(f"{t:6.1f}s  {text}\n")
    with open(summary_path, "w") as f:
        f.write(tracker.summary())
    print(tracker.summary())
    print(f"video:   {out_path}")
    print(f"events:  {events_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
