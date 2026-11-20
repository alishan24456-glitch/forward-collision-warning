# Forward Collision Warning (ADAS)

A dashcam-view forward collision warning system: the same idea that sits
behind real ADAS. YOLOv8n detects every vehicle ahead, a centroid tracker
follows them, and for each one the system estimates **time-to-collision
(TTC)** from the rate its bounding box grows — then warns before the gap
runs out.

## How TTC is estimated

A vehicle's image size grows as `1 / distance`. With
`u = 1 / sqrt(box area)`, the distance is proportional to `u`, and for a
vehicle closing at a constant speed `u` falls **linearly** with time, so

```
TTC = u / |du/dt|
```

The slope comes from a least-squares fit of `u` over the last ~3 s of the
track (box sizes median-smoothed over 3 rounds). Fitting `u` instead of
the size itself matters: size grows hyperbolically as a car approaches,
and a straight-line fit on the size lags badly. A single flickering YOLO
box cannot fake a warning, and an alert needs the fitted TTC below its
threshold for 6 straight rounds (0.6 s).

## States

| State | Meaning |
|---|---|
| `SAFE` | TTC above 4 s, or the gap is steady / opening |
| `CAUTION` | TTC 2–4 s — closing on the vehicle ahead |
| `COLLISION RISK` | TTC below 2 s — brake |
| `MONITORED` | Vehicle outside the forward path, or passing in the next lane |
| `CHECKING` | TTC below a threshold but not yet held for 6 rounds |
| `MEASURING` | Not enough size history yet |

Only vehicles in the **forward path** (centre x in 0.30–0.70 of the frame
— the ego lane) can raise an alert. A vehicle sweeping sideways across
the frame faster than 80 px/s is **passing** in the next lane: its box
grows fast too, but it is beside the ego car, not ahead of it, so it is
shown as MONITORED and can never alert. A detection must also overlap the
track's last box (IoU > 0.05), so a track on a small distant car cannot
jump onto a big close car that appears beside it. The ego car's own bonnet
is excluded from detection (the bottom 22% of the frame is never searched
— YOLO otherwise reads the bonnet as a car).

## Demo (real footage)

Real dashcam recording (Pexels video 3695999, free license): a highway
drive following a white car, with a truck ahead in the next lane. 30 s,
900 frames, 429 vehicle detections, 66 vehicles tracked.

- **1 CAUTION** — vehicle #23, the white car ahead, TTC 3.5 s at 20.9 s,
  back to SAFE at 21.0 s (the gap opened again).
- **0 COLLISION RISK** — no vehicle ahead ever closed to under 2 s on
  this drive; the closest measured TTC was 0.8 s on a vehicle that was
  passing in the next lane (MONITORED, correctly never alerted).
- Final alert: **SAFE**.

That is the honest result on this drive: the system watched the gap,
warned once as the car ahead was briefly approached, and never cried
wolf at the truck being overtaken next door.

## Simulation (internal logic check only — not the demo)

`--simulate` renders a dashcam view with three cars ahead: one in the ego
lane closing at a constant speed, one in the next lane holding a steady
gap, one pulling away. The simulation's vehicle source finds the cars with
a colour mask — the TTC, state and alert logic are the exact same code as
real mode, and the detector never sees the simulation's ground truth.

Verified: 30 s, 900 frames, 3 vehicles tracked — CAUTION at 10.3 s,
**COLLISION RISK at 12.2 s** (min TTC 1.6 s), then the car slides past
and the alert clears. The steady car reads SAFE (gap steady) and the
leaving car is MONITORED.

## Honest limits

- TTC from box growth is an **estimate** assuming a constant closing
  speed and a flat road; it is not a surveyed range measurement.
- A vehicle YOLO misses (small, occluded, night-time) gets no TTC.
- Adjacent-lane cars are separated from the ego lane by the forward-path
  band plus the passing test — on a multi-lane motorway where the ego car
  is overtaking slower traffic, the first version of this code fired
  false COLLISION RISK alerts on the cars being passed. Five real bugs
  were found and fixed during verification: the reciprocal (`u`) fit
  (a direct size fit lags), median smoothing plus a 3 s window plus
  6-round confirmation (YOLO box bursts), the passing test, the IoU
  match gate, and the ego-bonnet exclusion. The demo above is the result
  after those fixes.
- The demo clip's first candidate (Pexels 31901299, a motorway
  overtaking scene) was rejected for exactly that reason — it is a
  useful stress test, but the wrong demo.

## Run it

```bash
pip install -r requirements.txt
python forward_collision_warning.py                 # file picker
python forward_collision_warning.py --source drive.mp4
python forward_collision_warning.py --source 0      # webcam
python forward_collision_warning.py --simulate --max-seconds 30 --no-display
```

Press Q to quit. Output: `output.mp4`, `output_events.log`,
`output_summary.txt`. Weights `yolov8n.onnx` (12.8 MB) ship in this
folder; if missing, the script exports them from `yolov8n.pt` with
Ultralytics (downloading it if needed).

Author: Abdul Hanan — Computer Vision & AI Developer
