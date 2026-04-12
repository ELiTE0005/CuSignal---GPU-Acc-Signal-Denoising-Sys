# CuSignal GPU Signal Processing Simulator

Interactive browser-based simulation of your full RadarScenes + Sonar
processing pipeline — no build tools required.

## Quick Start

```
cusignal-sim/
├── index.html          ← open this in your browser
└── src/
    ├── style.css
    └── simulation.js
```

**Option 1 — Just open the file**
Double-click `index.html`. Works in Chrome and Edge without a server.
Firefox requires a local server (see Option 2).

**Option 2 — VS Code Live Server (recommended)**
1. Install the "Live Server" extension in VS Code
2. Right-click `index.html` → **Open with Live Server**
3. Auto-reloads when you save any file

**Option 3 — Python server**
```bash
cd cusignal-sim
python -m http.server 8080
# Open http://localhost:8080
```

---

## What Each Panel Shows

| Panel | What it simulates |
|---|---|
| **Radar PPI** | Rotating sweep, target dots, velocity arrows, clutter |
| **Sonar Beamformed** | 128-beam output, range axis, echo blobs |
| **Range-Doppler Map** | FMCW 77 GHz target signatures by range + velocity |
| **Multi-Object Tracking** | Kalman bounding boxes, trails, prediction ellipses |

---

## Controls

| Control | Effect |
|---|---|
| **Targets** slider | Number of objects in scene (1–20) |
| **Noise** slider | Clutter density / noise floor level |
| **Speed** slider | Simulation frame rate multiplier (1×–5×) |
| **Mode** buttons | Radar only / Sonar only / Fused overlay |
| **Phase** buttons | Progressively enables pipeline features |
| **Pause / Resume** | Freeze simulation |
| **Reset** | Reinitialise all targets and logs |

### Phase Behaviour

- **Phase 1 — Data Integration**: Basic detections, no trails
- **Phase 2 — RT Processing**: Velocity arrows + target trails enabled
- **Phase 3 — Tracking**: Track IDs, Kalman prediction ellipses

---

## Plugging In Real RadarScenes Data

The simulator exposes a global JS function you can call from any
data bridge (Python → JS, WebSocket, etc.):

```javascript
window.injectRealFrame({
  detections: [
    { range: 12.4, azimuth: 0.35, rcs: 8.2, velocity: 3.1, label_id: 0 },
    { range: 28.1, azimuth: -0.8, rcs: 14.0, velocity: 8.5, label_id: 7 },
    // ...
  ],
  timestamp: 1234567890.123,
  sequence_id: 1
});
```

### Python Bridge Example

```python
# radarscenes_bridge.py
# Run alongside a local WebSocket server that feeds the browser

import h5py
import json
import asyncio
import websockets

async def stream_sequence(seq_path, ws):
    with h5py.File(seq_path, 'r') as f:
        radar = f['radar_data'][:]          # shape: (N, 13+)
        timestamps = radar[:, 0]

        for ts in sorted(set(timestamps)):
            mask = timestamps == ts
            frame_data = radar[mask]

            detections = []
            for row in frame_data:
                detections.append({
                    "range":    float(row[2]),   # range_sc
                    "azimuth":  float(row[3]),   # azimuth_sc
                    "rcs":      float(row[4]),   # rcs
                    "velocity": float(row[5]),   # vr_compensated
                    "label_id": int(row[13]),    # label_id
                })

            payload = json.dumps({
                "type": "frame",
                "detections": detections,
                "timestamp": float(ts),
            })
            await ws.send(payload)
            await asyncio.sleep(1 / 15)        # 15 Hz

async def main():
    async with websockets.serve(
        lambda ws, path: stream_sequence(
            "data/sequence_1/radar_data.h5", ws
        ),
        "localhost", 8765
    ):
        await asyncio.Future()

asyncio.run(main())
```

### Browser-side WebSocket (add to simulation.js or console)

```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'frame') {
    window.injectRealFrame(msg);
  }
};
```

### RadarScenes label_id mapping used internally

| label_id | RadarScenes class | Display colour |
|---|---|---|
| 0 | PassengerCar | Blue |
| 1 | LargeVehicle | Orange |
| 2 | Truck | Orange |
| 3 | Bus | Orange |
| 6 | Motorcycle | Purple |
| 7 | Pedestrian | Green |
| others | Unknown | Blue |

---

## Files

```
src/simulation.js    Main RAF loop, all drawing, state management
src/style.css        Layout, dark mode, component styles
index.html           Entry point, canvas structure, controls HTML
README.md            This file
```

---

## Extending

- **Add CFAR threshold line** to the Range-Doppler canvas in `drawRangeDoppler()`
- **Add ego-motion compensation** by modifying `updateTargets()` with odometry data
- **Add Hungarian assignment visualisation** in `drawTracking()` for Phase 3
- **Add classification confidence** badges to bounding boxes in `drawTracking()`
