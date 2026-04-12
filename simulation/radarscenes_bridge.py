import h5py
import json
import asyncio
import os
import websockets

DEFAULT_H5 = os.environ.get(
    "RADARSCENES_H5",
    os.path.join("RadarScenes", "data", "sequence_1", "radar_data.h5"),
)

async def stream_sequence(seq_path, ws):
    print(f"Opening: {seq_path}")
    with h5py.File(seq_path, 'r') as f:
        radar = f['radar_data'][:]          # shape: (N, 13+)
        timestamps = radar['timestamp']

        for ts in sorted(set(timestamps)):
            mask = timestamps == ts
            frame_data = radar[mask]

            detections = []
            for row in frame_data:
                detections.append({
                    "range":    float(row['range_sc']),
                    "azimuth":  float(row['azimuth_sc']),
                    "rcs":      float(row['rcs']),
                    "velocity": float(row['vr_compensated']),
                    "label_id": int(row['label_id']),
                })

            payload = json.dumps({
                "type": "frame",
                "detections": detections,
                "timestamp": float(ts),
            })
            await ws.send(payload)
            await asyncio.sleep(1 / 15)        # 15 Hz

async def main():
    print("Starting WebSocket bridge server on ws://0.0.0.0:8081...")
    async with websockets.serve(
        lambda ws: stream_sequence(DEFAULT_H5, ws),
        "0.0.0.0", 8081, # Use 0.0.0.0 to allow docker-compose mapping
        ping_interval=None
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
