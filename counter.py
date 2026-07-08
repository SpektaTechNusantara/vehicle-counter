"""
Single-video vehicle counter (CLI).
Usage: python counter.py --video <path_to_video> [--model <path>] [--show]
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to input video or RTSP URL")
parser.add_argument("--model",   default=os.path.join(_HERE, "vcmodel1.onnx"),
                    help="Path to YOLO ONNX model file")
parser.add_argument("--tracker", default=os.path.join(_HERE, "bytetrack.yaml"))
parser.add_argument("--output",  default="output_counted.mp4")
parser.add_argument("--conf",    type=float, default=0.25)
parser.add_argument("--show",    action="store_true")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    args.output,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

model = YOLO(args.model, task="detect")

# Counting line — vertical at the horizontal middle of the frame.
# Trucks move left↔right so we count X crossings.
# Change LINE_X to reposition (e.g. w // 3 for left third).
LINE_X = w // 2

# Track which IDs have already crossed
crossed_ids = set()
counts = {0: 0, 1: 0}  # per-class crossing count
CLASS_NAMES = {0: "haul_truck", 1: "other_vehicles"}

# Remember first and last known X centre per track ID
prev_x = {}
first_x = {}
track_class = {}   # store which class each track ID belongs to
MIN_TRAVEL = 60    # px — a track must move at least this far to be counted

print(f"Processing {args.video} ...")
print(f"Counting line at x={LINE_X} (vertical, middle of frame). Edit LINE_X in the script to reposition.")

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=args.conf,
        classes=[0, 1],     # 0 = haul_truck, 1 = other_vehicles
        tracker=args.tracker,
        verbose=False,
    )

    annotated = results[0].plot()

    # Draw counting line (vertical)
    cv2.line(annotated, (LINE_X, 0), (LINE_X, h), (0, 255, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)

        clses = results[0].boxes.cls.cpu().numpy().astype(int)
        for box, tid, cls in zip(boxes, ids, clses):
            cx = int((box[0] + box[2]) / 2)

            if tid not in first_x:
                first_x[tid] = cx
                track_class[tid] = cls

            if tid in prev_x and tid not in crossed_ids:
                px = prev_x[tid]
                travelled = abs(cx - first_x[tid])
                if travelled >= MIN_TRAVEL:
                    if px < LINE_X <= cx or px > LINE_X >= cx:
                        crossed_ids.add(tid)
                        counts[track_class[tid]] += 1

            prev_x[tid] = cx

    # Overlay counts
    cv2.putText(annotated, f"Haul Truck:     {counts[0]}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated, f"Other Vehicles: {counts[1]}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    writer.write(annotated)
    if args.show:
        cv2.imshow("Haul Truck Counter", annotated)
        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
            break
    frame_num += 1
    if frame_num % 30 == 0:
        print(f"  Frame {frame_num} | haul_truck={counts[0]}  other_vehicles={counts[1]}", end="\r")

cap.release()
writer.release()
if args.show:
    cv2.destroyAllWindows()

print(f"\nDone. Output saved to: {args.output}")
for cls, name in CLASS_NAMES.items():
    print(f"  {name}: {counts[cls]}")
