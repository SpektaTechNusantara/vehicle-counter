"""
Multi-camera vehicle counter with live web dashboard.

Usage:
    ./run_server.sh                          # uses cameras.yaml by default
    ./run_server.sh --config cameras.yaml
    ./run_server.sh --config cameras.yaml --port 8080
"""

import os
import sys
import time
import threading
import queue
import json
import yaml
import argparse
import cv2
from flask import Flask, Response, render_template, stream_with_context
from waitress import serve
from ultralytics import YOLO

MJPEG_FPS = 25                          # max frame rate sent to browser
MJPEG_INTERVAL = 1.0 / MJPEG_FPS


# ── Layer 1: pure counting logic (no CV2, no model, no Flask) ───────────────

def check_crossing(tid, cx, line_x, state):
    """
    Returns True if track `tid` just crossed `line_x` this frame.
    Mutates `state` in place.
    """
    if tid not in state["first_x"]:
        state["first_x"][tid] = cx

    crossed = False
    if tid in state["prev_x"] and tid not in state["crossed_ids"]:
        px = state["prev_x"][tid]
        if abs(cx - state["first_x"][tid]) >= state["min_travel"]:
            if px < line_x <= cx or px > line_x >= cx:
                state["crossed_ids"].add(tid)
                crossed = True

    state["prev_x"][tid] = cx
    return crossed


# ── FrameGrabber ─────────────────────────────────────────────────────────────

class FrameGrabber(threading.Thread):
    """
    Reads frames from an RTSP/file source continuously in its own thread.
    Always keeps only the newest frame so workers never process stale data.
    Auto-reconnects on RTSP drop; loops video files if loop=True.
    """

    def __init__(self, url, cam_id, loop=False):
        super().__init__(daemon=True, name=f"grabber-{cam_id}")
        self.url = url
        self.cam_id = cam_id
        self.loop = loop
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.connected = False

    def latest(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def run(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                print(f"[Cam {self.cam_id}] Cannot open {self.url!r} — retrying in 3 s")
                time.sleep(3)
                continue

            self.connected = True
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            # For local files, throttle reads to the video's native FPS so we
            # don't decode hundreds of frames per second and thrash the GPU.
            # For RTSP the stream itself sets the pace so no sleep is needed.
            is_file = not str(self.url).startswith("rtsp")
            frame_interval = 1.0 / src_fps if is_file else 0
            print(f"[Cam {self.cam_id}] Connected — {src_fps:.0f} fps source")

            while not self._stop.is_set():
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print(f"[Cam {self.cam_id}] Stream ended — reconnecting")
                    break
                # Drop the previous frame if the worker hasn't consumed it yet
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put(frame)

                if frame_interval:
                    elapsed = time.monotonic() - t0
                    remaining = frame_interval - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            cap.release()
            self.connected = False
            if not self.loop:
                time.sleep(1)

    def stop(self):
        self._stop.set()


# ── FrameBuffer ───────────────────────────────────────────────────────────────

class FrameBuffer:
    """Thread-safe slot holding the latest annotated JPEG and counts."""

    def __init__(self, label):
        self.label = label
        self.active_classes: list[int] = []   # set by CameraWorker after init
        self.running = True                   # reflects worker pause state
        self._lock = threading.Lock()
        self._jpeg = None
        self._counts: dict[int, int] = {}
        self._seq = 0           # incremented on every new frame

    def put(self, jpeg_bytes, counts):
        with self._lock:
            self._jpeg = jpeg_bytes
            self._counts = dict(counts)
            self._seq += 1

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_jpeg_if_new(self, last_seq):
        """Returns (jpeg, seq) if a newer frame is available, else (None, last_seq)."""
        with self._lock:
            if self._seq == last_seq or self._jpeg is None:
                return None, last_seq
            return self._jpeg, self._seq

    def reset_counts(self, counts):
        """Zero counts only — does NOT touch the JPEG frame or seq."""
        with self._lock:
            self._counts = dict(counts)

    def get_counts(self):
        with self._lock:
            return dict(self._counts)


# ── CameraWorker ──────────────────────────────────────────────────────────────

CLASS_NAMES   = {0: "haul_truck", 1: "other_vehicles"}
CLASS_INDICES = {"haul_truck": 0, "other_vehicles": 1}
CLASS_COLORS  = {0: (0, 255, 255), 1: (0, 200, 255)}


class CameraWorker(threading.Thread):
    """
    Owns one YOLO model instance and one ByteTrack state.
    Pops frames from a FrameGrabber, runs inference, pushes annotated
    JPEG + counts into a FrameBuffer for the web layer to serve.
    """

    def __init__(self, cam_id, grabber, buf, cam_cfg, global_cfg):
        super().__init__(daemon=True, name=f"worker-{cam_id}")
        self.cam_id = cam_id
        self.grabber = grabber
        self.buf = buf
        self.line_x_frac = cam_cfg.get("line_x", 0.5)
        self.model_path = global_cfg["model"]
        self.tracker_path = global_cfg["tracker"]
        self.conf = global_cfg.get("conf", 0.25)
        # Which classes to detect, track and count — names from cameras.yaml
        show = global_cfg.get("show_classes", list(CLASS_INDICES.keys()))
        self.active_classes = [CLASS_INDICES[n] for n in show if n in CLASS_INDICES]
        self._stop = threading.Event()
        self._state = {
            "first_x": {},
            "prev_x": {},
            "track_class": {},
            "crossed_ids": set(),
            "min_travel": 60,
        }
        self._counts = {cls: 0 for cls in self.active_classes}
        self._lock = threading.Lock()
        self._paused = threading.Event()
        self._paused.set()      # start paused — user clicks Start to activate
        buf.active_classes = self.active_classes
        buf.running = False     # reflect initial stopped state

    def pause(self):
        self._paused.set()
        self.buf.running = False

    def resume(self):
        self._paused.clear()
        self.buf.running = True

    def reset(self):
        with self._lock:
            self._counts = {cls: 0 for cls in self.active_classes}
            self._state["crossed_ids"].clear()
            self._state["first_x"].clear()
            self._state["prev_x"].clear()
            self._state["track_class"].clear()
        # Zero the buffer counts immediately so SSE sees 0 on its very next tick.
        # Use reset_counts() — never buf.put() here — to avoid corrupting the MJPEG stream.
        self.buf.reset_counts(self._counts)

    def run(self):
        # Stagger workers so they don't hit the GPU simultaneously at startup
        time.sleep(self.cam_id * 0.5)
        model = YOLO(self.model_path, task="detect")
        ftimes = []

        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            frame = self.grabber.latest()
            if frame is None:
                time.sleep(0.02)
                continue

            t0 = time.time()
            h, w = frame.shape[:2]
            line_x = int(w * self.line_x_frac)

            results = model.track(
                frame,
                persist=True,
                conf=self.conf,
                classes=self.active_classes,
                tracker=self.tracker_path,
                verbose=False,
            )

            annotated = results[0].plot()
            cv2.line(annotated, (line_x, 0), (line_x, h), (0, 255, 255), 2)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)
                clses = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, tid, cls in zip(boxes, ids, clses):
                    cx = int((box[0] + box[2]) / 2)
                    if tid not in self._state["first_x"]:
                        self._state["track_class"][tid] = int(cls)
                    if check_crossing(tid, cx, line_x, self._state):
                        with self._lock:
                            self._counts[self._state["track_class"][tid]] += 1

            # Overlays — only show active classes
            cv2.putText(annotated, self.buf.label,
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y = 58
            for cls in self.active_classes:
                label = CLASS_NAMES[cls].replace("_", " ").title()
                color = CLASS_COLORS[cls]
                cv2.putText(annotated, f"{label}: {self._counts[cls]}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y += 28

            ftimes.append(time.time() - t0)
            if len(ftimes) > 30:
                ftimes.pop(0)
            fps = 1.0 / (sum(ftimes) / len(ftimes))
            cv2.putText(annotated, f"FPS: {fps:.1f}",
                        (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.buf.put(jpeg.tobytes(), self._counts)

    def stop(self):
        self._stop.set()


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
buffers: list[FrameBuffer] = []       # populated in main() before app.run()
workers: list[CameraWorker] = []


_TICK = 0.005   # 5ms sleep granularity — threads release quickly on disconnect

def _mjpeg(cam_id):
    """Generator yielding MJPEG frames, capped at MJPEG_FPS, skipping duplicates."""
    last_seq = -1
    try:
        while True:
            t0 = time.monotonic()
            jpeg, last_seq = buffers[cam_id].get_jpeg_if_new(last_seq)
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            # Sleep in small ticks so a disconnecting client releases the
            # Waitress thread within ~5ms rather than up to 40ms.
            elapsed = time.monotonic() - t0
            remaining = MJPEG_INTERVAL - elapsed
            while remaining > 0:
                time.sleep(min(_TICK, remaining))
                remaining -= _TICK
    except GeneratorExit:
        pass


def _sse_counts():
    """Generator yielding SSE events with JSON counts every second."""
    while True:
        counts_snapshot = {}
        for i, b in enumerate(buffers):
            raw = b.get_counts()
            counts_snapshot[str(i)] = {
                "label": b.label,
                "running": b.running,
                **{CLASS_NAMES[cls]: raw.get(cls, 0) for cls in b.active_classes},
            }
        yield f"data: {json.dumps(counts_snapshot)}\n\n"
        time.sleep(1)


@app.route("/")
def dashboard():
    cams = [{"id": i, "label": b.label} for i, b in enumerate(buffers)]
    return render_template("dashboard.html", cameras=cams)


@app.route("/stream/<int:cam_id>")
def stream(cam_id):
    if cam_id >= len(buffers):
        return "Not found", 404
    return Response(
        _mjpeg(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/counts")
def counts_sse():
    return Response(
        stream_with_context(_sse_counts()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/reset/<int:cam_id>", methods=["POST"])
def reset_cam(cam_id):
    if cam_id >= len(workers):
        return {"error": "not found"}, 404
    workers[cam_id].reset()
    return {"status": "ok", "cam_id": cam_id}


@app.route("/stop/<int:cam_id>", methods=["POST"])
def stop_cam(cam_id):
    if cam_id >= len(workers):
        return {"error": "not found"}, 404
    workers[cam_id].pause()
    return {"status": "ok", "running": False}


@app.route("/start/<int:cam_id>", methods=["POST"])
def start_cam(cam_id):
    if cam_id >= len(workers):
        return {"error": "not found"}, 404
    workers[cam_id].resume()
    return {"status": "ok", "running": True}


@app.route("/health")
def health():
    return {"status": "ok", "cameras": len(buffers)}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to cameras.yaml")
    parser.add_argument("--port", type=int, default=None, help="Override port from config")
    args = parser.parse_args()

    # Default config path: cameras.yaml next to this script
    if args.config is None:
        args.config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras.yaml")

    cfg_dir = os.path.dirname(os.path.abspath(args.config))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve tracker path relative to the config file if not absolute
    tracker = cfg.get("tracker", "bytetrack.yaml")
    if not os.path.isabs(tracker):
        cfg["tracker"] = os.path.join(cfg_dir, tracker)

    global buffers, workers
    grabbers = []

    for i, cam_cfg in enumerate(cfg["cameras"]):
        url   = cam_cfg["url"]
        label = cam_cfg.get("label", f"Camera {i}")
        loop  = cam_cfg.get("loop", False)

        buf     = FrameBuffer(label)
        grabber = FrameGrabber(url, i, loop=loop)
        worker  = CameraWorker(i, grabber, buf, cam_cfg, cfg)

        buffers.append(buf)
        grabbers.append(grabber)
        workers.append(worker)

    host = cfg.get("host", "0.0.0.0")
    port = args.port or cfg.get("port", 5000)

    print(f"\n  Dashboard → http://localhost:{port}/")
    print(f"  Cameras   : {len(buffers)} (all paused — click Start on each camera)")
    print(f"  Model     : {cfg['model']}")

    # Start grabbers and workers in background — web server is available immediately.
    # Workers start paused; user activates each camera via the dashboard.
    for g in grabbers:
        g.start()
    for w in workers:
        w.start()
    print()

    # Each browser tab needs (n_cameras + 1) threads for MJPEG + SSE.
    # Allocate enough for ~4 simultaneous clients plus headroom.
    thread_count = max(32, (len(buffers) + 1) * 4)
    serve(app, host=host, port=port, threads=thread_count)


if __name__ == "__main__":
    main()
