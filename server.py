"""
Multi-camera vehicle counter with live web dashboard.
Cameras are configured from the browser UI (localStorage), not from cameras.yaml.

Usage:
    python server.py [--config cameras.yaml] [--port 5000]
"""

import os
import sys
import time
import threading
import queue
import json
import base64
import logging
import yaml
import argparse
import cv2
import numpy as np
from flask import Flask, Response, render_template, stream_with_context, request
from waitress import serve
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("server.log", mode="a"),
    ],
)
log = logging.getLogger("vcounter")

MJPEG_FPS = 60
MJPEG_INTERVAL = 1.0 / MJPEG_FPS


# ── Layer 1: pure counting logic ───────────────────────────────────────────

def check_crossing(tid, cx, cy, line_pos, state, orientation="vertical", direction="forward"):
    first_key = f"first_{tid}"
    prev_key = f"prev_{tid}"
    pos = cy if orientation == "horizontal" else cx
    if first_key not in state:
        state[first_key] = pos

    crossed = False
    if prev_key in state and f"crossed_{tid}" not in state:
        prev = state[prev_key]
        if abs(pos - state[first_key]) >= state["min_travel"]:
            if direction == "forward":
                if prev < line_pos <= pos:
                    state[f"crossed_{tid}"] = True
                    crossed = True
            else:  # backward
                if prev > line_pos >= pos:
                    state[f"crossed_{tid}"] = True
                    crossed = True

    state[prev_key] = pos
    return crossed


# ── FrameGrabber ──────────────────────────────────────────────────────────

class FrameGrabber(threading.Thread):
    """
    Reads frames from RTSP/file/USB source continuously in its own thread.
    For 'browser' source, frames are fed via feed_frame() from the Flask endpoint.
    Always keeps only the newest frame so workers never process stale data.
    Auto-reconnects on RTSP drop; loops video files if loop=True.
    """

    def __init__(self, url, cam_id, loop=False, source="file"):
        super().__init__(daemon=True, name=f"grabber-{cam_id}")
        self.url = url
        self.cam_id = cam_id
        self.loop = loop
        self.source = source
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.connected = False

    def latest(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def feed_frame(self, jpeg_bytes):
        """For 'browser' source: receive a JPEG frame from the Flask endpoint."""
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(frame)

    def run(self):
        if self.source == "browser":
            # Browser source: no capture loop; frames arrive via feed_frame()
            self.connected = True
            log.info(f"[Cam {self.cam_id}] Browser source — waiting for frames")
            while not self._stop.is_set():
                time.sleep(0.1)
            return

        while not self._stop.is_set():
            if self.source == "usb":
                cap = cv2.VideoCapture(int(self.url))
            else:
                cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                log.warning(f"[Cam {self.cam_id}] Cannot open {self.url!r} — retrying in 3 s")
                time.sleep(3)
                continue

            self.connected = True
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            is_file = self.source == "file"
            frame_interval = 1.0 / src_fps if is_file else 0
            log.info(f"[Cam {self.cam_id}] Connected — {src_fps:.0f} fps source ({self.source})")

            while not self._stop.is_set():
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.warning(f"[Cam {self.cam_id}] Stream ended — reconnecting")
                    break
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


# ── FrameBuffer ───────────────────────────────────────────────────────────

class FrameBuffer:
    """Thread-safe slot holding the latest annotated JPEG, counts, and annotation data."""

    def __init__(self, label):
        self.label = label
        self.active_classes: list[int] = []
        self.running = True
        self._lock = threading.Lock()
        self._jpeg = None
        self._counts: dict[int, int] = {}
        self._annotations: dict | None = None
        self._seq = 0

    def put(self, jpeg_bytes, counts, annotations=None):
        with self._lock:
            self._jpeg = jpeg_bytes
            self._counts = dict(counts)
            if annotations is not None:
                self._annotations = annotations
            self._seq += 1

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_jpeg_if_new(self, last_seq):
        with self._lock:
            if self._seq == last_seq or self._jpeg is None:
                return None, last_seq
            return self._jpeg, self._seq

    def get_annotations(self):
        with self._lock:
            return self._annotations

    def reset_counts(self, counts):
        with self._lock:
            self._counts = dict(counts)

    def get_counts(self):
        with self._lock:
            return dict(self._counts)


# ── CameraWorker ──────────────────────────────────────────────────────────

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
        self.line_x_frac = cam_cfg.get("line_pos", cam_cfg.get("line_x", 0.5))
        self.line_orientation = cam_cfg.get("line_orientation", "vertical")
        self.line_direction = cam_cfg.get("line_direction", "forward")
        self.model_path = global_cfg["model"]
        self.tracker_path = global_cfg["tracker"]
        self.conf = global_cfg.get("conf", 0.25)
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
        self._paused.set()
        buf.active_classes = self.active_classes
        buf.running = False

    def pause(self):
        self._paused.set()
        self.buf.running = False

    def resume(self):
        self._paused.clear()
        self.buf.running = True

    def reset(self):
        with self._lock:
            self._counts = {cls: 0 for cls in self.active_classes}
            # Clear crossing state (per-track keys are dynamic)
            for k in list(self._state.keys()):
                if k != "min_travel" and k != "track_class":
                    del self._state[k]
        self.buf.reset_counts(self._counts)

    def run(self):
        time.sleep(self.cam_id * 0.5)
        log.info(f"[Cam {self.cam_id}] Loading model: {self.model_path}")
        try:
            model = YOLO(self.model_path, task="detect")
            log.info(f"[Cam {self.cam_id}] Model loaded OK")
        except Exception as e:
            log.error(f"[Cam {self.cam_id}] Model load FAILED: {e}")
            raise
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
            line_px = int((self.line_orientation == "horizontal" and h or w) * self.line_x_frac)

            results = model.track(
                frame,
                persist=True,
                conf=self.conf,
                classes=self.active_classes,
                tracker=self.tracker_path,
                verbose=False,
            )

            annotated = results[0].plot()

            ann_boxes = []
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy().astype(int)
                clses = results[0].boxes.cls.cpu().numpy().astype(int)
                n = len(boxes)
                log.debug(f"[Cam {self.cam_id}] Detected {n} objects")
                for box, tid, cls in zip(boxes, ids, clses):
                    x1, y1, x2, y2 = box.tolist()
                    ann_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "id": int(tid), "class": int(cls),
                    })
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    if f"first_{tid}" not in self._state:
                        self._state["track_class"][tid] = int(cls)
                    if check_crossing(tid, cx, cy, line_px, self._state, self.line_orientation, self.line_direction):
                        with self._lock:
                            self._counts[self._state["track_class"][tid]] += 1
            else:
                log.debug(f"[Cam {self.cam_id}] No detections")

            cv2.putText(annotated, self.buf.label,
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y = 58
            for cls in self.active_classes:
                label = CLASS_NAMES[cls].replace("_", " ").title()
                color = CLASS_COLORS[cls]
                cv2.putText(annotated, f"{label}: {self._counts[cls]}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y += 28

            elapsed = time.time() - t0
            ftimes.append(elapsed)
            if len(ftimes) > 30:
                ftimes.pop(0)
            fps = 1.0 / (sum(ftimes) / len(ftimes))
            cv2.putText(annotated, f"FPS: {fps:.1f}",
                        (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            ann_data = {
                "boxes": ann_boxes,
                "line_x": self.line_x_frac,
                "line_pos": self.line_x_frac,
                "line_orientation": self.line_orientation,
                "line_direction": self.line_direction,
                "width": w,
                "height": h,
            }
            self.buf.put(jpeg.tobytes(), self._counts, ann_data)

            # Throttle YOLO to 5 FPS to save CPU
            remaining = 0.2 - elapsed
            while remaining > 0 and not self._stop.is_set():
                time.sleep(min(0.01, remaining))
                remaining -= 0.01

    def stop(self):
        self._stop.set()


# ── Dynamic camera management ─────────────────────────────────────────────

app = Flask(__name__)
buffers: list[FrameBuffer] = []
workers: list[CameraWorker] = []
grabbers: list[FrameGrabber] = []
camera_configs: list[dict] = []
_cam_lock = threading.Lock()
_global_cfg: dict = {}


def _make_camera(i, cam_cfg, global_cfg):
    """Create grabber + worker + buffer for one camera (does not start threads)."""
    url   = cam_cfg["url"]
    label = cam_cfg.get("label", f"Camera {i}")
    loop  = cam_cfg.get("loop", False)
    source = cam_cfg.get("source", "file")

    buf     = FrameBuffer(label)
    grabber = FrameGrabber(url, i, loop=loop, source=source)
    worker  = CameraWorker(i, grabber, buf, cam_cfg, global_cfg)
    return buf, grabber, worker


def _start_camera(i, cam_cfg, global_cfg):
    """Create, append to globals, and start grabber + worker threads."""
    buf, grabber, worker = _make_camera(i, cam_cfg, global_cfg)
    buffers.append(buf)
    grabbers.append(grabber)
    workers.append(worker)
    grabber.start()
    worker.start()
    return buf, grabber, worker


def _remove_camera(i):
    """Stop and remove camera threads + buffers at index i."""
    if i < len(workers):
        workers[i].stop()
        grabbers[i].stop()
    # Remove from lists (order matters — pop larger index first is safer but
    # we always remove in reverse order from the caller).
    if i < len(buffers):
        buffers.pop(i)
    if i < len(grabbers):
        grabbers.pop(i)
    if i < len(workers):
        workers.pop(i)


def sync_cameras(incoming: list[dict], global_cfg: dict):
    """
    Replace the server camera list with `incoming`.
    Adds new cameras, removes deleted ones, restarts changed ones.
    Called from the /cameras/sync endpoint and optionally from main().
    """
    global camera_configs
    with _cam_lock:
        old_count = len(buffers)
        new_count = len(incoming)

        # Stop removed cameras (iterate in reverse so indices stay valid)
        for i in range(old_count - 1, new_count - 1, -1):
            _remove_camera(i)

        # Resize lists to match new count
        while len(buffers) < new_count:
            buffers.append(None)
            grabbers.append(None)
            workers.append(None)

        for i, cam_cfg in enumerate(incoming):
            existing = camera_configs[i] if i < len(camera_configs) else None
            if existing != cam_cfg:
                if existing is not None:
                    if i < old_count:
                        workers[i].stop()
                        grabbers[i].stop()
                buf, grabber, worker = _make_camera(i, cam_cfg, global_cfg)
                buffers[i] = buf
                grabbers[i] = grabber
                workers[i] = worker
                grabber.start()
                worker.start()

        camera_configs = list(incoming)


# ── Flask routes ─────────────────────────────────────────────────────────

_TICK = 0.005


def _mjpeg(cam_id):
    last_seq = -1
    placeholder = None
    try:
        while True:
            t0 = time.monotonic()
            jpeg, last_seq = buffers[cam_id].get_jpeg_if_new(last_seq)
            if jpeg is None:
                if placeholder is None:
                    blank = np.zeros((200, 320, 3), dtype=np.uint8)
                    _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 30])
                    placeholder = buf.tobytes()
                jpeg = placeholder
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            elapsed = time.monotonic() - t0
            remaining = MJPEG_INTERVAL - elapsed
            while remaining > 0:
                time.sleep(min(_TICK, remaining))
                remaining -= _TICK
    except GeneratorExit:
        pass


def _sse_counts():
    while True:
        counts_snapshot = {}
        with _cam_lock:
            for i, b in enumerate(buffers):
                if b is None:
                    continue
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
    with _cam_lock:
        cams = [{"id": i, "label": b.label} for i, b in enumerate(buffers) if b is not None]
    return render_template("dashboard.html", cameras=cams)


@app.route("/stream/<int:cam_id>")
def stream(cam_id):
    if cam_id >= len(buffers) or buffers[cam_id] is None:
        return "Not found", 404
    return Response(
        _mjpeg(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

_FRAME_PLACEHOLDER: bytes | None = None

@app.route("/frame/<int:cam_id>")
def single_frame(cam_id):
    """Return the latest annotated frame as a single JPEG (canvas-friendly)."""
    if cam_id >= len(buffers) or buffers[cam_id] is None:
        return "Not found", 404
    global _FRAME_PLACEHOLDER
    jpeg = buffers[cam_id].get_jpeg()
    if jpeg is not None:
        return Response(jpeg, mimetype="image/jpeg")
    if _FRAME_PLACEHOLDER is None:
        blank = np.zeros((200, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 30])
        _FRAME_PLACEHOLDER = buf.tobytes()
    return Response(_FRAME_PLACEHOLDER, mimetype="image/jpeg")


@app.route("/counts")
def counts_sse():
    return Response(
        stream_with_context(_sse_counts()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/reset/<int:cam_id>", methods=["POST"])
def reset_cam(cam_id):
    if cam_id >= len(workers) or workers[cam_id] is None:
        return {"error": "not found"}, 404
    workers[cam_id].reset()
    return {"status": "ok", "cam_id": cam_id}


@app.route("/stop/<int:cam_id>", methods=["POST"])
def stop_cam(cam_id):
    if cam_id >= len(workers) or workers[cam_id] is None:
        return {"error": "not found"}, 404
    workers[cam_id].pause()
    return {"status": "ok", "running": False}


@app.route("/start/<int:cam_id>", methods=["POST"])
def start_cam(cam_id):
    if cam_id >= len(workers) or workers[cam_id] is None:
        return {"error": "not found"}, 404
    workers[cam_id].resume()
    return {"status": "ok", "running": True}


@app.route("/camera/<int:cam_id>/frame", methods=["POST"])
def camera_frame(cam_id):
    """Receive a JPEG frame from a browser-based camera."""
    data = request.get_json(force=True)
    jpeg_bytes = base64.b64decode(data["frame"])
    with _cam_lock:
        if cam_id >= len(grabbers) or grabbers[cam_id] is None:
            return {"error": "not found"}, 404
        grabbers[cam_id].feed_frame(jpeg_bytes)
    return {"status": "ok"}


@app.route("/cameras/sync", methods=["POST"])
def cameras_sync():
    global _global_cfg
    incoming = request.get_json(force=True)
    if not isinstance(incoming, list):
        return {"error": "expected array"}, 400
    sync_cameras(incoming, _global_cfg)
    return {"status": "ok", "count": len(incoming)}


@app.route("/cameras/list")
def cameras_list():
    with _cam_lock:
        return {"cameras": list(camera_configs)}


@app.route("/annotations/<int:cam_id>")
def get_annotations(cam_id):
    if cam_id >= len(buffers) or buffers[cam_id] is None:
        return {"error": "not found"}, 404
    ann = buffers[cam_id].get_annotations()
    if ann is None:
        return {"boxes": [], "line_x": 0.5, "width": 0, "height": 0}
    return ann


@app.route("/camera/<int:cam_id>/line", methods=["POST"])
def update_line(cam_id):
    data = request.get_json(force=True)
    line_x = float(data.get("line_pos", data.get("line_x", 0.5)))
    orient = data.get("line_orientation", "vertical")
    direc = data.get("line_direction", "forward")
    with _cam_lock:
        if cam_id >= len(workers) or workers[cam_id] is None:
            return {"error": "not found"}, 404
        workers[cam_id].line_x_frac = max(0.0, min(1.0, line_x))
        if orient in ("vertical", "horizontal"):
            workers[cam_id].line_orientation = orient
        if direc in ("forward", "backward"):
            workers[cam_id].line_direction = direc
    return {"status": "ok", "line_x": workers[cam_id].line_x_frac,
            "line_orientation": workers[cam_id].line_orientation,
            "line_direction": workers[cam_id].line_direction}


@app.route("/health")
def health():
    with _cam_lock:
        return {"status": "ok", "cameras": len([b for b in buffers if b is not None])}


LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")


@app.route("/logs/clear", methods=["POST"])
def clear_logs():
    try:
        open(LOG_FILE, "w").close()
        log.info("Logs cleared via /logs/clear")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


@app.route("/logs")
def view_logs():
    lines = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                level = ""
                cls = ""
                if "[ERROR]" in line:   cls = "log-err"
                elif "[WARNING]" in line: cls = "log-warn"
                elif "[INFO]" in line:  cls = "log-info"
                lines.append(f"<tr><td><code class=\"{cls}\">{line}</code></td></tr>")
    except FileNotFoundError:
        lines.append('<tr><td><em>server.log not found yet</em></td></tr>')
    rows = "\n".join(lines[-500:])  # last 500 lines
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Vehicle Counter — Logs</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; color:#e6edf3; font-family:monospace; font-size:13px; padding:16px; }}
  h1 {{ font-size:1.1rem; margin-bottom:12px; color:#8b949e; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:2px 8px; border-bottom:1px solid #21262d; }}
  code {{ white-space:pre; }}
  .log-err  {{ color:#f85149; }}
  .log-warn {{ color:#d29922; }}
  .log-info {{ color:#e6edf3; }}
  a {{ color:#58a6ff; text-decoration:none; font-size:0.85rem; }}
</style></head>
<body>
<h1>Server Log <a href="/logs" style="margin-left:12px">&#x21bb;</a>
  <a href="/" style="margin-left:8px">&#x2190; Dashboard</a>
  <button onclick="fetch('/logs/clear',{{method:'POST'}}).then(()=>location.reload())"
    style="margin-left:12px;background:#f85149;color:#fff;border:none;border-radius:4px;padding:2px 10px;cursor:pointer;font-size:0.8rem">Clear</button></h1>
<table>{rows}</table>
<script>setTimeout(function(){{ location.reload(); }}, 3000);</script>
</body></html>"""
    return html


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to cameras.yaml (optional)")
    parser.add_argument("--port", type=int, default=None, help="Override port from config")
    args = parser.parse_args()

    global _global_cfg

    _global_cfg = {
        "model":   os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcmodel1.onnx"),
        "tracker": os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml"),
        "conf":    0.25,
        "show_classes": ["haul_truck"],
    }

    host = "0.0.0.0"
    port = 5000

    # Optionally load initial camera config from YAML
    if args.config:
        cfg_path = args.config
    else:
        default_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras.yaml")
        if os.path.exists(default_yaml):
            cfg_path = default_yaml
        else:
            cfg_path = None

    if cfg_path:
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f)

        # Merge YAML values into global config
        if "model" in yaml_cfg:
            _global_cfg["model"] = yaml_cfg["model"]
        if "tracker" in yaml_cfg:
            _global_cfg["tracker"] = yaml_cfg["tracker"]
        if "conf" in yaml_cfg:
            _global_cfg["conf"] = yaml_cfg["conf"]
        if "show_classes" in yaml_cfg:
            _global_cfg["show_classes"] = yaml_cfg["show_classes"]
        if "host" in yaml_cfg:
            host = yaml_cfg["host"]
        if "port" in yaml_cfg:
            port = yaml_cfg["port"]

        # Resolve tracker path relative to config dir
        if not os.path.isabs(_global_cfg["tracker"]):
            _global_cfg["tracker"] = os.path.join(cfg_dir, _global_cfg["tracker"])
        if not os.path.isabs(_global_cfg["model"]):
            _global_cfg["model"] = os.path.join(cfg_dir, _global_cfg["model"])

        # Load cameras from YAML as initial config
        if "cameras" in yaml_cfg:
            # Add default source type if not specified
            for cam in yaml_cfg["cameras"]:
                if "source" not in cam:
                    url = cam["url"]
                    if str(url).lstrip("-").isdigit():
                        cam["source"] = "usb"
                    elif str(url).startswith("rtsp"):
                        cam["source"] = "rtsp"
                    else:
                        cam["source"] = "file"
            sync_cameras(yaml_cfg["cameras"], _global_cfg)

    port = args.port or port

    # Startup checks
    model_path = _global_cfg["model"]
    if not os.path.exists(model_path):
        log.error(f"Model file not found: {model_path}")
        log.error(f"Place vcmodel1.onnx at: {model_path} (mount via docker-compose.yml or COPY in Dockerfile)")
    if not os.path.exists("videos"):
        log.warning("Directory 'videos/' not found — create it and place .mp4 files inside, or use RTSP URLs")
    else:
        mp4s = [f for f in os.listdir("videos") if f.endswith(".mp4")]
        if not mp4s:
            log.warning("videos/ exists but no .mp4 files found — place video files or use RTSP URLs")

    log.info(f"Dashboard → http://localhost:{port}/")
    log.info(f"Model     : {model_path}")
    log.info(f"Tracker   : {_global_cfg['tracker']}")
    cam_count = len([b for b in buffers if b is not None])
    log.info(f"Cameras   : {cam_count} (add/remove from browser UI)")
    print()
    print(f"  Dashboard → http://localhost:{port}/")

    thread_count = max(32, (cam_count + 1) * 4)
    serve(app, host=host, port=port, threads=thread_count)


if __name__ == "__main__":
    main()
