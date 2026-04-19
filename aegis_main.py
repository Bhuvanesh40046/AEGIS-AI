"""
=============================================================================
 AEGIS.AI - Complete Integrated Worker Safety System (FIXED)
=============================================================================
 Priority: P1 Gas/Temp > P2 Zone > P3 Helmet
 Usage:
   python aegis_main.py --model best.pt --camera 0
   python aegis_main.py --model best.pt --camera 0 --no-esp32
 Controls: Z=Zone | R=Reset | Q=Quit | D=Debug | S=Screenshot | M=Mute
=============================================================================
"""

import cv2
import numpy as np
import time
import argparse
import json
import os
import threading
from collections import defaultdict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from ultralytics import YOLO
except ImportError:
    print("Run: python -m pip install ultralytics")
    raise

TTS_AVAILABLE = False
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("[WARN] pyttsx3 not installed.")


class Config:
    CUSTOM_MODEL_PATH = "best.pt"
    CONFIDENCE_THRESHOLD = 0.25
    HELMET_CLASS_ID = 2
    HEAD_CLASS_ID = 1

    # Cooldowns
    HELMET_COOLDOWN = 15
    ZONE_COOLDOWN = 10
    GAS_COOLDOWN = 8
    TEMP_COOLDOWN = 10

    # ESP32 — FIXED THRESHOLDS based on your sensor readings
    # Your MQ sensor reads ~1100-1300 in clean air
    # So warning should be much higher than that
    ESP32_PORT = 5000
    TEMP_WARNING = 40.0
    TEMP_DANGER = 50.0
    GAS_RAW_WARNING = 2000    # FIXED: was 600, normal air reads ~1100-1300
    GAS_RAW_DANGER = 2800     # FIXED: was 800, only real gas triggers this

    WINDOW_NAME = "AEGIS.AI - Integrated Safety System"
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    COLOR_SAFE = (0, 200, 100)
    COLOR_DANGER = (0, 0, 240)
    COLOR_WARNING = (0, 180, 255)
    COLOR_ZONE = (255, 200, 0)
    COLOR_ZONE_VIOLATION = (0, 0, 255)
    COLOR_HELMET_BOX = (200, 180, 0)
    COLOR_HEAD_BOX = (0, 100, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_CYAN = (200, 180, 0)


# ── Helpers ──────────────────────────────────────────────────────────────

def box_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

def point_in_box(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

def box_distance(a, b):
    c1, c2 = box_center(a), box_center(b)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

def get_head_region(b, r=0.45):
    return [b[0], b[1], b[2], b[1] + (b[3] - b[1]) * r]

def calculate_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0

def associate_helmets_to_persons(persons, helmets, heads):
    results = []
    for p in persons:
        hr = get_head_region(p['box'])
        ph = p['box'][3] - p['box'][1]
        px = ph * 0.6
        hi = any(point_in_box(box_center(h['box']), hr) for h in helmets)
        hdi = any(point_in_box(box_center(h['box']), hr) for h in heads)
        bhi = max([calculate_iou(hr, h['box']) for h in helmets], default=0)
        bhdi = max([calculate_iou(hr, h['box']) for h in heads], default=0)
        ch = min([box_distance(hr, h['box']) for h in helmets], default=float('inf'))
        chd = min([box_distance(hr, h['box']) for h in heads], default=float('inf'))
        hip = any(point_in_box(box_center(h['box']), p['box']) for h in helmets)
        hdip = any(point_in_box(box_center(h['box']), p['box']) for h in heads)

        if hi or bhi > 0.05: s = "helmet"
        elif hdi or bhdi > 0.05: s = "no_helmet"
        elif ch < px and ch < chd: s = "helmet"
        elif chd < px and chd < ch: s = "no_helmet"
        elif hip: s = "helmet"
        elif hdip: s = "no_helmet"
        else: s = "no_helmet"
        results.append({**p, 'helmet_status': s, 'head_region': hr})
    return results


# ── Zone Manager ─────────────────────────────────────────────────────────

class ZoneManager:
    def __init__(self):
        self.points = []
        self.is_calibrated = False
        self.calibrating = False

    def start_calibration(self):
        self.calibrating = True
        self.points = []
        self.is_calibrated = False
        print("\n" + "=" * 50)
        print("  ZONE CALIBRATION - Click corners, ENTER to finish")
        print("=" * 50)

    def on_mouse(self, event, x, y, flags, param):
        if self.calibrating and event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")

    def finish(self):
        if len(self.points) >= 3:
            self.is_calibrated = True
            self.calibrating = False
            print(f"  Zone set with {len(self.points)} corners!")
            return True
        print("  Need 3+ points!")
        return False

    def cancel(self):
        self.calibrating = False
        self.points = []

    def reset(self):
        self.is_calibrated = False
        self.calibrating = False
        self.points = []
        print("[ZONE] Reset")

    def is_inside(self, pt):
        if not self.is_calibrated:
            return True
        poly = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    def check_worker(self, box):
        foot = (int((box[0] + box[2]) / 2), int(box[3]))
        return self.is_inside(foot), foot

    def draw(self, frame):
        h, w = frame.shape[:2]
        if self.calibrating:
            for i, pt in enumerate(self.points):
                cv2.circle(frame, pt, 8, Config.COLOR_ZONE, -1)
                cv2.circle(frame, pt, 11, Config.COLOR_ZONE, 2)
                if i > 0:
                    cv2.line(frame, self.points[i - 1], pt, Config.COLOR_ZONE, 2)
            if len(self.points) >= 3:
                cv2.line(frame, self.points[-1], self.points[0], Config.COLOR_ZONE, 1)
            ov = frame.copy()
            cv2.rectangle(ov, (0, h - 50), (w, h), Config.COLOR_BLACK, -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame,
                        f"CALIBRATING | Points:{len(self.points)} | Click corners, ENTER=done, C=cancel",
                        (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_ZONE, 1)
        elif self.is_calibrated:
            pts = np.array(self.points, dtype=np.int32)
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], (0, 100, 0))
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, Config.COLOR_ZONE, 2)
            cv2.putText(frame, "SAFE ZONE",
                        (self.points[0][0], self.points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_ZONE, 2)


# ── ESP32 Receiver ───────────────────────────────────────────────────────

class SensorData:
    def __init__(self):
        self.zones = {}
        self.lock = threading.Lock()

    def update(self, zid, data):
        with self.lock:
            self.zones[zid] = {**data, 'timestamp': time.time()}

    def get_all(self):
        with self.lock:
            return dict(self.zones)

    def is_stale(self, zid, t=10):
        with self.lock:
            if zid not in self.zones:
                return True
            return (time.time() - self.zones[zid]['timestamp']) > t


class ESP32Handler(BaseHTTPRequestHandler):
    sensor_data = None

    def do_POST(self):
        try:
            body = self.rfile.read(int(self.headers.get('Content-Length', 0))).decode()
            data = json.loads(body)
            self.sensor_data.update(data.get('zone', '?'), data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except:
            self.send_response(400)
            self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(self.sensor_data.get_all(), default=str).encode())

    def log_message(self, *a):
        pass


def start_esp32_server(sd, port=Config.ESP32_PORT):
    ESP32Handler.sensor_data = sd
    srv = HTTPServer(('0.0.0.0', port), ESP32Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[ESP32] Server on port {port}")


# ── Voice Engine (FIXED — runs on main concepts, no threading issues) ────

class VoiceEngine:
    """
    FIXED: pyttsx3 crashes in background threads.
    Now we queue messages and speak them from the main loop
    with a time gap so it doesn't block video too long.
    """

    def __init__(self):
        self.engine = None
        self.muted = False
        self.pending_msg = None
        self.last_speak_time = 0
        self.speak_interval = 3  # minimum seconds between voice alerts

        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 170)
                self.engine.setProperty('volume', 1.0)
                voices = self.engine.getProperty('voices')
                if len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
                print("[TTS] Voice engine ready.")
            except Exception as e:
                print(f"[TTS] Failed: {e}")

    def queue_alert(self, msg, priority):
        """Queue a message. Higher priority (lower number) replaces lower priority."""
        if self.muted or not self.engine:
            return
        now = time.time()
        if now - self.last_speak_time < self.speak_interval:
            return
        # Always accept P1, only accept P2/P3 if no pending
        if self.pending_msg is None or priority <= self.pending_priority:
            self.pending_msg = msg
            self.pending_priority = priority

    def speak_if_ready(self):
        """Call this from main loop. Speaks pending message if cooldown passed."""
        if not self.engine or self.muted or self.pending_msg is None:
            return
        now = time.time()
        if now - self.last_speak_time < self.speak_interval:
            return
        msg = self.pending_msg
        self.pending_msg = None
        self.last_speak_time = now
        try:
            self.engine.say(msg)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Error: {e}")
            # Reinitialize engine if it crashes
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 170)
                self.engine.setProperty('volume', 1.0)
            except:
                pass


# ── Alert Tracker ────────────────────────────────────────────────────────

class AlertTracker:
    def __init__(self):
        self.last_alert = {}
        self.alert_count = defaultdict(int)
        self.alert_log = []

    def should_alert(self, aid, cooldown):
        return (time.time() - self.last_alert.get(aid, 0)) >= cooldown

    def record(self, priority, aid, msg, cooldown):
        if not self.should_alert(aid, cooldown):
            return False
        self.last_alert[aid] = time.time()
        self.alert_count[aid] += 1
        self.alert_log.append({
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'id': aid,
            'message': msg,
        })
        return True


# ── Main App ─────────────────────────────────────────────────────────────

class AegisApp:
    def __init__(self, camera=0, model_path=None, use_esp32=True, debug=False):
        self.camera = camera
        self.model_path = model_path or Config.CUSTOM_MODEL_PATH
        self.use_esp32 = use_esp32
        self.debug = debug
        self.zone = ZoneManager()
        self.sd = SensorData()
        self.voice = VoiceEngine()
        self.alerts = AlertTracker()
        self.gas_emergency = False
        self.sensor_alerts = []
        self.fps = 0
        self.fc = 0
        self.ft = time.time()

    def load_models(self):
        print("[MODEL 1] Loading YOLOv8n (COCO)...")
        self.pm = YOLO("yolov8n.pt")
        print("[MODEL 1] OK!")
        self.has_hm = False
        if os.path.exists(self.model_path):
            print(f"[MODEL 2] Loading {self.model_path}...")
            self.hm = YOLO(self.model_path)
            print(f"[MODEL 2] Classes: {self.hm.names}")
            self.has_hm = True
        else:
            print(f"[MODEL 2] {self.model_path} not found")

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.DISPLAY_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.DISPLAY_HEIGHT)
        print(f"[CAMERA] {int(self.cap.get(3))}x{int(self.cap.get(4))}")

    def detect_persons(self, frame):
        r = self.pm.track(source=frame, conf=0.35, classes=[0],
                          tracker="bytetrack.yaml", persist=True, verbose=False)
        ps = []
        if r and r[0].boxes and len(r[0].boxes) > 0:
            for i in range(len(r[0].boxes)):
                b = r[0].boxes.xyxy[i].cpu().numpy().tolist()
                c = float(r[0].boxes.conf[i].cpu().numpy())
                t = int(r[0].boxes.id[i].cpu().numpy()) if r[0].boxes.id is not None else i
                ps.append({'box': b, 'id': t, 'conf': c})
        return ps

    def detect_helmets(self, frame):
        hs, hds = [], []
        if not self.has_hm:
            return hs, hds
        r = self.hm(frame, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
        if r and r[0].boxes and len(r[0].boxes) > 0:
            for i in range(len(r[0].boxes)):
                b = r[0].boxes.xyxy[i].cpu().numpy().tolist()
                cl = int(r[0].boxes.cls[i].cpu().numpy())
                c = float(r[0].boxes.conf[i].cpu().numpy())
                if cl == Config.HELMET_CLASS_ID:
                    hs.append({'box': b, 'conf': c})
                elif cl == Config.HEAD_CLASS_ID:
                    hds.append({'box': b, 'conf': c})
        return hs, hds

    def check_sensors(self):
        """Check ESP32 data. FIXED thresholds so normal air doesn't trigger."""
        self.sensor_alerts = []
        self.gas_emergency = False
        zones = self.sd.get_all()

        for zid, d in zones.items():
            t = d.get('temp', 0)
            g = d.get('gas_raw', 0)

            # Gas (P1)
            if g > Config.GAS_RAW_DANGER:
                self.gas_emergency = True
                if self.alerts.record(1, f"gas_d_{zid}", f"GAS LEAK {zid}!", Config.GAS_COOLDOWN):
                    self.voice.queue_alert("EMERGENCY! Gas leak detected! All workers evacuate immediately!", 1)
                self.sensor_alerts.append(('danger', f"GAS LEAK {zid}! EVACUATE! (Raw:{g})"))
            elif g > Config.GAS_RAW_WARNING:
                if self.alerts.record(1, f"gas_w_{zid}", f"Gas elevated {zid}", Config.GAS_COOLDOWN):
                    self.voice.queue_alert(f"Warning! Elevated gas levels in {zid}.", 1)
                self.sensor_alerts.append(('warning', f"Gas elevated {zid} (Raw:{g})"))

            # Temperature (P1 danger, P2 warning)
            if t > Config.TEMP_DANGER:
                self.gas_emergency = True
                if self.alerts.record(1, f"tmp_d_{zid}", f"TEMP {t:.0f}C {zid}!", Config.TEMP_COOLDOWN):
                    self.voice.queue_alert(f"DANGER! Temperature {t:.0f} degrees in {zid}! Evacuate!", 1)
                self.sensor_alerts.append(('danger', f"TEMP {t:.1f}C in {zid}!"))
            elif t > Config.TEMP_WARNING:
                if self.alerts.record(2, f"tmp_w_{zid}", f"Temp high {zid}", Config.TEMP_COOLDOWN):
                    self.voice.queue_alert(f"Warning. High temperature at {t:.0f} degrees in {zid}.", 2)
                self.sensor_alerts.append(('warning', f"Temp {t:.1f}C in {zid}"))

    def draw_sensor_panel(self, frame):
        zones = self.sd.get_all()
        if not zones:
            return
        h, w = frame.shape[:2]
        pw, px, py = 210, w - 220, 65
        ph = 20 + len(zones) * 85

        ov = frame.copy()
        cv2.rectangle(ov, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "ESP32 SENSORS", (px + 10, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_CYAN, 1)

        y = py + 38
        for zid, d in zones.items():
            t = d.get('temp', 0)
            hu = d.get('humidity', 0)
            g = d.get('gas_raw', 0)
            tc = Config.COLOR_DANGER if t > Config.TEMP_DANGER else \
                Config.COLOR_WARNING if t > Config.TEMP_WARNING else Config.COLOR_SAFE
            gc = Config.COLOR_DANGER if g > Config.GAS_RAW_DANGER else \
                Config.COLOR_WARNING if g > Config.GAS_RAW_WARNING else Config.COLOR_SAFE

            cv2.putText(frame, zid, (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_WHITE, 1)
            if self.sd.is_stale(zid):
                cv2.putText(frame, "[STALE]", (px + 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, Config.COLOR_DANGER, 1)
            cv2.putText(frame, f"Temp: {t:.1f}C", (px + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tc, 1)
            cv2.putText(frame, f"Hum: {hu:.0f}%", (px + 120, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, Config.COLOR_WHITE, 1)
            cv2.putText(frame, f"Gas: {g}", (px + 10, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, gc, 1)

            # Gas bar
            bx, bw = px + 70, 120
            bf = min(g / 3500, 1.0)
            cv2.rectangle(frame, (bx, y + 30), (bx + bw, y + 38), (50, 50, 50), -1)
            cv2.rectangle(frame, (bx, y + 30), (bx + int(bw * bf), y + 38), gc, -1)
            y += 75

    def run(self):
        self.load_models()
        self.open_camera()
        if self.use_esp32:
            start_esp32_server(self.sd)

        cv2.namedWindow(Config.WINDOW_NAME)
        cv2.setMouseCallback(Config.WINDOW_NAME, self.zone.on_mouse)

        print("\n" + "=" * 60)
        print("  AEGIS.AI - INTEGRATED WORKER SAFETY SYSTEM")
        print(f"  Helmet [ON] | Zone [ON] | ESP32 [{'ON' if self.use_esp32 else 'OFF'}] | Orchestrator [ON]")
        print("=" * 60)
        print("  Z=Zone | R=Reset | Q=Quit | D=Debug | S=Screenshot | M=Mute")
        print("=" * 60 + "\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]

            # Detect
            persons = self.detect_persons(frame)
            helmets, heads = self.detect_helmets(frame)
            if self.has_hm:
                dets = associate_helmets_to_persons(persons, helmets, heads)
            else:
                dets = [{**p, 'helmet_status': 'unknown', 'head_region': get_head_region(p['box'])} for p in persons]

            # Check sensors (P1)
            if self.use_esp32:
                self.check_sensors()

            # Check violations
            hv = 0
            zv = []
            for d in dets:
                # Helmet (P3)
                if d['helmet_status'] == 'no_helmet':
                    hv += 1
                    if not self.gas_emergency:
                        if self.alerts.record(3, f"helm_{d['id']}", f"No helmet P-{d['id']}", Config.HELMET_COOLDOWN):
                            self.voice.queue_alert(f"Attention! Worker {d['id']}, please wear your safety helmet.", 3)

                # Zone (P2)
                if self.zone.is_calibrated:
                    inside, foot = self.zone.check_worker(d['box'])
                    if not inside:
                        zv.append(d['id'])
                        if not self.gas_emergency:
                            if self.alerts.record(2, f"zone_{d['id']}", f"Zone violation P-{d['id']}", Config.ZONE_COOLDOWN):
                                self.voice.queue_alert(f"Warning! Worker {d['id']}, return to safe zone immediately.", 2)

            # SPEAK pending voice alert (FIXED: from main thread)
            self.voice.speak_if_ready()

            # ── Draw ────────────────────────────────────────────────
            self.zone.draw(frame)

            # Gas emergency flash
            if self.gas_emergency and int(time.time() * 3) % 2:
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 100), -1)
                cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)

            for d in dets:
                bx = [int(v) for v in d['box']]
                st = d['helmet_status']
                tid = d.get('id', '?')
                zf = tid in zv

                lb = []
                if st == "helmet":
                    co = Config.COLOR_SAFE
                    lb.append("HELMET OK")
                elif st == "no_helmet":
                    co = Config.COLOR_DANGER
                    lb.append("NO HELMET!")
                else:
                    co = Config.COLOR_WARNING
                    lb.append("...")

                if zf:
                    co = Config.COLOR_ZONE_VIOLATION
                    lb.append("ZONE!")

                la = f"P-{tid} [{' | '.join(lb)}]"
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), co, 2)
                (tw, th), _ = cv2.getTextSize(la, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (bx[0], bx[1] - th - 8), (bx[0] + tw + 4, bx[1]), co, -1)
                cv2.putText(frame, la, (bx[0] + 2, bx[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_WHITE, 1)

                if st == "no_helmet" or zf:
                    cx, cy = bx[2] - 12, bx[1] + 12
                    cv2.circle(frame, (cx, cy), 10, Config.COLOR_DANGER, -1)
                    cv2.putText(frame, "!", (cx - 4, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_WHITE, 2)

                if self.zone.is_calibrated:
                    fx = int((d['box'][0] + d['box'][2]) / 2)
                    fy = int(d['box'][3])
                    fc = Config.COLOR_ZONE_VIOLATION if zf else Config.COLOR_SAFE
                    cv2.circle(frame, (fx, fy), 5, fc, -1)

            if self.debug:
                for hl in helmets:
                    b = [int(v) for v in hl['box']]
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), Config.COLOR_HELMET_BOX, 1)
                for hd in heads:
                    b = [int(v) for v in hd['box']]
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), Config.COLOR_HEAD_BOX, 1)

            if self.use_esp32:
                self.draw_sensor_panel(frame)

            # ── HUD ─────────────────────────────────────────────────
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, "AEGIS.AI", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 2)
            cv2.putText(frame, "Integrated Worker Safety System", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            sx = w - 530
            cv2.putText(frame, f"FPS:{self.fps:.0f}", (sx, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
            cv2.putText(frame, f"Workers:{len(dets)}", (sx + 65, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_WHITE, 1)

            zt = "Zone:ON" if self.zone.is_calibrated else "Zone:OFF"
            zc = Config.COLOR_SAFE if self.zone.is_calibrated else (100, 100, 100)
            cv2.putText(frame, zt, (sx + 165, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, zc, 1)

            esp_on = self.use_esp32 and bool(self.sd.get_all())
            cv2.putText(frame, "ESP:ON" if esp_on else "ESP:--", (sx + 260, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        Config.COLOR_SAFE if esp_on else (100, 100, 100), 1)

            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (sx + 460, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Status
            tv = hv + len(zv)
            if self.gas_emergency:
                cv2.putText(frame, "!!! GAS EMERGENCY - EVACUATE !!!", (sx, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_DANGER, 2)
            elif tv > 0:
                pts = []
                if hv: pts.append(f"{hv} helmet")
                if zv: pts.append(f"{len(zv)} zone")
                cv2.putText(frame, "VIOLATIONS: " + " | ".join(pts), (sx, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_DANGER, 1)
            else:
                cv2.putText(frame, "ALL CLEAR", (sx, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_SAFE, 1)

            # Bottom alerts
            if self.sensor_alerts:
                bh = 10 + len(self.sensor_alerts) * 20
                ov2 = frame.copy()
                cv2.rectangle(ov2, (0, h - bh), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(ov2, 0.8, frame, 0.2, 0, frame)
                yy = h - bh + 16
                for at, msg in self.sensor_alerts:
                    ac = Config.COLOR_DANGER if at == 'danger' else Config.COLOR_WARNING
                    cv2.putText(frame, f"[ESP32] {msg}", (10, yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1)
                    yy += 20
            elif tv > 0 and not self.gas_emergency:
                ov2 = frame.copy()
                cv2.rectangle(ov2, (0, h - 30), (w, h), (0, 0, 180), -1)
                cv2.addWeighted(ov2, 0.8, frame, 0.2, 0, frame)
                pts = []
                if hv: pts.append(f"{hv} no helmet")
                if zv: pts.append(f"{len(zv)} zone")
                cv2.putText(frame, "  WARNING: " + " | ".join(pts), (10, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_WHITE, 1)

            if not self.zone.calibrating and not self.sensor_alerts and tv == 0:
                cv2.putText(frame, "Z:Zone | R:Reset | Q:Quit | D:Debug | S:SS | M:Mute",
                            (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

            # FPS
            self.fc += 1
            if time.time() - self.ft >= 1.0:
                self.fps = self.fc / (time.time() - self.ft)
                self.fc = 0
                self.ft = time.time()

            cv2.imshow(Config.WINDOW_NAME, frame)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.zone.start_calibration()
            elif key == 13:
                if self.zone.calibrating:
                    self.zone.finish()
            elif key == ord('c'):
                if self.zone.calibrating:
                    self.zone.cancel()
            elif key == ord('r'):
                self.zone.reset()
            elif key == ord('d'):
                self.debug = not self.debug
                print(f"[DEBUG] {'ON' if self.debug else 'OFF'}")
            elif key == ord('s'):
                fn = f"ss_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(fn, frame)
                print(f"[SS] {fn}")
            elif key == ord('m'):
                self.voice.muted = not self.voice.muted
                print(f"[AUDIO] {'MUTED' if self.voice.muted else 'UNMUTED'}")

        self.cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total alerts: {len(self.alerts.alert_log)}")
        p1 = [a for a in self.alerts.alert_log if a['priority'] == 1]
        p2 = [a for a in self.alerts.alert_log if a['priority'] == 2]
        p3 = [a for a in self.alerts.alert_log if a['priority'] == 3]
        print(f"  P1 (Gas/Temp): {len(p1)}  |  P2 (Zone): {len(p2)}  |  P3 (Helmet): {len(p3)}")
        print("=" * 60)
        if self.alerts.alert_log:
            fn = f"aegis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fn, 'w') as f:
                json.dump(self.alerts.alert_log, f, indent=2)
            print(f"  Log: {fn}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--no-esp32', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--mute', action='store_true')
    a = p.parse_args()
    app = AegisApp(camera=a.camera, model_path=a.model,
                   use_esp32=not a.no_esp32, debug=a.debug)
    if a.mute:
        app.voice.muted = True
    app.run()


if __name__ == "__main__":
    main()
