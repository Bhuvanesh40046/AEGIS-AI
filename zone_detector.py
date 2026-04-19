"""
=============================================================================
 AEGIS.AI — Phase 2: Zone Restriction (Standalone)
=============================================================================
 Detects persons using COCO YOLOv8 and checks if they are inside
 a user-defined safe zone. Voice alerts if someone exits the zone.
 
 Usage:
   python zone_detector.py                  # Default camera
   python zone_detector.py --camera 1       # External webcam

 Controls:
   Z = Start zone calibration (click corners, ENTER to finish)
   R = Reset zone
   Q = Quit
   M = Mute/Unmute voice
=============================================================================
"""

import cv2
import numpy as np
import time
import argparse
import json
from collections import defaultdict
from datetime import datetime

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
    print("[WARNING] pyttsx3 not installed.")


# ==========================================================================
#  CONFIGURATION
# ==========================================================================
class Config:
    WINDOW_NAME = "AEGIS.AI - Zone Restriction"
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    PERSON_CONFIDENCE = 0.35
    ZONE_ALERT_COOLDOWN = 10

    # Colors (BGR)
    COLOR_SAFE = (0, 200, 100)
    COLOR_DANGER = (0, 0, 240)
    COLOR_ZONE = (255, 200, 0)
    COLOR_ZONE_FILL = (0, 100, 0)
    COLOR_VIOLATION = (0, 0, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)


# ==========================================================================
#  ZONE MANAGER
# ==========================================================================
class ZoneManager:
    """Click-to-define safe zone polygon."""

    def __init__(self):
        self.points = []
        self.is_calibrated = False
        self.calibrating = False

    def start_calibration(self):
        self.calibrating = True
        self.points = []
        self.is_calibrated = False
        print()
        print("=" * 55)
        print("  ZONE CALIBRATION MODE")
        print("  Click corners of SAFE ZONE on the video feed.")
        print("  Minimum 3 points needed.")
        print("  Press ENTER when done. Press C to cancel.")
        print("=" * 55)

    def mouse_callback(self, event, x, y, flags, param):
        if not self.calibrating:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")

    def finish(self):
        if len(self.points) >= 3:
            self.is_calibrated = True
            self.calibrating = False
            print(f"  Done! Safe zone has {len(self.points)} corners.")
            return True
        print("  Need at least 3 points! Keep clicking.")
        return False

    def cancel(self):
        self.calibrating = False
        self.points = []
        print("  Cancelled.")

    def reset(self):
        self.is_calibrated = False
        self.calibrating = False
        self.points = []
        print("[ZONE] Reset.")

    def is_inside(self, point):
        if not self.is_calibrated:
            return True
        polygon = np.array(self.points, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def check_worker(self, person_box):
        """Check bottom-center (feet) of person bounding box."""
        x1, y1, x2, y2 = person_box
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)
        inside = self.is_inside((foot_x, foot_y))
        return inside, (foot_x, foot_y)

    def draw(self, frame):
        if self.calibrating:
            for i, pt in enumerate(self.points):
                cv2.circle(frame, pt, 8, Config.COLOR_ZONE, -1)
                cv2.circle(frame, pt, 11, Config.COLOR_ZONE, 2)
                if i > 0:
                    cv2.line(frame, self.points[i-1], pt, Config.COLOR_ZONE, 2)
            if len(self.points) >= 3:
                cv2.line(frame, self.points[-1], self.points[0], Config.COLOR_ZONE, 1)

            # Bottom instruction bar
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0, h-50), (w, h), Config.COLOR_BLACK, -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame,
                        f"CALIBRATING | Points: {len(self.points)} | Click to add | ENTER = finish | C = cancel",
                        (10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_ZONE, 1)

        elif self.is_calibrated:
            pts = np.array(self.points, dtype=np.int32)
            # Green transparent fill
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], Config.COLOR_ZONE_FILL)
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
            # Yellow border
            cv2.polylines(frame, [pts], True, Config.COLOR_ZONE, 2)
            # Label
            cv2.putText(frame, "SAFE ZONE",
                        (self.points[0][0], self.points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_ZONE, 2)


# ==========================================================================
#  ALERT TRACKER
# ==========================================================================
class AlertTracker:
    def __init__(self):
        self.last_alert_time = {}
        self.alert_count = defaultdict(int)
        self.alert_log = []

    def should_alert(self, alert_id, cooldown):
        return (time.time() - self.last_alert_time.get(alert_id, 0)) >= cooldown

    def record_alert(self, alert_id, message):
        self.last_alert_time[alert_id] = time.time()
        self.alert_count[alert_id] += 1
        self.alert_log.append({
            "timestamp": datetime.now().isoformat(),
            "alert_id": str(alert_id),
            "message": message,
        })


# ==========================================================================
#  VOICE ENGINE
# ==========================================================================
class VoiceEngine:
    def __init__(self):
        self.engine = None
        self._busy = False
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 160)
                self.engine.setProperty('volume', 1.0)
                voices = self.engine.getProperty('voices')
                if len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
                print("[TTS] Voice engine ready.")
            except Exception as e:
                print(f"[TTS] Failed: {e}")

    def speak(self, msg):
        if not self.engine or self._busy:
            return
        try:
            self._busy = True
            self.engine.say(msg)
            self.engine.runAndWait()
        except:
            pass
        finally:
            self._busy = False

    def is_busy(self):
        return self._busy


# ==========================================================================
#  MAIN APPLICATION
# ==========================================================================
class ZoneDetector:
    def __init__(self, camera=0):
        self.camera = camera
        self.muted = False
        self.zone = ZoneManager()
        self.alerts = AlertTracker()
        self.voice = VoiceEngine()
        self.fps = 0
        self.fc = 0
        self.ft = time.time()

    def load_model(self):
        print("[MODEL] Loading YOLOv8n (COCO) for person detection...")
        self.model = YOLO("yolov8n.pt")
        print("[MODEL] Loaded! Detects persons at any distance.")

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.DISPLAY_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.DISPLAY_HEIGHT)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] Resolution: {w}x{h}")

    def detect_persons(self, frame):
        results = self.model.track(
            source=frame,
            conf=Config.PERSON_CONFIDENCE,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )
        persons = []
        if results and results[0].boxes and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                tid = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else i
                persons.append({'box': box, 'id': tid, 'conf': conf})
        return persons

    def run(self):
        self.load_model()
        self.open_camera()

        cv2.namedWindow(Config.WINDOW_NAME)
        cv2.setMouseCallback(Config.WINDOW_NAME, self.zone.mouse_callback)

        print()
        print("-" * 55)
        print("  AEGIS.AI Zone Restriction Module")
        print("  Press Z to define safe zone, then ENTER to confirm.")
        print("  Q=Quit | Z=Zone | R=Reset | M=Mute")
        print("-" * 55)
        print()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Detect persons
            persons = self.detect_persons(frame)

            # Check zone violations
            violations = []
            safe_workers = []

            for p in persons:
                inside, foot = self.zone.check_worker(p['box'])
                if self.zone.is_calibrated and not inside:
                    violations.append(p)

                    # Voice alert
                    aid = f"zone_{p['id']}"
                    if self.alerts.should_alert(aid, Config.ZONE_ALERT_COOLDOWN):
                        msg = f"Warning! Worker {p['id']}, you are outside the safe zone. Return immediately."
                        self.alerts.record_alert(aid, msg)
                        if not self.muted and not self.voice.is_busy():
                            self.voice.speak(msg)
                else:
                    safe_workers.append(p)

            # ── Draw ────────────────────────────────────────────────
            # Draw zone
            self.zone.draw(frame)

            # Draw persons
            for p in safe_workers:
                box = [int(v) for v in p['box']]
                label = f"P-{p['id']} [SAFE]"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                              Config.COLOR_SAFE, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (box[0], box[1]-th-8), (box[0]+tw+4, box[1]),
                              Config.COLOR_SAFE, -1)
                cv2.putText(frame, label, (box[0]+2, box[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WHITE, 1)

                # Green foot dot
                if self.zone.is_calibrated:
                    fx = int((p['box'][0]+p['box'][2])/2)
                    fy = int(p['box'][3])
                    cv2.circle(frame, (fx, fy), 6, Config.COLOR_SAFE, -1)

            for p in violations:
                box = [int(v) for v in p['box']]
                label = f"P-{p['id']} [ZONE VIOLATION!]"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                              Config.COLOR_VIOLATION, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (box[0], box[1]-th-8), (box[0]+tw+4, box[1]),
                              Config.COLOR_VIOLATION, -1)
                cv2.putText(frame, label, (box[0]+2, box[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WHITE, 1)

                # Red foot dot
                fx = int((p['box'][0]+p['box'][2])/2)
                fy = int(p['box'][3])
                cv2.circle(frame, (fx, fy), 6, Config.COLOR_VIOLATION, -1)

                # Danger icon
                cx, cy = box[2]-12, box[1]+12
                cv2.circle(frame, (cx, cy), 12, Config.COLOR_DANGER, -1)
                cv2.putText(frame, "!", (cx-5, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WHITE, 2)

            # ── HUD ─────────────────────────────────────────────────
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 55), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, "AEGIS.AI", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 50), 2)
            cv2.putText(frame, "Zone Restriction Module", (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            sx = w - 450
            cv2.putText(frame, f"FPS:{self.fps:.0f}", (sx, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)
            cv2.putText(frame, f"Workers:{len(persons)}", (sx+70, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_WHITE, 1)

            zt = "Zone:ON" if self.zone.is_calibrated else "Zone:OFF (press Z)"
            zc = Config.COLOR_SAFE if self.zone.is_calibrated else (100, 100, 100)
            cv2.putText(frame, zt, (sx+175, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, zc, 1)

            cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (sx+380, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            if violations:
                cv2.putText(frame, f"VIOLATIONS: {len(violations)}", (sx+175, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_DANGER, 2)

                # Bottom warning bar
                ov2 = frame.copy()
                cv2.rectangle(ov2, (0, h-35), (w, h), (0, 0, 180), -1)
                cv2.addWeighted(ov2, 0.8, frame, 0.2, 0, frame)
                cv2.putText(frame,
                            f"  WARNING: {len(violations)} worker(s) outside safe zone!",
                            (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WHITE, 1)
            else:
                if self.zone.is_calibrated:
                    cv2.putText(frame, "ALL CLEAR", (sx+175, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_SAFE, 1)

            # Instructions
            if not self.zone.calibrating:
                iy = h-8 if not violations else h-45
                cv2.putText(frame, "Q:Quit | Z:Set Zone | R:Reset Zone | M:Mute",
                            (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

            # FPS
            self.fc += 1
            if time.time() - self.ft >= 1.0:
                self.fps = self.fc / (time.time() - self.ft)
                self.fc = 0
                self.ft = time.time()

            cv2.imshow(Config.WINDOW_NAME, frame)

            # ── Keyboard ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('z') or key == ord('Z'):
                if not self.zone.calibrating:
                    self.zone.start_calibration()
            elif key == 13:  # ENTER
                if self.zone.calibrating:
                    self.zone.finish()
            elif key == ord('c') or key == ord('C'):
                if self.zone.calibrating:
                    self.zone.cancel()
            elif key == ord('r') or key == ord('R'):
                self.zone.reset()
            elif key == ord('m') or key == ord('M'):
                self.muted = not self.muted
                print(f"[AUDIO] {'MUTED' if self.muted else 'UNMUTED'}")

        self.cap.release()
        cv2.destroyAllWindows()

        print(f"\n  Total zone alerts: {len(self.alerts.alert_log)}")
        if self.alerts.alert_log:
            fn = f"zone_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fn, 'w') as f:
                json.dump(self.alerts.alert_log, f, indent=2)
            print(f"  Saved: {fn}")


def main():
    parser = argparse.ArgumentParser(description="AEGIS.AI Zone Restriction")
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--mute', action='store_true')
    args = parser.parse_args()

    app = ZoneDetector(camera=args.camera)
    if args.mute:
        app.muted = True
    app.run()


if __name__ == "__main__":
    main()
