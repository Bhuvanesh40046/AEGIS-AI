"""
=============================================================================
 AEGIS.AI — Helmet Detection + Zone Restriction
=============================================================================
 Phase 1: Dual-model helmet detection (COCO persons + custom helmet/head)
 Phase 2: Click-to-define safe zone with real-time violation alerts
 
 Controls:
   Z = Start zone calibration (click corners, ENTER to finish)
   R = Reset zone | Q = Quit | D = Debug | S = Screenshot | M = Mute
=============================================================================
"""

import cv2
import numpy as np
import time
import argparse
import json
import os
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


class Config:
    CUSTOM_MODEL_PATH = "best.pt"
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.4
    HELMET_CLASS_ID = 2
    HEAD_CLASS_ID = 1
    COCO_PERSON_CLASS_ID = 0
    HEAD_REGION_RATIO = 0.45
    ALERT_COOLDOWN_SECONDS = 15
    ZONE_ALERT_COOLDOWN = 10
    WINDOW_NAME = "AEGIS.AI - Worker Safety System"
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


def box_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

def box_distance(box1, box2):
    c1, c2 = box_center(box1), box_center(box2)
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5

def get_head_region(person_box, ratio=0.45):
    x1, y1, x2, y2 = person_box
    return [x1, y1, x2, y1 + (y2-y1)*ratio]

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0],box2[0]), max(box1[1],box2[1])
    x2, y2 = min(box1[2],box2[2]), min(box1[3],box2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = a1+a2-inter
    return inter/union if union > 0 else 0.0

def associate_helmets_to_persons(persons, helmets, heads):
    results = []
    for person in persons:
        head_region = get_head_region(person['box'])
        ph = person['box'][3]-person['box'][1]
        prox = ph*0.6
        hi = any(point_in_box(box_center(h['box']),head_region) for h in helmets)
        hdi = any(point_in_box(box_center(h['box']),head_region) for h in heads)
        bhi = max([calculate_iou(head_region,h['box']) for h in helmets], default=0)
        bhdi = max([calculate_iou(head_region,h['box']) for h in heads], default=0)
        ch = min([box_distance(head_region,h['box']) for h in helmets], default=float('inf'))
        chd = min([box_distance(head_region,h['box']) for h in heads], default=float('inf'))
        hip = any(point_in_box(box_center(h['box']),person['box']) for h in helmets)
        hdip = any(point_in_box(box_center(h['box']),person['box']) for h in heads)

        if hi or bhi > 0.05: status = "helmet"
        elif hdi or bhdi > 0.05: status = "no_helmet"
        elif ch < prox and ch < chd: status = "helmet"
        elif chd < prox and chd < ch: status = "no_helmet"
        elif hip: status = "helmet"
        elif hdip: status = "no_helmet"
        else: status = "no_helmet"
        results.append({**person, 'helmet_status': status, 'head_region': head_region})
    return results


class ZoneManager:
    def __init__(self):
        self.points = []
        self.is_calibrated = False
        self.calibrating = False

    def start_calibration(self):
        self.calibrating = True
        self.points = []
        self.is_calibrated = False
        print("\n" + "="*50)
        print("  ZONE CALIBRATION — Click corners, ENTER to finish")
        print("="*50)

    def mouse_callback(self, event, x, y, flags, param):
        if not self.calibrating: return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")

    def finish(self):
        if len(self.points) >= 3:
            self.is_calibrated = True
            self.calibrating = False
            print(f"  Safe zone set with {len(self.points)} corners!")
            return True
        print("  Need at least 3 points!")
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
        if not self.is_calibrated: return True
        polygon = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(polygon, (float(point[0]),float(point[1])), False) >= 0

    def check_worker(self, person_box):
        x1,y1,x2,y2 = person_box
        foot = (int((x1+x2)/2), int(y2))
        return self.is_inside(foot), foot

    def draw(self, frame):
        if self.calibrating:
            for i, pt in enumerate(self.points):
                cv2.circle(frame, pt, 8, Config.COLOR_ZONE, -1)
                if i > 0: cv2.line(frame, self.points[i-1], pt, Config.COLOR_ZONE, 2)
            if len(self.points) >= 3:
                cv2.line(frame, self.points[-1], self.points[0], Config.COLOR_ZONE, 1)
            h = frame.shape[0]
            ov = frame.copy()
            cv2.rectangle(ov, (0,h-45), (frame.shape[1],h), Config.COLOR_BLACK, -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"CALIBRATING | Points:{len(self.points)} | Click corners, ENTER=finish, C=cancel",
                        (10,h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_ZONE, 1)
        elif self.is_calibrated:
            pts = np.array(self.points, dtype=np.int32)
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], (0,100,0))
            cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, Config.COLOR_ZONE, 2)
            cv2.putText(frame, "SAFE ZONE", (self.points[0][0], self.points[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_ZONE, 2)


class AlertTracker:
    def __init__(self):
        self.last_alert_time = {}
        self.alert_count = defaultdict(int)
        self.alert_log = []

    def should_alert(self, alert_id, cooldown):
        return (time.time()-self.last_alert_time.get(alert_id,0)) >= cooldown

    def record_alert(self, alert_id, message):
        self.last_alert_time[alert_id] = time.time()
        self.alert_count[alert_id] += 1
        self.alert_log.append({"timestamp": datetime.now().isoformat(), "id": str(alert_id), "message": message})


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
                if len(voices) > 1: self.engine.setProperty('voice', voices[1].id)
                print("[TTS] Ready.")
            except Exception as e:
                print(f"[TTS] Failed: {e}")

    def speak(self, msg):
        if not self.engine or self._busy: return
        try:
            self._busy = True
            self.engine.say(msg)
            self.engine.runAndWait()
        except: pass
        finally: self._busy = False

    def is_busy(self): return self._busy


class HelmetDetector:
    def __init__(self, camera=0, model_path=None, debug=False):
        self.camera = camera
        self.model_path = model_path or Config.CUSTOM_MODEL_PATH
        self.debug = debug
        self.muted = False
        self.alerts = AlertTracker()
        self.voice = VoiceEngine()
        self.zone = ZoneManager()
        self.fps = 0
        self.fc = 0
        self.ft = time.time()

    def load_models(self):
        print("[MODEL 1] Loading YOLOv8n (COCO) for person detection...")
        self.person_model = YOLO("yolov8n.pt")
        print("[MODEL 1] Loaded!")
        self.has_helmet_model = False
        if os.path.exists(self.model_path):
            print(f"[MODEL 2] Loading {self.model_path} for helmet/head...")
            self.helmet_model = YOLO(self.model_path)
            print(f"[MODEL 2] Classes: {self.helmet_model.names}")
            self.has_helmet_model = True
        else:
            print(f"[MODEL 2] {self.model_path} not found")

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open camera {self.camera}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.DISPLAY_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.DISPLAY_HEIGHT)
        print(f"[CAMERA] {int(self.cap.get(3))}x{int(self.cap.get(4))}")

    def detect_persons(self, frame):
        results = self.person_model.track(source=frame, conf=0.35, classes=[0],
                                          tracker="bytetrack.yaml", persist=True, verbose=False)
        persons = []
        if results and results[0].boxes and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                tid = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else i
                persons.append({'box': box, 'id': tid, 'conf': conf})
        return persons

    def detect_helmets(self, frame):
        helmets, heads = [], []
        if not self.has_helmet_model: return helmets, heads
        results = self.helmet_model(frame, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
        if results and results[0].boxes and len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                box = results[0].boxes.xyxy[i].cpu().numpy().tolist()
                cls = int(results[0].boxes.cls[i].cpu().numpy())
                conf = float(results[0].boxes.conf[i].cpu().numpy())
                if cls == Config.HELMET_CLASS_ID: helmets.append({'box': box, 'conf': conf})
                elif cls == Config.HEAD_CLASS_ID: heads.append({'box': box, 'conf': conf})
        return helmets, heads

    def run(self):
        self.load_models()
        self.open_camera()
        cv2.namedWindow(Config.WINDOW_NAME)
        cv2.setMouseCallback(Config.WINDOW_NAME, self.zone.mouse_callback)

        print("\n" + "-"*60)
        print("  Q=Quit | Z=Zone | R=Reset | D=Debug | S=Screenshot | M=Mute")
        print("-"*60 + "\n")

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            persons = self.detect_persons(frame)
            helmets, heads = self.detect_helmets(frame)

            if self.has_helmet_model:
                dets = associate_helmets_to_persons(persons, helmets, heads)
            else:
                dets = [{**p, 'helmet_status':'unknown', 'head_region':get_head_region(p['box'])} for p in persons]

            # Check violations
            hv = 0
            zv = []
            for d in dets:
                # Helmet check (P3)
                if d['helmet_status'] == 'no_helmet':
                    hv += 1
                    aid = f"helmet_{d['id']}"
                    if self.alerts.should_alert(aid, Config.ALERT_COOLDOWN_SECONDS):
                        msg = f"Attention! Worker {d['id']}, please wear your safety helmet."
                        self.alerts.record_alert(aid, msg)
                        if not self.muted and not self.voice.is_busy(): self.voice.speak(msg)

                # Zone check (P2)
                if self.zone.is_calibrated:
                    inside, foot = self.zone.check_worker(d['box'])
                    if not inside:
                        zv.append(d['id'])
                        aid = f"zone_{d['id']}"
                        if self.alerts.should_alert(aid, Config.ZONE_ALERT_COOLDOWN):
                            msg = f"Warning! Worker {d['id']}, return to the safe zone immediately."
                            self.alerts.record_alert(aid, msg)
                            if not self.muted and not self.voice.is_busy(): self.voice.speak(msg)

            # ── Draw ────────────────────────────────────────────────────
            h, w = frame.shape[:2]
            self.zone.draw(frame)

            for d in dets:
                box = [int(v) for v in d['box']]
                st = d['helmet_status']
                tid = d.get('id','?')
                zflag = tid in zv

                labels = []
                if st == "helmet": color = Config.COLOR_SAFE; labels.append("HELMET OK")
                else: color = Config.COLOR_DANGER; labels.append("NO HELMET!")
                if zflag: color = Config.COLOR_ZONE_VIOLATION; labels.append("ZONE!")
                label = f"P-{tid} [{' | '.join(labels)}]"

                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), color, 2)
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (box[0],box[1]-th-8), (box[0]+tw+4,box[1]), color, -1)
                cv2.putText(frame, label, (box[0]+2,box[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_WHITE, 1)

                if st == "no_helmet" or zflag:
                    cx,cy = box[2]-12, box[1]+12
                    cv2.circle(frame, (cx,cy), 10, Config.COLOR_DANGER, -1)
                    cv2.putText(frame, "!", (cx-4,cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.COLOR_WHITE, 2)

                if self.zone.is_calibrated:
                    fx,fy = int((d['box'][0]+d['box'][2])/2), int(d['box'][3])
                    fc = Config.COLOR_ZONE_VIOLATION if zflag else Config.COLOR_SAFE
                    cv2.circle(frame, (fx,fy), 5, fc, -1)

            if self.debug:
                for hl in helmets:
                    b=[int(v) for v in hl['box']]
                    cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),Config.COLOR_HELMET_BOX,1)
                for hd in heads:
                    b=[int(v) for v in hd['box']]
                    cv2.rectangle(frame,(b[0],b[1]),(b[2],b[3]),Config.COLOR_HEAD_BOX,1)

            # HUD
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(w,55),(0,0,0),-1)
            cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
            cv2.putText(frame,"AEGIS.AI",(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,200,50),2)
            cv2.putText(frame,"Helmet + Zone Detection",(10,42),cv2.FONT_HERSHEY_SIMPLEX,0.35,(180,180,180),1)

            sx = w-500
            cv2.putText(frame,f"FPS:{self.fps:.0f}",(sx,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,200,100),1)
            cv2.putText(frame,f"Workers:{len(dets)}",(sx+70,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,Config.COLOR_WHITE,1)
            zt = "Zone:ON" if self.zone.is_calibrated else "Zone:OFF"
            zc = Config.COLOR_SAFE if self.zone.is_calibrated else (100,100,100)
            cv2.putText(frame,zt,(sx+175,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,zc,1)
            cv2.putText(frame,datetime.now().strftime("%H:%M:%S"),(sx+420,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1)

            tv = hv + len(zv)
            if tv > 0:
                parts = []
                if hv > 0: parts.append(f"{hv} helmet")
                if zv: parts.append(f"{len(zv)} zone")
                cv2.putText(frame,"VIOLATIONS: "+" | ".join(parts),(sx+275,22),cv2.FONT_HERSHEY_SIMPLEX,0.4,Config.COLOR_DANGER,2)
                ov2 = frame.copy()
                cv2.rectangle(ov2,(0,h-35),(w,h),(0,0,180),-1)
                cv2.addWeighted(ov2,0.8,frame,0.2,0,frame)
                cv2.putText(frame,f"  WARNING: {' | '.join(parts)}",(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,Config.COLOR_WHITE,1)
            else:
                cv2.putText(frame,"ALL CLEAR",(sx+275,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,Config.COLOR_SAFE,1)

            if not self.zone.calibrating:
                iy = h-8 if tv==0 else h-45
                cv2.putText(frame,"Q:Quit | Z:Zone | R:Reset | D:Debug | S:Screenshot | M:Mute",(10,iy),cv2.FONT_HERSHEY_SIMPLEX,0.3,(100,100,100),1)

            self.fc += 1
            if time.time()-self.ft >= 1.0:
                self.fps = self.fc/(time.time()-self.ft)
                self.fc = 0; self.ft = time.time()

            cv2.imshow(Config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('z'): self.zone.start_calibration()
            elif key == 13:
                if self.zone.calibrating: self.zone.finish()
            elif key == ord('c'):
                if self.zone.calibrating: self.zone.cancel()
            elif key == ord('r'): self.zone.reset()
            elif key == ord('d'): self.debug = not self.debug; print(f"[DEBUG] {'ON' if self.debug else 'OFF'}")
            elif key == ord('s'): fn=f"ss_{datetime.now().strftime('%H%M%S')}.jpg"; cv2.imwrite(fn,frame); print(f"[SS] {fn}")
            elif key == ord('m'): self.muted = not self.muted; print(f"[AUDIO] {'MUTED' if self.muted else 'UNMUTED'}")

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n  Total alerts: {len(self.alerts.alert_log)}")
        if self.alerts.alert_log:
            fn = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fn,'w') as f: json.dump(self.alerts.alert_log, f, indent=2)
            print(f"  Saved: {fn}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--mute', action='store_true')
    a = p.parse_args()
    d = HelmetDetector(camera=a.camera, model_path=a.model, debug=a.debug)
    if a.mute: d.muted = True
    d.run()

if __name__ == "__main__":
    main()
