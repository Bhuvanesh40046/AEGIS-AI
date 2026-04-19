# AEGIS.AI — Agentic AI-Based Smart Worker Safety and Monitoring System for Industrial IoT Environments

## Project Summary (30-second pitch)

We built a real-time worker safety system for chemical/industrial factories that uses AI cameras and IoT sensors to protect workers. The system has three autonomous AI agents that work together: one watches if workers are wearing helmets, one tracks if workers enter restricted danger zones, and one monitors the factory environment for gas leaks and temperature spikes. All three agents are coordinated by an Orchestrator that prioritizes alerts — a gas leak evacuation alert overrides everything else. Workers receive voice alerts through speakers in real-time.

---

## Problem Statement

In industrial environments like chemical factories, worker injuries and fatalities occur due to:
- Not wearing Personal Protective Equipment (PPE) like helmets
- Workers accidentally entering restricted/hazardous areas
- Gas leaks and temperature spikes going undetected until too late
- Human supervisors cannot monitor everything 24/7

**Our solution:** Replace manual supervision with an autonomous AI-based multi-agent system that monitors, detects, and alerts in real-time — without any human in the loop.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LAPTOP (Python)                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  PPE Agent   │  │  Zone Agent  │  │ Env Agent │ │
│  │  (Phase 1)   │  │  (Phase 2)   │  │ (Phase 3) │ │
│  │              │  │              │  │           │ │
│  │ YOLOv8n     │  │ Polygon +    │  │ ESP32 +   │ │
│  │ (persons)   │  │ pointPoly    │  │ BME280 +  │ │
│  │ + Custom    │  │ Test         │  │ MQ Sensor │ │
│  │ YOLOv8n     │  │              │  │ via HTTP  │ │
│  │ (helmet/    │  │              │  │           │ │
│  │  head)      │  │              │  │           │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│         │                 │                 │       │
│         ▼                 ▼                 ▼       │
│  ┌─────────────────────────────────────────────┐    │
│  │         ORCHESTRATOR (Phase 4)              │    │
│  │                                             │    │
│  │  Priority Queue:                            │    │
│  │  P1: Gas leak/High temp → "EVACUATE!"       │    │
│  │  P2: Zone violation → "Return to safe zone" │    │
│  │  P3: No helmet → "Wear your helmet"         │    │
│  │                                             │    │
│  │  • Cooldown per worker per alert type       │    │
│  │  • P1 suppresses P2 and P3                  │    │
│  └─────────────────┬───────────────────────────┘    │
│                    │                                │
│                    ▼                                │
│           ┌────────────────┐                        │
│           │  pyttsx3 TTS   │                        │
│           │  Voice Alerts  │                        │
│           │  via Speaker   │                        │
│           └────────────────┘                        │
│                                                     │
└─────────────────────────────────────────────────────┘
                    │
                    │ Wi-Fi (HTTP POST)
                    │
              ┌─────┴─────┐
              │  ESP32     │
              │  BME280    │
              │  MQ Sensor │
              │  (Zone-A)  │
              └────────────┘
```

---

## Models and Technologies Used

### AI / ML Models:
| Model | Purpose | Details |
|-------|---------|---------|
| YOLOv8n (COCO pretrained) | Person Detection | Detects humans at any distance, any angle. Standard COCO dataset with 80 classes, we use only class 0 (person). |
| YOLOv8n (Custom trained) | Helmet & Head Detection | Trained on 1,280 images from Roboflow. 3 classes: Person, head (no helmet), helmet. |
| ByteTrack | Person Tracking | Assigns persistent IDs to each person across frames (Person-1, Person-2, etc.) so we can track who has been alerted. |

### Why Dual Model?
A single custom model couldn't detect persons reliably at close range because it was trained on construction site images (far distance). So we use:
- **Model 1 (COCO)** → Always detects persons reliably at any distance
- **Model 2 (Custom)** → Detects helmet and bare head specifically

Then we **associate** helmet/head detections to persons using center-point containment, IoU overlap, and proximity matching.

### Hardware:
| Component | Purpose |
|-----------|---------|
| ESP32 DevKit | Microcontroller — reads sensors, sends data over Wi-Fi |
| BME280 | Temperature + Humidity + Pressure sensor (±0.5°C accuracy) |
| MQ Gas Sensor | Detects combustible gases (LPG, CO, methane). Analog output. |
| Laptop Camera / USB Webcam | Video feed for AI detection |
| Laptop Speakers | Voice alert output via pyttsx3 |

### Software / Libraries:
| Library | Purpose |
|---------|---------|
| Ultralytics (YOLOv8) | Object detection + ByteTrack tracking |
| OpenCV | Camera capture, image processing, zone polygon math |
| pyttsx3 | Offline text-to-speech for voice alerts |
| Flask/HTTPServer | Receives sensor data from ESP32 via HTTP POST |
| NumPy | Array operations for bounding box math |
| Arduino IDE | ESP32 firmware development |

---

## Phase-by-Phase Explanation

### Phase 1: Helmet Detection (PPE Agent)
- **Input:** Live camera feed
- **Process:**
  1. YOLOv8n (COCO) detects all persons → gives bounding boxes
  2. ByteTrack assigns persistent IDs (Person-1, Person-2...)
  3. Custom YOLOv8n detects helmet and head (bare head) objects
  4. For each person, we check the top 45% of their bounding box (head region)
  5. If a helmet's center point falls inside the head region → SAFE (green box)
  6. If a bare head's center falls inside → NO HELMET (red box)
  7. Also uses IoU overlap and proximity as fallback matching
- **Output:** Each person labeled as "HELMET OK" or "NO HELMET!"
- **Alert:** Voice says "Attention! Worker 3, please wear your safety helmet."
- **Cooldown:** 15 seconds per worker (doesn't repeat same alert too fast)

### Phase 2: Zone Restriction (Zone Agent)
- **Input:** Same camera feed + user-defined safe zone
- **Process:**
  1. User presses Z and clicks corners on the video to define a safe zone polygon
  2. For each detected person, we take the bottom-center of their bounding box (represents feet)
  3. OpenCV's `pointPolygonTest()` checks if feet are inside or outside the polygon
  4. If outside → ZONE VIOLATION (blue box)
- **Output:** Green zone overlay on video, workers outside get blue warning
- **Alert:** Voice says "Warning! Worker 3, return to safe zone immediately."
- **Cooldown:** 10 seconds per worker

### Phase 3: Environmental Monitoring (Environmental Agent)
- **Input:** ESP32 sensor data via Wi-Fi (HTTP POST every 3 seconds)
- **Hardware Setup:**
  - BME280 connected to ESP32 via I2C (GPIO 21/22)
  - MQ gas sensor connected to ESP32 analog pin (GPIO 34)
  - ESP32 connects to same Wi-Fi as laptop
  - Sends JSON: `{"zone":"Zone-A", "temp":25.3, "humidity":47, "gas_raw":1150}`
- **Process:**
  1. Python HTTP server on laptop receives sensor data
  2. Checks temperature against thresholds (Warning: 40°C, Danger: 50°C)
  3. Checks gas level against thresholds (Warning: 2000 raw, Danger: 2800 raw)
  4. MQ sensor calibrated using R0 (clean air baseline)
- **Output:** Sensor panel on right side of screen showing live readings
- **Alert:** Gas leak → "EMERGENCY! Gas leak! All workers evacuate immediately!"

### Phase 4: Orchestrator (Coordination Agent)
- **Purpose:** Coordinates all three agents with a priority system
- **Priority System:**
  - P1 (HIGHEST): Gas leak or dangerous temperature → EVACUATE
  - P2: Zone violation → Return to safe zone
  - P3 (LOWEST): No helmet → Wear helmet
- **Key Rule:** When P1 is active (gas emergency), P2 and P3 alerts are suppressed. You don't care about helmets during an evacuation.
- **Voice Alert System:**
  - Uses pyttsx3 (offline TTS) — works without internet
  - 3-second minimum gap between voice alerts (prevents overlapping)
  - Cooldown per worker per alert type (no repetitive nagging)
- **Emergency Mode:** Screen flashes red during gas emergency
- **Logging:** All alerts saved to JSON file with timestamp, priority, and worker ID

---

## What Makes This "Agentic AI"?

Traditional system: Cameras record → Human watches monitor → Human decides → Human alerts

**Our agentic system:** Cameras feed AI → AI detects → AI decides → AI alerts → All autonomous

Each "agent" has:
1. **Perception** — camera / sensors
2. **Reasoning** — YOLOv8 inference / threshold logic / polygon math
3. **Action** — voice alert / visual warning

The Orchestrator coordinates multiple agents — this is a **multi-agent system** where agents communicate through a shared priority queue.

---

## Demo Flow (for evaluator)

1. **Start the system:** `python aegis_main.py --model best.pt --camera 0`
2. **Show helmet detection:** Stand in front of camera without helmet → red box + voice alert. Put on helmet → green box.
3. **Show zone restriction:** Press Z, click 4 corners to draw safe zone, press ENTER. Walk outside → blue box + voice alert. Walk back in → green.
4. **Show ESP32 sensors:** Point to sensor panel on right side showing live temperature, humidity, gas values.
5. **Show gas emergency:** (Option A: Bring lighter gas near MQ sensor. Option B: Temporarily lower threshold in code to 1500 to simulate.)
6. **Show priority system:** During gas emergency, helmet alerts are suppressed. Screen flashes red.
7. **Show session summary:** Press Q to quit — see total alerts by priority level.

---

## Results / Metrics

- Person detection: ~95% accuracy (COCO YOLOv8n)
- Helmet detection: ~85% mAP50 (custom trained)
- FPS: 15-30 FPS on NVIDIA GTX 1650 (4GB)
- Alert response time: < 1 second from detection to voice alert
- Sensor data latency: ~3 seconds (ESP32 polling interval)
- False positive rate: Low (dual-model approach + multi-method association)

---

## Future Enhancements

1. Face recognition for worker-specific name-based alerts
2. Multiple cameras for full factory coverage
3. Cloud dashboard for remote monitoring
4. Predictive analytics (fatigue prediction, accident forecasting)
5. Integration with factory emergency systems (auto-shutdown, alarm)
6. Mobile app for supervisor notifications
