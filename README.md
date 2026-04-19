# 🛡 AEGIS.AI — Agentic AI-Based Smart Worker Safety and Monitoring System

> **Real-time worker safety monitoring for industrial IoT environments using multi-agent AI**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![ESP32](https://img.shields.io/badge/ESP32-IoT-red?logo=espressif)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 About

AEGIS.AI is an integrated worker safety system that uses **computer vision**, **IoT sensors**, and a **multi-agent AI orchestrator** to protect workers in industrial environments like chemical factories, construction sites, and manufacturing plants.

The system runs three autonomous AI agents coordinated by a central orchestrator with a priority-based alert mechanism:

| Priority | Agent | Function |
|----------|-------|----------|
| P1 (Highest) | Environmental Agent | Gas leak / temperature danger → EVACUATE |
| P2 | Zone Agent | Restricted area violation → RETURN TO SAFE ZONE |
| P3 | PPE Agent | No helmet detected → WEAR HELMET |

**Voice alerts** are delivered in real-time through laptop speakers using offline text-to-speech — no internet required.

---

## 🎥 Demo Video

Click below to watch the project in action:

👉 [AEGIS-AI Demo Video](./demo.mp4)

---
## 🎯 Features

- **Helmet Detection** — Dual YOLOv8 model approach (COCO for person detection + custom trained for helmet/head)
- **Person Tracking** — ByteTrack assigns persistent IDs to each worker across frames
- **Zone Restriction** — Click-to-define safe zone with real-time boundary violation detection
- **Environmental Monitoring** — ESP32 + BME280 (temperature/humidity) + MQ gas sensor via Wi-Fi
- **Priority-Based Orchestrator** — Gas emergencies suppress lower-priority alerts
- **Voice Alerts** — Offline TTS with cooldown system (no repeated nagging)
- **Session Logging** — All alerts saved to JSON with timestamps for audit trail
- **Gas Emergency Mode** — Screen flashes red during evacuation events

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LAPTOP (Python)                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  PPE Agent   │  │  Zone Agent  │  │ Env Agent │ │
│  │  YOLOv8n     │  │  Polygon +   │  │ ESP32 +   │ │
│  │  (COCO) +    │  │  pointPoly   │  │ BME280 +  │ │
│  │  YOLOv8n     │  │  Test        │  │ MQ Sensor │ │
│  │  (Custom)    │  │              │  │ via HTTP  │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│         │                 │                 │       │
│         ▼                 ▼                 ▼       │
│  ┌─────────────────────────────────────────────┐    │
│  │         ORCHESTRATOR (Priority Queue)       │    │
│  │  P1: Gas/Temp → EVACUATE                    │    │
│  │  P2: Zone → RETURN TO SAFE ZONE             │    │
│  │  P3: Helmet → WEAR HELMET                   │    │
│  └─────────────────┬───────────────────────────┘    │
│                    ▼                                │
│              Voice Alerts (pyttsx3)                  │
└─────────────────────────────────────────────────────┘
                    │ Wi-Fi
              ┌─────┴─────┐
              │   ESP32    │
              │   BME280   │
              │   MQ Sensor│
              └────────────┘
```

---

## 📁 Project Structure

```
AEGIS-AI/
├── aegis_main.py                 # Complete integrated system (all 4 phases)
├── helmet_detector.py            # Standalone helmet detection module
├── zone_detector.py              # Standalone zone restriction module
├── esp32_simulator.py            # ESP32 simulator (test without hardware)
├── train_helmet_model.py         # YOLOv8 training script
├── test_setup.py                 # System check / dependency validator
├── fix_split.py                  # Dataset train/valid split utility
├── best.pt                       # Trained YOLOv8 helmet detection model
├── requirements.txt              # Python dependencies
├── esp32_firmware/
│   └── esp32_sensor_node.ino     # Arduino firmware for ESP32
├── docs/
│   ├── PROJECT_EXPLANATION.md    # Detailed project explanation
│   ├── SETUP_GUIDE.md            # Step-by-step setup instructions
│   └── WIRING_DIAGRAM.md        # ESP32 wiring instructions
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU (4GB+ VRAM recommended) or CPU
- Webcam (built-in or USB)
- ESP32 + BME280 + MQ sensor (optional — simulator available)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AEGIS-AI.git
cd AEGIS-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Setup
```bash
python test_setup.py
```

### 4. Run the Integrated System
```bash
# With ESP32 sensors
python aegis_main.py --model best.pt --camera 0

# Without ESP32 (testing)
python aegis_main.py --model best.pt --camera 0 --no-esp32
```

### 5. Run Individual Modules
```bash
# Helmet detection only
python helmet_detector.py --model best.pt --camera 0

# Zone restriction only
python zone_detector.py --camera 0
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Z` | Start zone calibration (click corners, ENTER to finish) |
| `R` | Reset zone |
| `Q` | Quit |
| `D` | Toggle debug mode |
| `S` | Save screenshot |
| `M` | Mute/Unmute voice alerts |

---

## 🔧 ESP32 Setup

### Wiring
```
BME280          ESP32
──────          ──────
VIN      →      3.3V
GND      →      GND
SCL      →      GPIO 22
SDA      →      GPIO 21

MQ Sensor       ESP32
──────          ──────
VCC      →      5V (VIN)
GND      →      GND
AO       →      GPIO 34
```

### Firmware
1. Open `esp32_firmware/esp32_sensor_node.ino` in Arduino IDE
2. Update Wi-Fi credentials and laptop IP address
3. Install libraries: `Adafruit BME280`, `Adafruit Unified Sensor`
4. Upload to ESP32

### Testing Without Hardware
```bash
# In a separate terminal
python esp32_simulator.py              # Normal simulation
python esp32_simulator.py --gas-leak   # Simulate gas leak after 30s
```

---

## 🧠 Models Used

| Model | Type | Purpose |
|-------|------|---------|
| YOLOv8n (COCO) | Pretrained | Person detection at any distance |
| YOLOv8n (Custom) | Fine-tuned | Helmet and bare head detection |
| ByteTrack | Tracker | Persistent person IDs across frames |

### Training Your Own Model
```bash
# 1. Download dataset from Roboflow (head, helmet, Person classes)
# 2. Place in datasets/helmet_dataset/
# 3. Train
python train_helmet_model.py --dataset ./datasets/helmet_dataset --epochs 30 --batch 16 --size n --device 0
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Person Detection Accuracy | ~95% (COCO YOLOv8n) |
| Helmet Detection mAP50 | ~85% (custom trained) |
| FPS | 15-30 FPS (NVIDIA GTX 1650) |
| Alert Response Time | < 1 second |
| Sensor Data Latency | ~3 seconds |

---

## 🛠 Tech Stack

- **AI/ML:** YOLOv8, ByteTrack, OpenCV
- **IoT:** ESP32, BME280, MQ Gas Sensor
- **Communication:** HTTP/Wi-Fi
- **Voice:** pyttsx3 (offline TTS)
- **Languages:** Python, C++ (Arduino)

---

## 🔮 Future Enhancements

- [ ] Fall detection using pose estimation
- [ ] Fatigue/drowsiness detection via eye tracking
- [ ] Live web dashboard for remote monitoring
- [ ] WhatsApp/Telegram alerts to supervisors
- [ ] Multi-camera support with view stitching
- [ ] Edge deployment on Jetson Nano
- [ ] Predictive analytics for violation patterns
- [ ] Digital twin of factory floor

---

## 👥 Team

- **B.V.S. Bhuvanesh** — AI/ML Development, Core System Integration, Testing & Validation  
- **N. Varsha** — IoT Development, Integration Support, Documentation  
- **Venkat Prasad** — AI/ML & IoT Integration Support, Testing & Validation 

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection framework
- [Roboflow](https://roboflow.com) — Dataset hosting and annotation
- [OpenCV](https://opencv.org) — Computer vision library
- [Espressif ESP32](https://www.espressif.com) — IoT microcontroller
