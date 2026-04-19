# ESP32 Wiring Diagram

## Components Required (per node)

| Component | Qty | Approx Cost (₹) |
|-----------|-----|-----------------|
| ESP32 DevKit V1 | 1 | 500 |
| BME280 Sensor (I2C) | 1 | 250 |
| MQ-2 Gas Sensor Module | 1 | 150 |
| Breadboard | 1 | 80 |
| Jumper Wires | ~10 | 50 |

## Wiring

```
    ┌──────────────────────────────────────────────┐
    │                  ESP32 DevKit                 │
    │                                              │
    │  3.3V ──────────── VIN (BME280)              │
    │  GND  ──────────── GND (BME280)              │
    │  GPIO 22 (SCL) ─── SCL (BME280)              │
    │  GPIO 21 (SDA) ─── SDA (BME280)              │
    │                                              │
    │  5V (VIN) ──────── VCC (MQ Sensor)           │
    │  GND  ──────────── GND (MQ Sensor)           │
    │  GPIO 34 ───────── AO  (MQ Sensor)           │
    │                                              │
    └──────────────────────────────────────────────┘
```

## BME280 Pinout

```
BME280 Module:
┌─────────┐
│  VIN    │ → ESP32 3.3V
│  GND    │ → ESP32 GND
│  SCL    │ → ESP32 GPIO 22
│  SDA    │ → ESP32 GPIO 21
└─────────┘
```

## MQ-2 Gas Sensor Pinout

```
MQ-2 Module:
┌─────────┐
│  VCC    │ → ESP32 5V (VIN pin)
│  GND    │ → ESP32 GND
│  AO     │ → ESP32 GPIO 34 (analog)
│  DO     │ → Not used
└─────────┘
```

## Important Notes

1. **BME280 address:** Usually 0x76 or 0x77 (firmware tries both)
2. **MQ sensor warm-up:** Needs 24-48 hours continuous power for accurate readings
3. **MQ sensor calibration:** R0 is auto-calibrated in clean air at startup
4. **Power:** ESP32 can be powered via USB cable from laptop or any 5V USB adapter
5. **Wi-Fi:** ESP32 and laptop must be on the same Wi-Fi network
