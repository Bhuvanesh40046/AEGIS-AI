/*
 * ================================================================
 *  AEGIS.AI — ESP32 Sensor Node Firmware
 * ================================================================
 *  Reads BME280 (temperature, humidity, pressure) and MQ gas sensor.
 *  Sends data to laptop via HTTP POST every 3 seconds.
 *  
 *  WIRING:
 *  ┌─────────────┬───────────────┐
 *  │ BME280      │ ESP32         │
 *  ├─────────────┼───────────────┤
 *  │ VIN         │ 3.3V          │
 *  │ GND         │ GND           │
 *  │ SCL         │ GPIO 22       │
 *  │ SDA         │ GPIO 21       │
 *  ├─────────────┼───────────────┤
 *  │ MQ Sensor   │ ESP32         │
 *  ├─────────────┼───────────────┤
 *  │ VCC         │ 5V (VIN pin)  │
 *  │ GND         │ GND           │
 *  │ AO (analog) │ GPIO 34       │
 *  └─────────────┴───────────────┘
 *  
 *  SETUP:
 *  1. Install Arduino IDE
 *  2. Add ESP32 board: File > Preferences > Additional Board URLs:
 *     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 *  3. Install libraries: Adafruit BME280, Adafruit Unified Sensor
 *  4. Change WIFI_SSID, WIFI_PASS, SERVER_IP below
 *  5. Change ZONE_ID for each ESP32 node ("Zone-A", "Zone-B")
 *  6. Upload to ESP32
 * ================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// ═══════════════════════════════════════════════════════════════════
//  CONFIGURATION — CHANGE THESE FOR YOUR SETUP
// ═══════════════════════════════════════════════════════════════════

// WiFi credentials
const char* WIFI_SSID = "m5";
const char* WIFI_PASS = "12345678";

// Laptop's IP address
const char* SERVER_IP = "10.139.50.73";
const int SERVER_PORT = 5000;

// Zone identifier (change for each ESP32 node)
const char* ZONE_ID = "Zone-A";                 // ← "Zone-A" or "Zone-B"

// Sensor pins
const int MQ_ANALOG_PIN = 34;                   // MQ sensor analog output

// Reading interval (milliseconds)
const int READ_INTERVAL = 3000;                 // 3 seconds

// MQ sensor calibration
const float RL_VALUE = 10.0;                    // Load resistance in kOhms (on MQ module)
float R0 = 10.0;                                // Clean air resistance (calibrate this!)
bool mq_calibrated = false;

// ═══════════════════════════════════════════════════════════════════

Adafruit_BME280 bme;
bool bme_found = false;

void setup() {
    Serial.begin(115200);
    Serial.println();
    Serial.println("================================");
    Serial.println(" AEGIS.AI — ESP32 Sensor Node");
    Serial.println("================================");
    Serial.printf(" Zone: %s\n", ZONE_ID);
    
    // ── Initialize BME280 ──────────────────────────────────────────
    Wire.begin(21, 22);  // SDA=21, SCL=22
    if (bme.begin(0x76)) {
        bme_found = true;
        Serial.println("[BME280] Found at 0x76");
    } else if (bme.begin(0x77)) {
        bme_found = true;
        Serial.println("[BME280] Found at 0x77");
    } else {
        Serial.println("[BME280] NOT FOUND! Check wiring.");
        Serial.println("         Will send dummy temperature data.");
    }
    
    // ── Initialize MQ sensor pin ───────────────────────────────────
    pinMode(MQ_ANALOG_PIN, INPUT);
    Serial.println("[MQ] Analog pin configured.");
    Serial.println("[MQ] NOTE: Sensor needs 24-48hr burn-in for accuracy.");
    
    // ── Connect to WiFi ────────────────────────────────────────────
    Serial.printf("[WIFI] Connecting to: %s", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println(" Connected!");
        Serial.printf("[WIFI] IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println(" FAILED!");
        Serial.println("[WIFI] Check SSID and password.");
    }
    
    // ── Calibrate MQ sensor (R0 in clean air) ─────────────────────
    Serial.println("[MQ] Calibrating R0 in clean air...");
    calibrateMQ();
    
    Serial.println();
    Serial.println("System ready. Sending data every 3 seconds.");
    Serial.println("────────────────────────────────────────");
}

void calibrateMQ() {
    float sum = 0;
    int samples = 50;
    
    for (int i = 0; i < samples; i++) {
        int raw = analogRead(MQ_ANALOG_PIN);
        float voltage = raw * (3.3 / 4095.0);
        float rs = RL_VALUE * (3.3 - voltage) / voltage;
        sum += rs;
        delay(50);
    }
    
    float rs_avg = sum / samples;
    // In clean air, Rs/R0 ratio is approximately 9.8 for MQ-2
    R0 = rs_avg / 9.8;
    mq_calibrated = true;
    
    Serial.printf("[MQ] R0 calibrated: %.2f kOhms\n", R0);
    Serial.printf("[MQ] Rs in clean air: %.2f kOhms\n", rs_avg);
}

void loop() {
    // ── Read BME280 ────────────────────────────────────────────────
    float temperature = 0;
    float humidity = 0;
    float pressure = 0;
    
    if (bme_found) {
        temperature = bme.readTemperature();
        humidity = bme.readHumidity();
        pressure = bme.readPressure() / 100.0F;
    } else {
        // Dummy data if sensor not connected (for testing)
        temperature = 25.0 + random(-20, 20) / 10.0;
        humidity = 50.0 + random(-50, 50) / 10.0;
        pressure = 1013.25;
    }
    
    // ── Read MQ Sensor ─────────────────────────────────────────────
    int gas_raw = analogRead(MQ_ANALOG_PIN);
    float voltage = gas_raw * (3.3 / 4095.0);
    float rs = RL_VALUE * (3.3 - voltage) / voltage;
    float gas_ratio = rs / R0;  // Lower ratio = more gas
    
    // Determine gas status
    String gas_status = "normal";
    if (gas_ratio < 1.0) {
        gas_status = "danger";
    } else if (gas_ratio < 2.0) {
        gas_status = "warning";
    }
    
    // ── Print to serial ────────────────────────────────────────────
    Serial.printf("[%s] Temp: %.1f°C | Humidity: %.1f%% | Gas Raw: %d | Rs/R0: %.2f | Status: %s\n",
                  ZONE_ID, temperature, humidity, gas_raw, gas_ratio, gas_status.c_str());
    
    // ── Send to laptop via HTTP POST ───────────────────────────────
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        String url = "http://" + String(SERVER_IP) + ":" + String(SERVER_PORT) + "/";
        
        // Build JSON payload
        String json = "{";
        json += "\"zone\":\"" + String(ZONE_ID) + "\",";
        json += "\"temp\":" + String(temperature, 1) + ",";
        json += "\"humidity\":" + String(humidity, 1) + ",";
        json += "\"pressure\":" + String(pressure, 1) + ",";
        json += "\"gas_raw\":" + String(gas_raw) + ",";
        json += "\"gas_ratio\":" + String(gas_ratio, 2) + ",";
        json += "\"gas_status\":\"" + gas_status + "\"";
        json += "}";
        
        http.begin(url);
        http.addHeader("Content-Type", "application/json");
        
        int httpCode = http.POST(json);
        
        if (httpCode == 200) {
            Serial.println("  → Sent OK");
        } else {
            Serial.printf("  → Send FAILED (code: %d)\n", httpCode);
            Serial.println("    Check if laptop server is running.");
        }
        
        http.end();
    } else {
        Serial.println("  → WiFi disconnected! Reconnecting...");
        WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
    
    delay(READ_INTERVAL);
}
