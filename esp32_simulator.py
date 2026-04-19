"""
=============================================================================
 AEGIS.AI — ESP32 Sensor Simulator
=============================================================================
 Use this to TEST the full system WITHOUT actual ESP32 hardware.
 Simulates two sensor zones sending data to the main app.
 
 Usage:
   python esp32_simulator.py                # Normal simulation
   python esp32_simulator.py --gas-leak     # Simulate gas leak after 30 sec
   python esp32_simulator.py --hot          # Simulate high temperature
=============================================================================
"""

import requests
import time
import random
import argparse
import math

def simulate(args):
    url = f"http://127.0.0.1:{args.port}/"
    
    print("=" * 50)
    print("  ESP32 Sensor Simulator")
    print(f"  Sending to: {url}")
    print(f"  Gas leak mode: {'YES' if args.gas_leak else 'No'}")
    print(f"  High temp mode: {'YES' if args.hot else 'No'}")
    print("=" * 50)
    
    start_time = time.time()
    cycle = 0
    
    while True:
        elapsed = time.time() - start_time
        cycle += 1
        
        for zone in ["Zone-A", "Zone-B"]:
            # Base values
            base_temp = 32.0 if zone == "Zone-A" else 28.0
            base_humidity = 55.0 if zone == "Zone-A" else 48.0
            base_gas = 200 if zone == "Zone-A" else 150
            
            # Add some random variation
            temp = base_temp + random.uniform(-1, 1)
            humidity = base_humidity + random.uniform(-3, 3)
            gas_raw = base_gas + random.randint(-20, 20)
            
            # Simulate high temperature after 60 seconds
            if args.hot and elapsed > 60:
                temp += min((elapsed - 60) * 0.3, 25)  # Gradually increase
            
            # Simulate gas leak after 30 seconds (Zone-A only)
            if args.gas_leak and elapsed > 30 and zone == "Zone-A":
                gas_raw += min(int((elapsed - 30) * 20), 700)  # Gradually increase
            
            gas_ratio = max(0.3, 10.0 - (gas_raw / 100.0))  # Approximate ratio
            
            gas_status = "normal"
            if gas_raw > 800:
                gas_status = "danger"
            elif gas_raw > 600:
                gas_status = "warning"
            
            data = {
                "zone": zone,
                "temp": round(temp, 1),
                "humidity": round(humidity, 1),
                "pressure": 1013.25,
                "gas_raw": gas_raw,
                "gas_ratio": round(gas_ratio, 2),
                "gas_status": gas_status
            }
            
            try:
                resp = requests.post(url, json=data, timeout=2)
                status = "OK" if resp.status_code == 200 else f"ERR:{resp.status_code}"
            except requests.exceptions.ConnectionError:
                status = "NO CONNECTION (is aegis_main.py running?)"
            except Exception as e:
                status = f"ERR: {e}"
            
            print(f"  [{zone}] Temp:{temp:.1f}C Gas:{gas_raw} Status:{gas_status} → {status}")
        
        print(f"  --- Cycle {cycle} | Elapsed: {elapsed:.0f}s ---")
        time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--gas-leak', action='store_true', help='Simulate gas leak after 30s')
    parser.add_argument('--hot', action='store_true', help='Simulate high temp after 60s')
    args = parser.parse_args()
    simulate(args)
