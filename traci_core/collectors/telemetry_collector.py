import csv
import sys
import random
import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import traci
from traci_core.config import DATASET_RAW_PATH, SIM_CONFIG_PATH, STEP_LENGTH, EGO_VEHICLE_ID, MAX_SPEED
from traci_core.collectors.sensor_utils import SensorUtils
from traci_core.collectors.v2v_module import V2VModule
from traci_core.scenarios import EmergencyBrakingScenario

TOTAL_SAMPLES = 50000


def run_sumo_data_generation(target_samples: int = TOTAL_SAMPLES):
    DATASET_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "vehicle_speed",
        "leader_speed",
        "leader_distance",
        "relative_speed",
        "time_to_collision",
        "speed_limit",
        "v2v_emergency_flag",
        "target_speed"
    ]

    # Fast headless mode
    sumo_cmd = [
        "sumo",
        "-c", str(SIM_CONFIG_PATH),
        "--step-length", str(STEP_LENGTH),
        "--no-step-log", "true",
        "--collision.action", "none"
    ]

    samples_collected = 0
    print(f"Starting Headless SUMO to capture {target_samples} calibrated real-physics samples...")

    with open(DATASET_RAW_PATH, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        while samples_collected < target_samples:
            traci.start(sumo_cmd)
            
            v2v = V2VModule(comm_range_meters=150.0)
            scenario = EmergencyBrakingScenario(
                lead_id="lead_car",
                trigger_step=random.randint(35, 65),
                stop_duration=random.randint(30, 45),
                resume_speed=11.0
            )
            step_count = 0

            while traci.simulation.getMinExpectedNumber() > 0 and samples_collected < target_samples:
                traci.simulationStep()
                step_count += 1
                active_vehicles = traci.vehicle.getIDList()

                # Scenario execution (Emergency brake trigger & V2V broadcast)
                scenario.update(current_step=step_count, active_vehicles=active_vehicles, v2v_module=v2v)

                # Ego vehicle extraction
                if EGO_VEHICLE_ID in active_vehicles:
                    telemetry = SensorUtils.get_vehicle_telemetry(EGO_VEHICLE_ID)
                    if not telemetry:
                        v2v.step_cleanup()
                        continue

                    ego_speed = telemetry["speed"]
                    speed_limit = telemetry["speed_limit"]
                    ego_pos = traci.vehicle.getPosition(EGO_VEHICLE_ID)

                    # Radar ground truth readings
                    radar = SensorUtils.detect_leader(EGO_VEHICLE_ID, radar_range=100.0, current_speed=ego_speed)
                    leader_speed = radar["leader_speed"]
                    leader_gap = radar["gap"]
                    leader_id = radar.get("leader_id")

                    # V2V module checks
                    v2v_alerts = v2v.receive_for_vehicle(EGO_VEHICLE_ID, ego_pos)
                    v2v_emergency = any(msg.get("event") == "EMERGENCY_BRAKE" for msg in v2v_alerts)
                    v2v_flag = 1 if v2v_emergency else 0

                    # 1. Physics Engine Ground Truth Calculation (Consistent Distance & Responsive Restart)
                    standstill_gap = 5.5   # Consistent safe stopping buffer (meters)
                    comfort_decel = 2.8    # Standard smooth braking deceleration (m/s^2)

                    if v2v_flag == 1 or leader_gap <= standstill_gap:
                        # Smooth parabolic kinematic deceleration curve: v = sqrt(2 * a * d)
                        effective_gap = max(0.0, leader_gap - standstill_gap)
                        target_speed = math.sqrt(2.0 * comfort_decel * effective_gap)
                        
                        # Full stop deadband buffer
                        if leader_gap <= 4.0:
                            target_speed = 0.0

                    elif leader_gap < 75.0 and leader_id and leader_id in active_vehicles:
                        desired_gap = standstill_gap + (ego_speed * 1.2)

                        # Standstill Takeoff: Ego slow/stopped, front car starts rolling
                        if ego_speed < 1.0 and leader_speed > 0.5 and leader_gap > 4.5:
                            target_speed = min(leader_speed, 4.0)
                        elif leader_gap < desired_gap:
                            ratio = max(0.0, (leader_gap - standstill_gap) / max(1.0, (desired_gap - standstill_gap)))
                            target_speed = leader_speed * ratio
                        else:
                            headway_ratio = min(1.0, (leader_gap - desired_gap) / 30.0)
                            target_speed = leader_speed + headway_ratio * (speed_limit - leader_speed)
                    else:
                        target_speed = min(MAX_SPEED, speed_limit)

                    target_speed = max(0.0, min(speed_limit, target_speed))

                    # 2. Realistic, Balanced Automotive Sensor Jitter
                    noisy_ego_speed = max(0.0, ego_speed + random.gauss(0, 0.15))
                    noisy_leader_speed = max(0.0, leader_speed + random.gauss(0, 0.20))
                    noisy_gap = max(0.5, leader_gap + random.gauss(0, 0.40))

                    # Derived relative metrics from noisy measurements
                    noisy_relative_speed = noisy_ego_speed - noisy_leader_speed
                    if noisy_relative_speed > 0.1:
                        noisy_ttc = min(99.0, max(0.1, noisy_gap / noisy_relative_speed))
                    else:
                        noisy_ttc = 99.0

                    writer.writerow([
                        round(noisy_ego_speed, 2),
                        round(noisy_leader_speed, 2),
                        round(noisy_gap, 2),
                        round(noisy_relative_speed, 2),
                        round(noisy_ttc, 2),
                        round(speed_limit, 2),
                        v2v_flag,
                        round(target_speed, 2)
                    ])

                    samples_collected += 1
                    if samples_collected % 5000 == 0:
                        print(f"Captured [{samples_collected}/{target_samples}] samples from SUMO...")

                v2v.step_cleanup()

            traci.close()

    print(f"\nDone! Successfully collected {samples_collected} calibrated SUMO samples at:\n{DATASET_RAW_PATH}")


if __name__ == "__main__":
    run_sumo_data_generation()