import traci


class LongitudinalController:
    SPEED_MODE_SAFE_ML = 31

    COLOR_CRUISE = (0, 255, 0, 255)
    COLOR_EMERGENCY = (255, 0, 0, 255)

    @staticmethod
    def initialize_vehicle(ego_id: str) -> None:
        try:
            if ego_id in traci.vehicle.getIDList():
                # SpeedMode 31 - External speed control smoothly accept karne ke liye
                traci.vehicle.setSpeedMode(ego_id, LongitudinalController.SPEED_MODE_SAFE_ML)
                traci.vehicle.setEmergencyDecel(ego_id, 9.0)
                traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_CRUISE)
        except traci.TraCIException:
            pass

    @staticmethod
    def apply_control(ego_id: str, target_speed: float, duration: float = 0.5) -> None:
        try:
            if ego_id not in traci.vehicle.getIDList():
                return

            if target_speed <= 1.5:
                traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_EMERGENCY)
                traci.vehicle.setEmergencyDecel(ego_id, 9.0)
                traci.vehicle.slowDown(ego_id, 0.0, duration=0.2)

            else:
                traci.vehicle.setColor(ego_id, LongitudinalController.COLOR_CRUISE)
                traci.vehicle.slowDown(ego_id, max(0.0, float(target_speed)), duration=duration)

        except traci.TraCIException:
            pass