import traci
from ml_engine.src.logger import logging


class EmergencyBrakingScenario:
    def __init__(self, lead_id="lead_car", trigger_step=50, stop_duration=40, resume_speed=11.0):
        self.lead_id = lead_id
        self.trigger_step = trigger_step
        self.stop_duration = stop_duration
        self.resume_speed = resume_speed
        
        self.brake_applied = False
        self.resumed = False
        self.initial_speed_set = False

    def update(self, current_step: int, active_vehicles: list, v2v_module=None):
        if self.lead_id not in active_vehicles:
            return

        try:
            if not self.initial_speed_set:
                traci.vehicle.setSpeedMode(self.lead_id, 31)
                traci.vehicle.setSpeed(self.lead_id, 13.0)
                traci.vehicle.setColor(self.lead_id, (0, 150, 255, 255))
                self.initial_speed_set = True

            lead_pos = traci.vehicle.getPosition(self.lead_id)

            if current_step >= self.trigger_step and not self.brake_applied:
                traci.vehicle.setSpeedMode(self.lead_id, 0)
                traci.vehicle.setSpeed(self.lead_id, 0.0)
                traci.vehicle.setColor(self.lead_id, (255, 0, 0, 255))
                logging.warning(f"Step {current_step}: Lead car performed sudden EMERGENCY BRAKE!")
                self.brake_applied = True

            if self.brake_applied and not self.resumed and v2v_module:
                v2v_module.broadcast(
                    sender_id=self.lead_id,
                    event_type="EMERGENCY_BRAKE",
                    speed=0.0,
                    decel=9.0,
                    pos=lead_pos
                )

            resume_step = self.trigger_step + self.stop_duration
            if current_step >= resume_step and not self.resumed:
                traci.vehicle.setSpeedMode(self.lead_id, 31)
                traci.vehicle.setSpeed(self.lead_id, self.resume_speed)
                traci.vehicle.setColor(self.lead_id, (0, 150, 255, 255))
                logging.info(f"Step {current_step}: Lead car resumed driving at {self.resume_speed} m/s.")
                self.resumed = True

        except traci.TraCIException as err:
            logging.warning(f"Scenario TraCI warning at step {current_step}: {err}")