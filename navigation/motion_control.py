"""
Motion Stabilization with PID Control
Maintains stable orientation using IMU feedback
"""

import time
import numpy as np
from controller import ControlCommand


class PIDController:
    """
    Standard PID controller implementation.
    """
    
    def __init__(self, kp=0.5, ki=0.1, kd=0.2, output_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        
    def update(self, error):
        """
        Calculate control output.
        
        Args:
            error: Difference between setpoint and current value
            
        Returns:
            Control signal
        """
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.01
        
        # PID terms
        p_term = self.kp * error
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)  # anti-windup
        i_term = self.ki * self.integral
        
        d_term = self.kd * (error - self.prev_error) / dt
        
        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """Reset internal state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()


class MotionController:
    """
    High-level motion control with attitude stabilization.
    Uses PID feedback from IMU to maintain level orientation.
    """
    
    def __init__(self):
        # PID controllers for each axis
        self.pid_roll = PIDController(kp=0.5, ki=0.1, kd=0.2)
        self.pid_pitch = PIDController(kp=0.5, ki=0.1, kd=0.2)
        self.pid_yaw = PIDController(kp=0.3, ki=0.05, kd=0.15)
        self.pid_depth = PIDController(kp=1.0, ki=0.2, kd=0.5)
        
        # Target orientation (level)
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        
    def stabilize(self, cmd, imu_data, depth_data):
        """
        Apply stabilization corrections to control commands.
        
        Args:
            cmd: Raw ControlCommand from pipeline controller
            imu_data: dict with 'roll', 'pitch', 'yaw' in degrees
            depth_data: dict with 'depth' and 'target_depth'
            
        Returns:
            Stabilized ControlCommand
        """
        stabilized = ControlCommand()
        
        # Attitude stabilization
        roll_error = self.target_roll - imu_data['roll']
        roll_correction = self.pid_roll.update(roll_error)
        
        pitch_error = self.target_pitch - imu_data['pitch']
        pitch_correction = self.pid_pitch.update(pitch_error)
        
        stabilized.sway = cmd.sway - roll_correction * 0.3
        stabilized.surge = cmd.surge - pitch_correction * 0.3
        
        # Heading control
        yaw_error = self._normalize_angle(self.target_yaw - imu_data['yaw'])
        yaw_correction = self.pid_yaw.update(yaw_error)
        stabilized.yaw = cmd.yaw + yaw_correction
        
        # Depth control
        depth_error = depth_data['target_depth'] - depth_data['depth']
        depth_correction = self.pid_depth.update(depth_error)
        stabilized.heave = cmd.heave + depth_correction
        
        # Clip outputs
        stabilized.surge = np.clip(stabilized.surge, -1.0, 1.0)
        stabilized.sway = np.clip(stabilized.sway, -1.0, 1.0)
        stabilized.heave = np.clip(stabilized.heave, -1.0, 1.0)
        stabilized.yaw = np.clip(stabilized.yaw, -1.0, 1.0)
        
        return stabilized
    
    def set_target_heading(self, yaw):
        """Set desired heading angle in degrees."""
        self.target_yaw = yaw
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-180, 180] range."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

"""
if __name__ == "__main__":
    from controller import PipelineController
    
    controller = PipelineController()
    motion = MotionController()
    
    # Test with tilted AUV
    bbox = [200, 150, 400, 350]
    imu = {'roll': 5.0, 'pitch': -3.0, 'yaw': 90.0}
    depth = {'depth': 2.3, 'target_depth': 2.0}
    
    raw_cmd = controller.compute_control(bbox, depth)
    print(f"Raw: surge={raw_cmd.surge:.2f}, sway={raw_cmd.sway:.2f}")
    
    stable_cmd = motion.stabilize(raw_cmd, imu, depth)
    print(f"Stabilized: surge={stable_cmd.surge:.2f}, sway={stable_cmd.sway:.2f}")
"""
