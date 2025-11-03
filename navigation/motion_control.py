import time
import numpy as np


class PIDController:
    """
    Simple PID controller for one axis
    PID = Proportional + Integral + Derivative
    """
    
    def __init__(self, kp=0.5, ki=0.1, kd=0.2, output_limit=1.0):
        """
        Args:
            kp: Proportional gain (react to current error)
            ki: Integral gain (eliminate steady-state error)
            kd: Derivative gain (reduce overshoot)
            output_limit: Maximum output value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        # Internal state
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        
    def update(self, error):
        """
        Calculate PID output based on error
        
        Args:
            error: Difference between target and current value
            
        Returns:
            Control output (between -output_limit and +output_limit)
        """
        # Calculate time step
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.01  # Avoid division by zero
        
        # Proportional term: react to current error
        p_term = self.kp * error
        
        # Integral term: sum of all past errors
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)  # Anti-windup
        i_term = self.ki * self.integral
        
        # Derivative term: rate of error change
        d_term = self.kd * (error - self.prev_error) / dt
        
        # Total output
        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        # Update state for next iteration
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller (call when starting new task)"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()


class MotionController:
    """
    High-level motion control with stabilization
    Uses PID + IMU to keep AUV stable and level
    """
    
    def __init__(self):
        # Create PID controllers for each axis
        self.pid_roll = PIDController(kp=0.5, ki=0.1, kd=0.2)    # Left/right tilt
        self.pid_pitch = PIDController(kp=0.5, ki=0.1, kd=0.2)   # Forward/back tilt
        self.pid_yaw = PIDController(kp=0.3, ki=0.05, kd=0.15)   # Rotation
        self.pid_depth = PIDController(kp=1.0, ki=0.2, kd=0.5)   # Depth
        
        # Target orientation (we want level: 0° roll, 0° pitch)
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        
    def stabilize(self, cmd, imu_data, depth_data):
        """
        Apply stabilization to raw control commands
        
        Args:
            cmd: ControlCommand from controller.py
            imu_data: Dict with {'roll', 'pitch', 'yaw'} in degrees
            depth_data: Dict with {'depth', 'target_depth'} in meters
            
        Returns:
            Stabilized ControlCommand
        """
        #from controller import ControlCommand
        stabilized = ControlCommand()
        
        # === ATTITUDE STABILIZATION ===
        # Keep AUV level (roll and pitch near 0°)
        
        # Roll correction (left/right tilt)
        roll_error = self.target_roll - imu_data['roll']
        roll_correction = self.pid_roll.update(roll_error)
        
        # Pitch correction (forward/back tilt)
        pitch_error = self.target_pitch - imu_data['pitch']
        pitch_correction = self.pid_pitch.update(pitch_error)
        
        # Apply corrections
        stabilized.sway = cmd.sway - roll_correction * 0.3
        stabilized.surge = cmd.surge - pitch_correction * 0.3
        
        # === YAW CONTROL ===
        # Keep heading stable or follow commanded yaw
        yaw_error = self._normalize_angle(self.target_yaw - imu_data['yaw'])
        yaw_correction = self.pid_yaw.update(yaw_error)
        stabilized.yaw = cmd.yaw + yaw_correction
        
        # === DEPTH CONTROL ===
        # Maintain target depth
        depth_error = depth_data['target_depth'] - depth_data['depth']
        depth_correction = self.pid_depth.update(depth_error)
        stabilized.heave = cmd.heave + depth_correction
        
        # Clip all outputs to [-1, 1]
        stabilized.surge = np.clip(stabilized.surge, -1.0, 1.0)
        stabilized.sway = np.clip(stabilized.sway, -1.0, 1.0)
        stabilized.heave = np.clip(stabilized.heave, -1.0, 1.0)
        stabilized.yaw = np.clip(stabilized.yaw, -1.0, 1.0)
        
        return stabilized
    
    def set_target_heading(self, yaw):
        """
        Set desired heading angle
        
        Args:
            yaw: Target heading in degrees (0-360)
        """
        self.target_yaw = yaw
    
    def _normalize_angle(self, angle):
        """
        Normalize angle to [-180, 180] range
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

"""
# === EXAMPLE USAGE ===
if __name__ == "__main__":
    #from controller import PipelineController, ControlCommand
    
    # Initialize
    controller = PipelineController()
    motion = MotionController()
    
    # Example: Pipeline detected, but AUV is tilted
    bbox = [200, 150, 400, 350]
    imu = {'roll': 5.0, 'pitch': -3.0, 'yaw': 90.0}  # Tilted!
    depth = {'depth': 2.3, 'target_depth': 2.0}
    
    # Get raw command from controller
    raw_cmd = controller.compute_control(bbox, depth)
    print(f"Raw: surge={raw_cmd.surge:.2f}, sway={raw_cmd.sway:.2f}")
    
    # Apply stabilization
    stable_cmd = motion.stabilize(raw_cmd, imu, depth)
    print(f"Stabilized: surge={stable_cmd.surge:.2f}, sway={stable_cmd.sway:.2f}")
    print("→ Motion control corrected for roll/pitch to keep AUV level")
"""
