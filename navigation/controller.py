import numpy as np
from dataclasses import dataclass


@dataclass
class ControlCommand:
    """Motor control commands (normalized -1 to 1)"""
    surge: float = 0.0      # Forward(+) / Backward(-)
    sway: float = 0.0       # Right(+) / Left(-)
    heave: float = 0.0      # Up(+) / Down(-)
    yaw: float = 0.0        # Rotate right(+) / left(-)


class PipelineController:
    
    def __init__(self, 
                 frame_width=640,
                 frame_height=480,
                 tolerance=20,
                 camera_angle=45.0):
        """
        Args:
            frame_width: Camera width in pixels
            frame_height: Camera height in pixels
            tolerance: Deadzone in pixels (ignore small errors)
            camera_angle: Camera tilt (0=forward, 90=down)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tolerance = tolerance
        self.camera_angle = camera_angle
        
        # How much of frame should pipeline occupy (30%)
        self.target_pipeline_width = 0.3
        
        # Control gains (tune these for your AUV)
        self.kp_horizontal = 0.01   # Horizontal centering
        self.kp_vertical = 0.01     # Vertical centering
        self.kp_distance = 0.5      # Distance keeping
        
    def compute_control(self, box, depth_data=None):
        """
        Main function: Convert bounding box to control commands
        Robot ALWAYS moves forward, only adjusts lateral position
        
        Args:
            box: [x1, y1, x2, y2] from YOLO detection
            depth_data: Optional dict with {'depth': float, 'target_depth': float}
            
        Returns:
            ControlCommand with values between -1 and 1
        """
        cmd = ControlCommand()
        
        # If no detection, return zero commands (path_planning will handle this)
        if box is None or len(box) != 4:
            return cmd
        
        # Calculate bounding box center and size
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        box_width = box[2] - box[0]
        
        # Frame center point
        frame_center_x = self.frame_width / 2
        frame_center_y = self.frame_height / 2
        
        # How far is pipeline from center? (in pixels)
        error_x = x_center - frame_center_x
        error_y = y_center - frame_center_y
        
        # === STEP 1: CONSTANT FORWARD MOTION ===
        # Robot always moves forward when pipeline is detected
        cmd.surge = 0.5  # Constant forward speed (adjust 0.3-0.7 based on your needs)
        
        # === STEP 2: Horizontal Centering (Left/Right) ===
        if abs(error_x) > self.tolerance:
            # Positive error = pipeline is right → move right
            cmd.sway = np.clip(error_x * self.kp_horizontal, -1.0, 1.0)
        
        # === STEP 3: Vertical Centering (Up/Down or Forward/Back) ===
        if abs(error_y) > self.tolerance:
            if self.camera_angle > 60:
                # Camera looking down: vertical error affects forward speed slightly
                # But don't stop - just slow down or speed up a bit
                speed_adjustment = -error_y * self.kp_vertical * 0.3
                cmd.surge += np.clip(speed_adjustment, -0.2, 0.2)
            else:
                # Camera looking forward: vertical error = up/down
                cmd.heave = np.clip(-error_y * self.kp_vertical, -1.0, 1.0)
        
        # === STEP 4: Distance Warning (Optional - for safety) ===
        # If pipeline is getting VERY large, slow down (too close!)
        desired_width = self.frame_width * self.target_pipeline_width
        if box_width > desired_width * 1.5:  # Pipeline is 50% larger than desired
            cmd.surge *= 0.5  # Slow down to half speed
            print(" Warning: Too close to pipeline, slowing down")
        
        # === STEP 5: Depth Control (if depth sensor available) ===
        if depth_data:
            depth_error = depth_data['depth'] - depth_data['target_depth']
            if abs(depth_error) > 0.2:  # 20cm tolerance
                cmd.heave += np.clip(depth_error * 0.5, -0.3, 0.3)
        
        # === STEP 6: Yaw Control (Rotation) ===
        # If pipeline is far off-center, rotate to face it
        if abs(error_x) > self.frame_width * 0.2:
            cmd.yaw = np.clip(error_x * 0.005, -0.3, 0.3)
        
        # Make sure all commands are within [-1, 1]
        cmd.surge = np.clip(cmd.surge, -1.0, 1.0)
        cmd.sway = np.clip(cmd.sway, -1.0, 1.0)
        cmd.heave = np.clip(cmd.heave, -1.0, 1.0)
        cmd.yaw = np.clip(cmd.yaw, -1.0, 1.0)
        
        return cmd
    
    def get_binary_commands(self, box, depth_data=None):
        """
        Simple ON/OFF commands for basic thrusters
        Returns: dict with 0 or 1 for each direction
        """
        cmd = self.compute_control(box, depth_data)
        
        return {
            "left": 1 if cmd.sway < -0.1 else 0,
            "right": 1 if cmd.sway > 0.1 else 0,
            "up": 1 if cmd.heave > 0.1 else 0,
            "down": 1 if cmd.heave < -0.1 else 0,
            "forward": 1 if cmd.surge > 0.1 else 0,
            "backward": 1 if cmd.surge < -0.1 else 0,
        }
"""
# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Initialize controller
    controller = PipelineController(
        frame_width=640,
        frame_height=480,
        camera_angle=45.0
    )
    
    # Example 1: Pipeline detected by YOLO
    bbox = [200, 150, 400, 350]  # [x1, y1, x2, y2]
    depth = {'depth': 2.3, 'target_depth': 2.0}
    
    # Get control commands
    cmd = controller.compute_control(bbox, depth)
    print(f"Control: surge={cmd.surge:.2f}, sway={cmd.sway:.2f}, "
          f"heave={cmd.heave:.2f}, yaw={cmd.yaw:.2f}")
    
    # Or get binary commands
    binary = controller.get_binary_commands(bbox, depth)
    print(f"Binary: {binary}")
    """
