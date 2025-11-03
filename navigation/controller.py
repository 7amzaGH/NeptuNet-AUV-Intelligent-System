"""
Pipeline Following Controller
Converts YOLO bounding box detections to motor commands
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class ControlCommand:
    """Motor control commands (normalized -1 to 1)"""
    surge: float = 0.0      # Forward/Backward
    sway: float = 0.0       # Right/Left
    heave: float = 0.0      # Up/Down
    yaw: float = 0.0        # Rotation

class PipelineController:
    """
    Generates control commands to center pipeline in camera view
    while maintaining forward motion.
    """
    
    def __init__(self, frame_width=640, frame_height=480, 
                 tolerance=20, camera_angle=45.0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tolerance = tolerance
        self.camera_angle = camera_angle
        
        self.target_pipeline_width = 0.3  # 30% of frame
        
        # Control gains - tuned experimentally
        self.kp_horizontal = 0.01
        self.kp_vertical = 0.01
        self.kp_distance = 0.5
        
    def compute_control(self, box, depth_data=None):
        """
        Convert bounding box to control commands.
        
        Args:
            box: [x1, y1, x2, y2] from YOLO detection
            depth_data: dict with 'depth' and 'target_depth' keys
            
        Returns:
            ControlCommand with normalized values
        """
        cmd = ControlCommand()
        
        if box is None or len(box) != 4:
            return cmd
        
        # Calculate box center and dimensions
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        box_width = box[2] - box[0]
        
        frame_center_x = self.frame_width / 2
        frame_center_y = self.frame_height / 2
        
        # Position errors
        error_x = x_center - frame_center_x
        error_y = y_center - frame_center_y
        
        # Constant forward motion when pipeline detected
        cmd.surge = 0.5
        
        # Horizontal centering
        if abs(error_x) > self.tolerance:
            cmd.sway = np.clip(error_x * self.kp_horizontal, -1.0, 1.0)
        
        # Vertical centering (depends on camera orientation)
        if abs(error_y) > self.tolerance:
            if self.camera_angle > 60:
                # Downward camera: adjust forward speed
                speed_adj = -error_y * self.kp_vertical * 0.3
                cmd.surge += np.clip(speed_adj, -0.2, 0.2)
            else:
                # Forward camera: adjust altitude
                cmd.heave = np.clip(-error_y * self.kp_vertical, -1.0, 1.0)
        
        # Safety: slow down if too close
        desired_width = self.frame_width * self.target_pipeline_width
        if box_width > desired_width * 1.5:
            cmd.surge *= 0.5
        
        # Depth control
        if depth_data:
            depth_error = depth_data['depth'] - depth_data['target_depth']
            if abs(depth_error) > 0.2:
                cmd.heave += np.clip(depth_error * 0.5, -0.3, 0.3)
        
        # Yaw correction for large lateral errors
        if abs(error_x) > self.frame_width * 0.2:
            cmd.yaw = np.clip(error_x * 0.005, -0.3, 0.3)
        
        # Final clipping
        cmd.surge = np.clip(cmd.surge, -1.0, 1.0)
        cmd.sway = np.clip(cmd.sway, -1.0, 1.0)
        cmd.heave = np.clip(cmd.heave, -1.0, 1.0)
        cmd.yaw = np.clip(cmd.yaw, -1.0, 1.0)
        
        return cmd
    
    def get_binary_commands(self, box, depth_data=None):
        """
        Simple on/off commands for basic thruster control.
        
        Returns:
            Dictionary with binary direction commands
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


if __name__ == "__main__":
    controller = PipelineController(frame_width=640, frame_height=480)
    
    # Test with sample detection
    bbox = [200, 150, 400, 350]
    depth = {'depth': 2.3, 'target_depth': 2.0}
    
    cmd = controller.compute_control(bbox, depth)
    print(f"Control: surge={cmd.surge:.2f}, sway={cmd.sway:.2f}, "
          f"heave={cmd.heave:.2f}, yaw={cmd.yaw:.2f}")
    
    binary = controller.get_binary_commands(bbox, depth)
    print(f"Binary: {binary}")
