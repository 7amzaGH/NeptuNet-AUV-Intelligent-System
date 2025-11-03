"""
path_planning.py - Navigation State Machine
Handles what to do when pipeline is lost, waypoint navigation
"""

import numpy as np
import time


class PathPlanner:
    """
    High-level navigation decision maker
    Decides between: SEARCHING, FOLLOWING, LOST states
    """
    
    def __init__(self, lost_threshold=30):
        """
        Args:
            lost_threshold: How many frames without detection before "LOST" state
        """
        # Current state
        self.state = "SEARCHING"  # SEARCHING, FOLLOWING, LOST
        
        # Lost pipeline tracking
        self.lost_counter = 0
        self.lost_threshold = lost_threshold
        
        # Search pattern variables
        self.search_time = 0.0
        self.search_start_time = time.time()
        
    def update_state(self, detection_found):
        """
        Update navigation state based on detection
        
        Args:
            detection_found: True if YOLO detected pipeline in frame
            
        Returns:
            Current state string
        """
        if detection_found:
            # Pipeline found! Follow it
            self.state = "FOLLOWING"
            self.lost_counter = 0
        else:
            # No detection
            self.lost_counter += 1
            
            if self.lost_counter > self.lost_threshold:
                # Lost for too long, start searching
                self.state = "LOST"
            elif self.state == "FOLLOWING":
                # Just lost, but not long enough to declare "LOST"
                self.state = "SEARCHING"
        
        return self.state
    
    def get_search_command(self):
        """
        Generate search pattern when pipeline is lost
        Creates a sweep/spiral pattern to find pipeline again
        
        Returns:
            ControlCommand for search behavior
        """
        #from controller import ControlCommand
        cmd = ControlCommand()
        
        # Strategy: Move forward slowly while rotating (sweep pattern)
        cmd.surge = 0.2  # Slow forward movement
        
        # Oscillating yaw for sweeping left-right
        self.search_time = time.time() - self.search_start_time
        cmd.yaw = 0.3 * np.sin(self.search_time * 0.5)
        
        return cmd
    
    def compute_waypoint_heading(self, current_pos, waypoint):
        """
        Calculate heading needed to reach a waypoint
        Used for dead-reckoning when no visual detection
        
        Args:
            current_pos: (x, y) current position in meters
            waypoint: (x, y) target waypoint in meters
            
        Returns:
            Target heading in degrees (0-360)
        """
        # Calculate vector to waypoint
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        
        # Calculate angle (0° = East, 90° = North)
        target_heading = np.degrees(np.arctan2(dy, dx))
        
        # Normalize to 0-360
        if target_heading < 0:
            target_heading += 360
        
        return target_heading
    
    def get_distance_to_waypoint(self, current_pos, waypoint):
        """
        Calculate distance to waypoint
        
        Args:
            current_pos: (x, y) current position
            waypoint: (x, y) target waypoint
            
        Returns:
            Distance in meters
        """
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        return np.sqrt(dx**2 + dy**2)
    
    def is_waypoint_reached(self, current_pos, waypoint, tolerance=1.0):
        """
        Check if waypoint is reached
        
        Args:
            current_pos: (x, y) current position
            waypoint: (x, y) target waypoint
            tolerance: Acceptance radius in meters
            
        Returns:
            True if within tolerance
        """
        distance = self.get_distance_to_waypoint(current_pos, waypoint)
        return distance < tolerance

"""
# === EXAMPLE USAGE ===
if __name__ == "__main__":
    #from controller import PipelineController
    
    # Initialize
    planner = PathPlanner(lost_threshold=30)
    controller = PipelineController()
    
    print("=== SCENARIO: Pipeline Following ===")
    
    # Simulate detection over time
    detections = [
        (1, True, "Pipeline visible"),
        (2, True, "Still following"),
        (3, False, "Lost detection!"),
        (4, False, "Still searching..."),
        (5, False, "Still searching..."),
    ]
    
    for frame, detected, note in detections:
        state = planner.update_state(detected)
        print(f"Frame {frame}: {note} → State: {state}")
        
        if state == "LOST":
            # Generate search command
            search_cmd = planner.get_search_command()
            print(f"  → Search: surge={search_cmd.surge:.2f}, yaw={search_cmd.yaw:.2f}")
    
    print("\n=== SCENARIO: Waypoint Navigation ===")
    
    # Current position and target waypoint
    current = (0.0, 0.0)
    waypoint = (10.0, 5.0)
    
    # Calculate navigation
    heading = planner.compute_waypoint_heading(current, waypoint)
    distance = planner.get_distance_to_waypoint(current, waypoint)
    reached = planner.is_waypoint_reached(current, waypoint, tolerance=1.0)
    
    print(f"Current position: {current}")
    print(f"Target waypoint: {waypoint}")
    print(f"Required heading: {heading:.1f}°")
    print(f"Distance: {distance:.2f}m")
    print(f"Reached: {reached}")
"""
