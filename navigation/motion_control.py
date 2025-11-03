"""
Path Planning and Search Behavior
Handles pipeline loss and search patterns
"""

import numpy as np
import time
from controller import ControlCommand


class PathPlanner:
    """
    State machine for navigation behavior.
    Manages transitions between SEARCHING, FOLLOWING, and LOST states.
    """
    
    def __init__(self, lost_threshold=30):
        self.state = "SEARCHING"
        self.lost_counter = 0
        self.lost_threshold = lost_threshold
        
        self.search_start_time = time.time()
        
    def update_state(self, detection_found):
        """
        Update state based on detection status.
        
        Args:
            detection_found: True if pipeline detected in current frame
            
        Returns:
            Current state string
        """
        if detection_found:
            self.state = "FOLLOWING"
            self.lost_counter = 0
        else:
            self.lost_counter += 1
            
            if self.lost_counter > self.lost_threshold:
                self.state = "LOST"
            elif self.state == "FOLLOWING":
                self.state = "SEARCHING"
        
        return self.state
    
    def get_search_command(self):
        """
        Generate search pattern when pipeline is not visible.
        Uses sweeping motion to relocate pipeline.
        
        Returns:
            ControlCommand for search behavior
        """
        cmd = ControlCommand()
        
        # Slow forward motion with oscillating rotation
        cmd.surge = 0.2
        
        search_time = time.time() - self.search_start_time
        cmd.yaw = 0.3 * np.sin(search_time * 0.5)
        
        return cmd
    
    def compute_waypoint_heading(self, current_pos, waypoint):
        """
        Calculate required heading to reach waypoint.
        
        Args:
            current_pos: (x, y) in meters
            waypoint: (x, y) target position
            
        Returns:
            Heading in degrees [0, 360)
        """
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        
        target_heading = np.degrees(np.arctan2(dy, dx))
        
        if target_heading < 0:
            target_heading += 360
        
        return target_heading
    
    def get_distance_to_waypoint(self, current_pos, waypoint):
        """Calculate Euclidean distance to waypoint."""
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        return np.sqrt(dx**2 + dy**2)
    
    def is_waypoint_reached(self, current_pos, waypoint, tolerance=1.0):
        """
        Check if within acceptance radius of waypoint.
        
        Args:
            current_pos: Current (x, y) position
            waypoint: Target (x, y) position
            tolerance: Acceptance radius in meters
            
        Returns:
            True if waypoint reached
        """
        distance = self.get_distance_to_waypoint(current_pos, waypoint)
        return distance < tolerance

"""
if __name__ == "__main__":
    from controller import PipelineController
    
    planner = PathPlanner(lost_threshold=30)
    controller = PipelineController()
    
    print("Testing state transitions:")
    
    detections = [
        (1, True, "Pipeline visible"),
        (2, True, "Following"),
        (3, False, "Lost detection"),
        (4, False, "Searching..."),
        (5, False, "Still searching..."),
    ]
    
    for frame, detected, note in detections:
        state = planner.update_state(detected)
        print(f"Frame {frame}: {note} -> State: {state}")
        
        if state == "LOST":
            search_cmd = planner.get_search_command()
            print(f"  Search command: surge={search_cmd.surge:.2f}, yaw={search_cmd.yaw:.2f}")
    
    print("\nTesting waypoint navigation:")
    current = (0.0, 0.0)
    waypoint = (10.0, 5.0)
    
    heading = planner.compute_waypoint_heading(current, waypoint)
    distance = planner.get_distance_to_waypoint(current, waypoint)
    reached = planner.is_waypoint_reached(current, waypoint)
"""
    
    print(f"Current: {current}, Target: {waypoint}")
    print(f"Heading: {heading:.1f}°, Distance: {distance:.2f}m, Reached: {reached}")
