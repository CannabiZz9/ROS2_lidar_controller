import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
import numpy as np

class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot3_controller')
        # Set QoS profile to Best Effort
        qos_profile = QoSProfile(
            depth=10,  # Queue size
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(LaserScan, 'scan', self.lidar_collision_callback, qos_profile)
        self.create_subscription(LaserScan, 'pcscan', self.lidar_control_callback, 10)
        self.stop_client = self.create_client(Trigger, 'emergency_stop')
        
        self.speed_factor = 1.0
        self.control_direction = None
        self.collision_front = 1.0
        self.collision_back = 1.0
        self.is_emergency_stop = False
        
        self.timer = self.create_timer(0.1, self.update_movement)
        self.get_logger().info("TurtleBot3 Controller Initialized")

    def lidar_collision_callback(self, msg):
        lidar_data = np.array(msg.ranges)
        lidar_data = lidar_data[~np.isnan(lidar_data)]  # Remove NaN values

        if lidar_data.size == 0:
            self.get_logger().warn("No valid lidar data available.")
            return
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(lidar_data))
        angles_mod = np.mod(angles, 2 * np.pi)
    
        back_mask = (angles_mod >= 4*np.pi/6) & (angles_mod <= 8*np.pi/6)  # 120° to 240°
        front_mask = (angles_mod <= 2*np.pi/6) | (angles_mod >= 10*np.pi/6)  # -60° to 60° (equivalent to 300° to 60°)
        
        # Find the index of the minimum distance
        front_min_idx = np.argmin(lidar_data[front_mask]) if np.any(front_mask) else None
        back_min_idx = np.argmin(lidar_data[back_mask]) if np.any(back_mask) else None
        
        front_min_angle = np.degrees(angles[front_mask][front_min_idx]) if front_min_idx is not None else None
        back_min_angle = np.degrees(angles[back_mask][back_min_idx]) if back_min_idx is not None else None
        
        front_min = np.min(lidar_data[front_mask]) if np.any(front_mask) else float('inf')
        back_min = np.min(lidar_data[back_mask]) if np.any(back_mask) else float('inf')
    
        # Emergency stop logic
        if front_min < 0.15 or back_min < 0.15:  # Check if obstacles are within 150 mm
            if not self.is_emergency_stop:  # Trigger only once
                self.get_logger().warn("Obstacle within 150 mm! Full emergency stop triggered.")
                self.call_emergency_stop()
                self.is_emergency_stop = True  # Lock emergency stop
        elif front_min < 0.22:
            self.collision_front = 0.0
            self.get_logger().warn("Obstacle in Behide within 150 mm! stop triggered.")
            self.get_logger().info(f"Distance: {front_min:.2f} meters at {front_min_angle:.2f} degrees")
            self.call_emergency_stop()
        elif back_min < 0.22:
            self.collision_back = 0.0
            self.get_logger().warn("Obstacle Front within 150 mm! stop triggered.")
            self.get_logger().info(f"Distance: {back_min:.2f} meters at {back_min_angle:.2f} degrees")
        else:
            self.is_emergency_stop = False  # Allow movement if at least one direction is clear
            self.collision_back = 1.0
            self.collision_front = 1.0

       

        

    def lidar_control_callback(self, msg):
        lidar_data = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(lidar_data))
        angles_mod = np.mod(angles, 2 * np.pi)
        
        front_mask = (angles_mod <= np.pi/6) | (angles_mod >= 11*np.pi/6)
        right_mask = (angles_mod >= np.pi/3) & (angles_mod <= 2*np.pi/3)
        back_mask = (angles_mod >= 5*np.pi/6) & (angles_mod <= 7*np.pi/6)
        left_mask = (angles_mod >= 4*np.pi/3) & (angles_mod <= 5*np.pi/3)
        
        sector_distances = {
            "Back": np.min(lidar_data[back_mask]) if np.any(back_mask) else float('inf'),
            "Left": np.min(lidar_data[left_mask]) if np.any(left_mask) else float('inf'),
            "Front": np.min(lidar_data[front_mask]) if np.any(front_mask) else float('inf'),
            "Right": np.min(lidar_data[right_mask]) if np.any(right_mask) else float('inf'),
        }
        
        dominant_direction = min(sector_distances, key=sector_distances.get)
        dominant_distance = sector_distances[dominant_direction]
        
        self.speed_factor = max(0.0, min(1.5, 1.5 - ((dominant_distance - 0.15) / 0.15) * 1.5)) if np.isfinite(dominant_distance) else 0.0
        self.control_direction = dominant_direction if self.speed_factor > 0 else None
        
    def update_movement(self):
        twist = Twist()
        
        if self.is_emergency_stop:
            twist.linear.x = 0.0  # Stop movement
        else:
            if self.control_direction == "Front":
                twist.linear.x = 0.1 * self.speed_factor * self.collision_front
            elif self.control_direction == "Back":
                twist.linear.x = -0.1 * self.speed_factor * self.collision_back
            elif self.control_direction == "Left":
                twist.angular.z = 0.5 * self.speed_factor
            elif self.control_direction == "Right":
                twist.angular.z = -0.5 * self.speed_factor
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        
        self.publisher.publish(twist)

    def call_emergency_stop(self):
        if self.stop_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            future = self.stop_client.call_async(request)
            self.get_logger().warn("Emergency Stop Service Called!")
        else:
            self.get_logger().error("Emergency Stop Service Unavailable!")

def main(args=None):
    rclpy.init()
    node = TurtleBotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.call_emergency_stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

