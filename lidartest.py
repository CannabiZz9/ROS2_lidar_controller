import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class SimpleListener(Node):
    def __init__(self):
        super().__init__('simple_listener')

        # Set QoS profile to Best Effort
        qos_profile = QoSProfile(
            depth=10,  # Queue size
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.subscription = self.create_subscription(
            LaserScan, 
            'scan', 
            self.lidar_callback, 
            qos_profile
        )

    def lidar_callback(self, msg):
        lidar_data = np.array(msg.ranges)
        lidar_data = lidar_data[~np.isnan(lidar_data)]  # Filter out NaNs

        if lidar_data.size == 0:
            self.get_logger().warn("No valid lidar data available.")
            return

        angles = np.linspace(msg.angle_min, msg.angle_max, len(lidar_data))
        angles_mod = np.mod(angles, 2 * np.pi)

        front_mask = (angles_mod >= 5*np.pi/6) & (angles_mod <= 7*np.pi/6)
        back_mask = (angles_mod <= np.pi/6) | (angles_mod >= 11*np.pi/6)

        # Find the index of the minimum distance
        front_min_idx = np.argmin(lidar_data[front_mask]) if np.any(front_mask) else None
        back_min_idx = np.argmin(lidar_data[back_mask]) if np.any(back_mask) else None

        # Calculate the angles in degrees for the minimum distances
        front_min_angle = np.degrees(angles[front_mask][front_min_idx]) if front_min_idx is not None else None
        back_min_angle = np.degrees(angles[back_mask][back_min_idx]) if back_min_idx is not None else None

        front_min = np.min(lidar_data[front_mask]) if np.any(front_mask) else float('inf')
        back_min = np.min(lidar_data[back_mask]) if np.any(back_mask) else float('inf')

        self.collision_front = 0.0 if front_min < 0.22 else 1.0
        self.collision_back = 0.0 if back_min < 0.22 else 1.0

        self.get_logger().info(f"Front Minimum distance: {front_min:.2f} meters at {front_min_angle:.2f} degrees")
        self.get_logger().info(f"Back Minimum distance: {back_min:.2f} meters at {back_min_angle:.2f} degrees")

def main(args=None):
    rclpy.init(args=args)
    node = SimpleListener()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

