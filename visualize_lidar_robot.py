import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import threading
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class LidarPolarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_polar_visualizer')
        qos_profile = QoSProfile(
            depth=10,  # Queue size
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.subscription = self.create_subscription(LaserScan, 'scan', self.lidar_callback,qos_profile)
        self.lidar_data = None
        self.lock = threading.Lock()

        # Enable interactive mode for matplotlib
        plt.ion()

        # Initialize real-time polar plot
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.lidar_plot, = self.ax.plot([], [], 'ro', markersize=2)

        # Timer for updating plot
        self.timer = self.create_timer(0.05, self.update_plot)

        self.get_logger().info("LiDAR Polar Visualizer Started.")

    def lidar_callback(self, msg):
        """ Callback function for processing LiDAR data. """
        lidar_data = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(lidar_data))
        with self.lock:
            self.lidar_data = (angles, lidar_data)

    def update_plot(self):
        """ Function to update the polar plot in real-time. """
        with self.lock:
            if self.lidar_data is not None:
                angles, lidar_data = self.lidar_data

                # Filter valid distances (ignore inf values)
                valid_mask = np.isfinite(lidar_data)
                filtered_angles = angles[valid_mask]
                filtered_distances = lidar_data[valid_mask]

                # Update LiDAR polar plot
                self.lidar_plot.set_data(filtered_angles, filtered_distances)
                self.ax.relim()
                self.ax.autoscale_view()

                # Redraw the plot
                plt.draw()
                plt.pause(0.05)  # Ensures the plot is updated

    def stop(self):
        """ Stop the node. """
        self.timer.cancel()
        plt.ioff()  # Turn off interactive mode
        plt.close()  # Close the figure when stopping


def main(args=None):
    rclpy.init(args=args)
    node = LidarPolarVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

