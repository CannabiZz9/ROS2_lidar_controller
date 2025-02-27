import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
import threading
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')
        qos_profile = QoSProfile(
            depth=10,  # Queue size
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Subscribe to the LiDAR data for both topics
        self.lidar_subscription = self.create_subscription(LaserScan, 'scan', self.lidar_callback_scan, qos_profile)
        self.lidar_subscription_pcscan = self.create_subscription(LaserScan, 'pcscan', self.lidar_callback_pcscan, qos_profile)
        
        self.lidar_data_scan = None
        self.lidar_data_pcscan = None
        self.lock = threading.Lock()

        # Enable interactive mode for matplotlib
        plt.ion()

        # Create real-time plot with GridSpec layout
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])  # Create a grid for 2 rows and 2 columns
        self.ax_lidar_scan = self.fig.add_subplot(self.gs[0, 0], projection='polar')  # Top left plot for "scan"
        self.ax_lidar_pcscan = self.fig.add_subplot(self.gs[0, 1], projection='polar')  # Top right plot for "pcscan"
        self.ax_activation = self.fig.add_subplot(self.gs[1, :])  # Bottom plot for activation bar

        self.lidar_plot_scan, = self.ax_lidar_scan.plot([], [], 'ro', markersize=2)
        self.lidar_plot_pcscan, = self.ax_lidar_pcscan.plot([], [], 'bo', markersize=2)

        # Set up activation bar plot
        self.ax_activation.set_title("Strongest Activation Direction")
        self.bars = self.ax_activation.bar(["Back", "Left", "Front", "Right"], [0, 0, 0, 0], color='red')
        self.ax_activation.set_ylim(0, 1.5)
        self.ax_activation.set_ylabel("Activation Level")

        # Timer for updating plot
        self.timer = self.create_timer(0.05, self.update_plot)

        self.get_logger().info("LiDAR Visualizer Started.")

    def lidar_callback_scan(self, msg):
        """ Callback function for processing LiDAR data from 'scan'. """
        lidar_data = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(lidar_data))
        with self.lock:
            self.lidar_data_scan = (angles, lidar_data)

    def lidar_callback_pcscan(self, msg):
        """ Callback function for processing LiDAR data from 'pcscan'. """
        lidar_data = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(lidar_data))
        with self.lock:
            self.lidar_data_pcscan = (angles, lidar_data)

    def update_plot(self):
        """ Function to update plots in real-time. """
        with self.lock:
            if self.lidar_data_scan is not None:
                angles_scan, lidar_data_scan = self.lidar_data_scan

                # Filter valid distances (0.15m - 0.30m, ignore inf)
                valid_mask_scan = (lidar_data_scan >= 0.15) & (lidar_data_scan <= 0.30) & np.isfinite(lidar_data_scan)
                filtered_angles_scan = angles_scan[valid_mask_scan]
                filtered_distances_scan = lidar_data_scan[valid_mask_scan]

                # Update LiDAR plot for 'scan'
                self.lidar_plot_scan.set_data(filtered_angles_scan, filtered_distances_scan)
                self.ax_lidar_scan.relim()
                self.ax_lidar_scan.autoscale_view()

            if self.lidar_data_pcscan is not None:
                angles_pcscan, lidar_data_pcscan = self.lidar_data_pcscan

                # Filter valid distances (0.15m - 0.30m, ignore inf)
                valid_mask_pcscan = (lidar_data_pcscan >= 0.15) & (lidar_data_pcscan <= 0.30) & np.isfinite(lidar_data_pcscan)
                filtered_angles_pcscan = angles_pcscan[valid_mask_pcscan]
                filtered_distances_pcscan = lidar_data_pcscan[valid_mask_pcscan]

                # Update LiDAR plot for 'pcscan'
                self.lidar_plot_pcscan.set_data(filtered_angles_pcscan, filtered_distances_pcscan)
                self.ax_lidar_pcscan.relim()
                self.ax_lidar_pcscan.autoscale_view()

            # Update activation bar plot using data from 'pcscan'
            if self.lidar_data_pcscan is not None:
                self.update_activation(lidar_data_pcscan, angles_pcscan)

            # Redraw the plot
            plt.draw()
            plt.pause(0.05)  # Ensures the plot is updated

    def update_activation(self, lidar_data, angles):
        """ Determines strongest activation direction from 'pcscan'. """
        angles_mod = np.mod(angles, 2 * np.pi)

        # Define angle ranges
        back_mask = (angles_mod <= np.pi/6) | (angles_mod >= 11*np.pi/6)
        right_mask = (angles_mod >= np.pi/3) & (angles_mod <= 2*np.pi/3)
        front_mask = (angles_mod >= 5*np.pi/6) & (angles_mod <= 7*np.pi/6)
        left_mask = (angles_mod >= 4*np.pi/3) & (angles_mod <= 5*np.pi/3)

        # Find closest valid distances per direction
        back_min = np.min(lidar_data[back_mask]) if np.any(back_mask) else float('inf')
        left_min = np.min(lidar_data[left_mask]) if np.any(left_mask) else float('inf')
        front_min = np.min(lidar_data[front_mask]) if np.any(front_mask) else float('inf')
        right_min = np.min(lidar_data[right_mask]) if np.any(right_mask) else float('inf')

        sector_distances = {"Back": back_min, "Left": left_min, "Front": front_min, "Right": right_min}
        dominant_direction = min(sector_distances, key=sector_distances.get)  # Selects smallest valid distance
        dominant_distance = sector_distances[dominant_direction]

        # Compute activation (ignore inf values)
        activation_value = max(0.0, min(1.5, 1.5 - ((dominant_distance - 0.15) / 0.15) * 1.5)) if np.isfinite(dominant_distance) else 0.0

        # Update activation bar plot
        for bar, label in zip(self.bars, ["Back", "Left", "Front", "Right"]):
            bar.set_height(activation_value if label == dominant_direction else 0.0)

    def stop(self):
        """ Stop the node. """
        self.timer.cancel()
        plt.ioff()  # Turn off interactive mode
        plt.close()  # Close the figure when stopping

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

