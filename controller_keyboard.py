import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import sys, select, termios, tty
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import asyncio
import threading
import time
import os

sys.stdout.flush()

class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot3_controller')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
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

        # Collision detection flags
        self.collision_front = False  # No collision in front (initially free to move)
        self.collision_back = False   # No collision in back (initially free to move)
        
        if os.isatty(sys.stdin.fileno()):
            self.settings = termios.tcgetattr(sys.stdin)
        else:
            self.settings = None  # Avoid using termios in non-TTY environments
        self.speed = 1.0  # Default speed
        self.min_distance = float('inf')
        self.get_logger().info("TurtleBot3 Controller Initialized. Use 'w/a/s/d/x' to move, '[' to increase speed, ']' to decrease speed.\n")

    def lidar_callback(self, msg):
        """ Callback function for processing LiDAR data. """
        lidar_data = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(lidar_data))

        # Find the minimum distance in the front and back sectors
        front_mask = (angles >= -np.pi/6) & (angles <= np.pi/6)
        back_mask = (angles >= 5*np.pi/6) & (angles <= 7*np.pi/6)

        front_min = np.min(lidar_data[front_mask]) if np.any(front_mask) else float('inf')
        back_min = np.min(lidar_data[back_mask]) if np.any(back_mask) else float('inf')

        # Set collision flags based on minimum distances in front and back sectors
        if front_min < 0.22:
            self.collision_front = True  # Collision detected in front
        else:
            self.collision_front = False  # No collision in front

        if back_min < 0.22:
            self.collision_back = True  # Collision detected in back
        else:
            self.collision_back = False  # No collision in back

    async def get_key_async(self):
        tty.setraw(sys.stdin.fileno())
        while True:
            key = await asyncio.to_thread(select.select, [sys.stdin], [], [], 0)
            if key[0]:
                return sys.stdin.read(1)

    def get_key(self):
        if self.settings is None:
            return ''  # Avoid errors in non-TTY environments

        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    async def run(self):
        twist = Twist()
        try:
            while rclpy.ok():
                key = await self.get_key_async()

                self.get_logger().info(f"Current Min Distance: {self.min_distance:.2f} meters\n")

                if key == 'w':  # Move forward
                    if not self.collision_front:  # Only move forward if no collision in front
                        twist.linear.x = 0.1 * self.speed
                        twist.angular.z = 0.0
                    else:
                        twist.linear.x = 0.0  # Stop moving forward if collision in front

                elif key == 'a':  # Turn left
                    twist.linear.x = 0.0
                    twist.angular.z = 0.5 * self.speed

                elif key == 's':  # Stop moving (no forward/backward movement)
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                elif key == 'd':  # Turn right
                    twist.linear.x = 0.0
                    twist.angular.z = -0.5 * self.speed

                elif key == 'x':  # Move backward
                    if not self.collision_back:  # Only move backward if no collision in back
                        twist.linear.x = -0.1 * self.speed
                        twist.angular.z = 0.0
                    else:
                        twist.linear.x = 0.0  # Stop moving backward if collision in back

                elif key == '[':  # Increase speed
                    self.speed += 0.1
                    self.get_logger().info(f"Speed increased to {self.speed:.2f}\n")

                elif key == ']':  # Decrease speed
                    self.speed = max(0.1, self.speed - 0.1)
                    self.get_logger().info(f"Speed decreased to {self.speed:.2f}\n")

                elif key == '\x03':  # Ctrl+C to stop
                    break

                self.publisher.publish(twist)
                await asyncio.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
        finally:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            self.get_logger().info("Shutting down TurtleBot3 Controller.\n")

def spin_thread(node):
    """Function to run ROS2 spin in a separate thread."""
    rclpy.spin(node)

def main(args=None):
    rclpy.init()
    node = TurtleBotController()

    # Run ROS2 spin in a separate thread
    spin_threading = threading.Thread(target=spin_thread, args=(node,))
    spin_threading.start()

    # Run the async method for controlling the robot
    asyncio.run(node.run())

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

