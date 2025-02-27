import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist

class EmergencyStopService(Node):
    def __init__(self):
        super().__init__('emergency_stop_service')
        
        # Create the service
        self.srv = self.create_service(Trigger, 'emergency_stop', self.emergency_stop_callback)

        # Publisher for stopping the robot
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info("Emergency Stop Service Ready")

    def emergency_stop_callback(self, request, response):
        """ Callback function to stop the robot when the service is called. """
        self.get_logger().warn("Emergency Stop Activated! Stopping the robot.")

        # Publish a zero velocity message to stop the robot
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(stop_msg)

        # Respond with success
        response.success = True
        response.message = "Robot has stopped successfully."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = EmergencyStopService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Emergency Stop Service")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

