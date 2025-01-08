#!/usr/bin/env python
import rospy
import csv
from nav_msgs.msg import Odometry

class PositionLogger:
    def __init__(self):
        rospy.init_node('position_logger', anonymous=True)
        self.robot_name = rospy.get_param("~robot_name", "mir_100")

        # Fallback to /tmp directory if output_file is not found
        fallback_output_file = "/tmp/position_log.csv"
        self.output_file = rospy.get_param("~output_file", fallback_output_file)

        # Ensure the output directory exists
        import os
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Open CSV file for logging
        self.csv_file = open(self.output_file, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'x', 'y', 'z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
            'linear_velocity_x', 'linear_velocity_y', 'linear_velocity_z',
            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'
        ])

        # Subscribe to /odom topic
        rospy.Subscriber('/odom', Odometry, self.callback)
        rospy.loginfo(f"Logging odometry data for {self.robot_name} to {self.output_file}")

    def callback(self, msg):
        # Extract pose and twist data from Odometry
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        linear_velocity = msg.twist.twist.linear
        angular_velocity = msg.twist.twist.angular

        # Write data to CSV
        self.csv_writer.writerow([
            round(rospy.Time.now().to_sec(), 3),
            round(position.x, 3), round(position.y, 3), round(position.z, 3),
            round(orientation.x, 3), round(orientation.y, 3), round(orientation.z, 3), round(orientation.w, 3),
            round(linear_velocity.x, 3), round(linear_velocity.y, 3), round(linear_velocity.z, 3),
            round(angular_velocity.x, 3), round(angular_velocity.y, 3), round(angular_velocity.z, 3)
        ])

    def run(self):
        rospy.spin()
        self.csv_file.close()


if __name__ == '__main__':
    logger = PositionLogger()
    try:
        logger.run()
    except rospy.ROSInterruptException:
        pass
