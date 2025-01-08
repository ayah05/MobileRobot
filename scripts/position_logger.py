#!/usr/bin/env python
import rospy
import csv
from gazebo_msgs.msg import ModelStates

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
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'])
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        rospy.loginfo(f"Logging position data for {self.robot_name} to {self.output_file}")

    def callback(self, msg):
        try:
            index = msg.name.index(self.robot_name)  # Find the index of the robot in the ModelStates array
            pose = msg.pose[index]
            position = pose.position
            orientation = pose.orientation

            # Write position and orientation data to CSV
            self.csv_writer.writerow([
                round(rospy.Time.now().to_sec(), 3),
                round(position.x, 3), round(position.y, 3), round(position.z, 3),
                round(orientation.x, 3), round(orientation.y, 3), round(orientation.z, 3), round(orientation.w, 3)
            ])
        except ValueError:
            rospy.logwarn(f"Robot {self.robot_name} not found in ModelStates!")

    def run(self):
        rospy.spin()
        self.csv_file.close()

if __name__ == '__main__':
    logger = PositionLogger()
    try:
        logger.run()
    except rospy.ROSInterruptException:
        pass
