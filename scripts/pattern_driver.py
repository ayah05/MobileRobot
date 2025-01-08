#!/usr/bin/env python
import rospy
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib


class PatternDriver:
    def __init__(self):
        rospy.init_node('pattern_driver', anonymous=True)

        # Initialize the move_base action client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")

        # Define default parameters
        self.speed = 0.5
        self.radius = 1.0

    def send_goal(self, x, y, theta):
        """Send a navigation goal to the move_base action server."""
        rospy.loginfo(f"Sending goal to (x={x}, y={y}, theta={theta})")

        # Create the goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        # Convert theta to quaternion
        quaternion = self.theta_to_quaternion(theta)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        # Send the goal and wait for the result
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        rospy.loginfo(f"Reached goal: (x={x}, y={y}, theta={theta})")

    @staticmethod
    def theta_to_quaternion(theta):
        """Convert an angle (theta) to a quaternion."""
        return [0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)]

    def drive_circle(self, clockwise):
        """Drive in a circular pattern using move_base."""
        rospy.loginfo(f"Driving a circle, {'clockwise' if clockwise else 'counterclockwise'}")
        step = 0.5 if clockwise else -0.5
        for angle in range(0, 360, 30):
            theta = math.radians(angle)
            x = self.radius * math.cos(theta)
            y = self.radius * math.sin(theta)
            self.send_goal(x, y, theta)

    def drive_rectangle(self):
        """Drive in a rectangular pattern."""
        rospy.loginfo("Driving a rectangle")
        rectangle_points = [
            (1.0, 0.0, 0.0),  # Move forward
            (1.0, 1.0, math.pi / 2),  # Turn 90 degrees and move up
            (0.0, 1.0, math.pi),  # Turn 90 degrees and move backward
            (0.0, 0.0, -math.pi / 2),  # Turn 90 degrees and return to start
        ]
        for point in rectangle_points:
            self.send_goal(*point)

    def run_patterns(self):
        """Run all patterns."""
        try:
            self.drive_circle(clockwise=False)
            self.drive_circle(clockwise=True)
            self.drive_rectangle()
        except rospy.ROSInterruptException:
            rospy.loginfo("Pattern driving interrupted")


if __name__ == '__main__':
    driver = PatternDriver()
    driver.run_patterns()

