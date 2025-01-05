#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time


class PatternDriver:
    def __init__(self):
        rospy.init_node('pattern_driver', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.speeds = [0.2, 0.4, 0.6]  # Varying speeds

    def stop_robot(self):
        """Stop the robot."""
        stop = Twist()
        self.cmd_vel_pub.publish(stop)
        rospy.loginfo("Robot stopped")
        time.sleep(1)

    def drive_circle(self, speed, clockwise):
        """Drive the robot in a circle."""
        rospy.loginfo(f"Driving a circle at speed {speed}, {'clockwise' if clockwise else 'counterclockwise'}")
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = -speed if clockwise else speed  # Negative angular speed for clockwise
        for _ in range(100):  # Approx. 10 seconds at 10 Hz
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()
        self.stop_robot()

    def drive_figure_8(self, speed):
        """Drive the robot in a figure-8."""
        rospy.loginfo(f"Driving a figure-8 at speed {speed}")
        self.drive_circle(speed, clockwise=False)  # First loop counterclockwise
        self.drive_circle(speed, clockwise=True)   # Second loop clockwise

    def drive_rectangle(self, speed, clockwise):
        """Drive the robot in a rectangle."""
        rospy.loginfo(f"Driving a rectangle at speed {speed}, {'clockwise' if clockwise else 'counterclockwise'}")
        twist = Twist()

        for _ in range(2):  # Two laps around the rectangle
            for _ in range(2):  # Two long sides
                twist.linear.x = speed
                twist.angular.z = 0
                for _ in range(50):  # Drive straight for 5 seconds
                    self.cmd_vel_pub.publish(twist)
                    self.rate.sleep()

                # Turn 90 degrees
                twist.linear.x = 0
                twist.angular.z = -0.5 if clockwise else 0.5
                for _ in range(20):  # Approx. 2 seconds to turn 90 degrees
                    self.cmd_vel_pub.publish(twist)
                    self.rate.sleep()

            for _ in range(2):  # Two short sides
                twist.linear.x = speed
                twist.angular.z = 0
                for _ in range(25):  # Drive straight for 2.5 seconds
                    self.cmd_vel_pub.publish(twist)
                    self.rate.sleep()

                # Turn 90 degrees
                twist.linear.x = 0
                twist.angular.z = -0.5 if clockwise else 0.5
                for _ in range(20):  # Approx. 2 seconds to turn 90 degrees
                    self.cmd_vel_pub.publish(twist)
                    self.rate.sleep()

        self.stop_robot()

    def run_patterns(self):
        """Run all patterns at varying speeds and in both directions."""
        try:
            for speed in self.speeds:
                self.drive_circle(speed, clockwise=False)
                self.drive_circle(speed, clockwise=True)
                self.drive_figure_8(speed)
                self.drive_rectangle(speed, clockwise=False)
                self.drive_rectangle(speed, clockwise=True)
        except rospy.ROSInterruptException:
            pass
        finally:
            self.stop_robot()
            rospy.loginfo("Finished all patterns")


if __name__ == '__main__':
    driver = PatternDriver()
    driver.run_patterns()
