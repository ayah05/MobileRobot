#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math
from geometry_msgs.msg import Pose

# Global variable to store the predicted pose
predicted_pose = None

def predicted_pose_callback(msg):
    """
    Callback function to update the predicted pose.
    """
    global predicted_pose
    # Extract position and orientation from the incoming message
    predicted_pose = (
        msg.position.x,
        msg.position.y,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w
    )


def predict_state(current_x, current_y, current_qx, current_qy, current_qz, current_qw):
    """
    Simulate the next state of the model with quaternion orientation.
    Here, we move 0.1 units in the x-direction and retain the same orientation.
    """
    global predicted_pose

    # Subscribe to the topic to get the predicted pose
    rospy.Subscriber("/predicted_pose", Pose, predicted_pose_callback)

    # Wait until the predicted_pose is updated
    while predicted_pose is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # Use the predicted pose
    next_x, next_y, next_qx, next_qy, next_qz, next_qw = predicted_pose

    return next_x, next_y, (next_qx, next_qy, next_qz, next_qw)

def main():
    rospy.init_node("model_quaternion_visualizer")

    # Publisher for the predicted path
    path_pub = rospy.Publisher("/my_model_path", Path, queue_size=10)

    # Initial state
    current_x, current_y = 0, 0
    current_orientation = (0.0, 0.0, 0.0, 1.0)  # Initial quaternion (no rotation)
    trajectory = []

    rate = rospy.Rate(10)  # 10 Hz
    step = 0

    while not rospy.is_shutdown():
        # Predict the next state
        next_x, next_y, next_orientation = predict_state(current_x, current_y, *current_orientation)
        current_x, current_y = next_x, next_y
        current_orientation = next_orientation
        trajectory.append((next_x, next_y, *next_orientation))

        # Create the Path message
        path = Path()
        path.header.frame_id = "map"  # Ensure this matches Rviz's fixed frame
        path.header.stamp = rospy.Time.now()

        for x, y, qx, qy, qz, qw in trajectory:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)

        # Publish the path
        path_pub.publish(path)
        rospy.loginfo(f"Published path with {len(trajectory)} poses.")

        step += 1
        rate.sleep()

if __name__ == "__main__":
    main()
