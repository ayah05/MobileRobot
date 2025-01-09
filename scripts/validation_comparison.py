#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math

# Global variable for ground truth path
ground_truth_path = []

def predict_state(current_x, current_y, current_qx, current_qy, current_qz, current_qw):
    # Define simple forward motion
    delta_x = 0.1
    delta_y = 0.0
    next_x = current_x + delta_x
    next_y = current_y + delta_y
    return next_x, next_y, (current_qx, current_qy, current_qz, current_qw)

def ground_truth_callback(msg):
    global ground_truth_path
    ground_truth_path = [(pose.pose.position.x, pose.pose.position.y,
                          pose.pose.orientation.x, pose.pose.orientation.y,
                          pose.pose.orientation.z, pose.pose.orientation.w)
                         for pose in msg.poses]

def calculate_errors(predicted_path, ground_truth_path):
    position_errors = []
    orientation_errors = []

    for i, (pred_x, pred_y, pred_qx, pred_qy, pred_qz, pred_qw) in enumerate(predicted_path):
        if i >= len(ground_truth_path):
            break

        gt_x, gt_y, gt_qx, gt_qy, gt_qz, gt_qw = ground_truth_path[i]

        pos_error = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        position_errors.append(pos_error)

        dot_product = pred_qx * gt_qx + pred_qy * gt_qy + pred_qz * gt_qz + pred_qw * gt_qw
        orientation_error = 2 * math.acos(min(max(dot_product, -1.0), 1.0))
        orientation_errors.append(orientation_error)

    return position_errors, orientation_errors

def main():
    rospy.init_node("model_quaternion_visualizer")

    path_pub = rospy.Publisher("/my_model_path", Path, queue_size=10)
    gt_path_pub = rospy.Publisher("/ground_truth_path_visualization", Path, queue_size=10)

    rospy.Subscriber("/ground_truth_path", Path, ground_truth_callback)

    current_x, current_y = 0, 0
    current_orientation = (0.0, 0.0, 0.0, 1.0)
    trajectory = []

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        next_x, next_y, next_orientation = predict_state(current_x, current_y, *current_orientation)
        current_x, current_y = next_x, next_y
        current_orientation = next_orientation
        trajectory.append((next_x, next_y, *next_orientation))

        path = Path()
        path.header.frame_id = "map"
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

        path_pub.publish(path)

        if ground_truth_path:
            gt_path = Path()
            gt_path.header.frame_id = "map"
            gt_path.header.stamp = rospy.Time.now()

            for x, y, qx, qy, qz, qw in ground_truth_path:
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw
                gt_path.poses.append(pose)

            gt_path_pub.publish(gt_path)

            position_errors, orientation_errors = calculate_errors(trajectory, ground_truth_path)
            rospy.loginfo(f"Position Error (mean, max): {sum(position_errors)/len(position_errors):.4f}, {max(position_errors):.4f}")
            rospy.loginfo(f"Orientation Error (mean, max): {sum(orientation_errors)/len(orientation_errors):.4f}, {max(orientation_errors):.4f}")

        rate.sleep()

if __name__ == "__main__":
    main()

