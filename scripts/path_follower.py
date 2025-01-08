#!/usr/bin/env python
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib


class PathFollower:
    def __init__(self):
        rospy.init_node('path_follower')

        # Get the routes parameter
        self.routes = rospy.get_param('~routes', [])
        rospy.loginfo(f"Loaded routes parameter: {self.routes}")
        self.current_route_index = 0
        self.current_waypoint_index = 0

        if not self.routes:
            rospy.logerr("No routes provided! Shutting down.")
            rospy.signal_shutdown("No routes provided.")
            return

        rospy.loginfo(f"Loaded {len(self.routes)} routes.")

        # Initialize move_base client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")

        self.process_next_goal()

    def process_next_goal(self):
        if self.current_route_index >= len(self.routes):
            rospy.loginfo("All routes completed.")
            rospy.signal_shutdown("Finished all routes.")
            return

        current_route = self.routes[self.current_route_index]
        if self.current_waypoint_index >= len(current_route):
            self.current_route_index += 1
            self.current_waypoint_index = 0
            self.process_next_goal()
            return

        waypoint = current_route[self.current_waypoint_index]
        rospy.loginfo(f"Sending goal: {waypoint}")
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = waypoint[0]
        goal.target_pose.pose.position.y = waypoint[1]
        goal.target_pose.pose.orientation.w = 1.0

        # Cancel any active or pending goals to avoid conflicts
        if self.move_base_client.get_state() in [
            actionlib.GoalStatus.ACTIVE,
            actionlib.GoalStatus.PENDING,
        ]:
            rospy.logwarn("Cancelling active/pending goal before sending a new one.")
            self.move_base_client.cancel_all_goals()
            rospy.sleep(1)  # Give some time for the cancellation to take effect

        # Send the new goal
        self.move_base_client.send_goal(goal, done_cb=self.goal_done_cb)

    def goal_done_cb(self, status, result):
        """Callback for when a goal is completed."""
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal reached.")
            self.current_waypoint_index += 1
            rospy.sleep(1)  # Short delay before processing the next goal
            self.process_next_goal()
        elif status in [actionlib.GoalStatus.PREEMPTED, actionlib.GoalStatus.ABORTED]:
            rospy.logwarn(f"Failed to reach goal. Status: {status}. Retrying...")
            # Ensure all active/pending goals are cleared before retrying
            self.move_base_client.cancel_all_goals()
            rospy.sleep(1)
            self.process_next_goal()
        else:
            rospy.logwarn(f"Unexpected status: {status}. Retrying...")
            self.move_base_client.cancel_all_goals()
            rospy.sleep(1)
            self.process_next_goal()


if __name__ == "__main__":
    try:
        PathFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

