#!/usr/bin/env python3
import rospy
import torch
import csv
import os
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from collections import deque
from lstm_train import KinematicsLSTM


class LSTMPredictor:
    def __init__(self):
        rospy.init_node('lstm_predictor')
        self.model = KinematicsLSTM(input_size=6, hidden_size=128, num_layers=3, output_size=6)

        model_file = rospy.get_param("~model_file", "best_model.pt")
        if not os.path.isfile(model_file):
            rospy.logerr(f"Model file not found: {model_file}")
            rospy.signal_shutdown("Model file missing")

        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

        self.robot_name = rospy.get_param("~robot_name", "mir_100")
        self.state_buffer = deque(maxlen=10)
        self.pub = rospy.Publisher('/predicted_pose', PoseStamped, queue_size=10)

        self.output_file = rospy.get_param("~output_file", "/tmp/prediction_log.csv")
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.csv_file = open(self.output_file, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp',
            'actual_x', 'actual_y', 'actual_qx', 'actual_qy', 'actual_qz', 'actual_qw',
            'predicted_x', 'predicted_y', 'predicted_qx', 'predicted_qy', 'predicted_qz', 'predicted_qw'
        ])

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.state_callback)
        rospy.loginfo(f"Predicting and logging data for {self.robot_name}")

    def state_callback(self, msg):
        try:
            index = msg.name.index(self.robot_name)
            pose = msg.pose[index]
            current_time = rospy.Time.now().to_sec()

            # Only process at 20Hz
            epsilon = 1e-9
            if abs((current_time % 0.05)) > epsilon:
                return

            current_state = [
                pose.position.x,
                pose.position.y,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ]

            self.state_buffer.append(current_state)

            if len(self.state_buffer) == 10:
                sequence = torch.tensor(list(self.state_buffer)).float().unsqueeze(0)

                with torch.no_grad():
                    prediction, _ = self.model(sequence)
                    prediction = prediction.squeeze()

                # Log data
                self.csv_writer.writerow([
                    round(current_time, 3),
                    round(pose.position.x, 3),
                    round(pose.position.y, 3),
                    round(pose.orientation.x, 3),
                    round(pose.orientation.y, 3),
                    round(pose.orientation.z, 3),
                    round(pose.orientation.w, 3),
                    round(prediction[0].item(), 3),
                    round(prediction[1].item(), 3),
                    round(prediction[2].item(), 3),
                    round(prediction[3].item(), 3),
                    round(prediction[4].item(), 3),
                    round(prediction[5].item(), 3)
                ])

                # Publish prediction
                msg = PoseStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "map"
                msg.pose.position.x = prediction[0].item()
                msg.pose.position.y = prediction[1].item()
                msg.pose.orientation.x = prediction[2].item()
                msg.pose.orientation.y = prediction[3].item()
                msg.pose.orientation.z = prediction[4].item()
                msg.pose.orientation.w = prediction[5].item()

                self.pub.publish(msg)

        except ValueError:
            rospy.logwarn(f"Robot {self.robot_name} not found in ModelStates!")

    def run(self):
        try:
            rospy.spin()
        finally:
            self.csv_file.close()


if __name__ == '__main__':
    predictor = LSTMPredictor()
    try:
        predictor.run()
    except rospy.ROSInterruptException:
        predictor.csv_file.close()