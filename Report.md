# Project 1: Learning the Kinematic Model of a Mobile Robot

For this task the MIR 100 Robot was loaded into a Gazebo World, with several nodes to enable autonomous exploration of the robot.


### **Position Logging**

The **Position Logger** is a custom ROS node (`position_logger.py`) that logs odometry data from the `/odom` topic to a CSV file. This node is critical for collecting motion data, which serves as the foundation for training the kinematic model.

#### **Features of the Position Logger**
1. **Odometry Data Collection**
   - Subscribes to the `/odom` topic, which provides `nav_msgs/Odometry` messages containing:
     - **Position**: The robot's position in 3D space (`x, y, z`).
     - **Orientation**: The robot's rotation represented as a quaternion (`x, y, z, w`).
     - **Linear Velocity**: The robot's velocity in 3D space (`x, y, z`).
     - **Angular Velocity**: The robot's rotational velocity (`x, y, z`).

2. **Data Logging**
   - Logs the above data into a CSV file with the following fields:
     - `timestamp`
     - `x, y, z` (position)
     - `orientation_x, orientation_y, orientation_z, orientation_w` (quaternion orientation)
     - `linear_velocity_x, linear_velocity_y, linear_velocity_z` (linear velocities)
     - `angular_velocity_x, angular_velocity_y, angular_velocity_z` (angular velocities)

3. **Dynamic Configuration**
   - Accepts parameters via the launch file:
     - `robot_name`: Specifies the robot being monitored (default: `mir_100`).
     - `output_file`: Specifies the path to save the CSV log (default: `/tmp/position_log.csv`).
   - If the output directory does not exist, it is created automatically.

4. **Rounding**
   - Rounds all numerical values to three decimal places for cleaner and more compact logs.
