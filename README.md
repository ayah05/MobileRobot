# Project 1: Learning the Kinematic Model of a Mobile Robot

# Setup Docker using FHTW Docker Container
## Initial Setup
- [Docker Container Link](https://github.com/TW-Robotics/Docker-ROS/tree/master)
- clone repository
- start docker container using steps as described in the repository README - below shorter explanation
- start XLaunch as seen in the FHTW Repository
- run `run_docker_from_hub.bat` - should download and setup everything necessary and then you can already get started 
- catkin_ws is already setup automatically
- everything inside the docker container inside `catkin_ws/src/fhtw` is mounted / can be found under `catkin_ws/src/` in your file explorer

- Install the required packages inside `catkin_ws/src`: 
```
apt update
apt install python3-catkin-tools
catkin_create_pkg project1 sensor_msgs nav_msgs
```
- Now, navigate to `/project1` and then create the following dirs:
```
mkdir src
mkdir launch
mkdir maps
mkdir param
```
- go back to `/catkin_ws` and run:
```
catkin init 
catkin_make
``` 

- Test the Setup:
```
rospack list
```
- Install Vim:
```
apt update
apt install vim -y
```
- Install Gazebo:
```
apt update
apt install gazebo11 -y
gazebo --version
```
## Download the Required Dependencies to Start the Mir-Robot 100 
- To download the required packages for the robot run:
```
sudo apt install ros-noetic-mir-robot
```
- Copy the contents of `full_simulation.launch` file from this repository to `catkin_ws/src/project1/launch` via vim or however you want to:
```
vim full_simulation.launch
```

--> paste the new contents into the file
--> now click (Esc+P) and then Esc again and then write `:wq` to save and quit the file

-  Copy the `position_logger.py` file from this repository to `catkin_ws/src/project1/src`
```
- Now we need to run inside of one terminal inside of the docker container the following cmd: 
```
roscore
```
- Inside of another terminal, also inside of the docker container, run:
```
roslaunch project1 full_simulation.launch 
```
- Now inside of the Gazebo Window (at the bottom), you need to start the simulation and then open the rViz Window to see where the robot is positioned and open the Robot Steering Window to move the robot.
```


# Setup Docker from Scratch
## Initial Setup

- First, pull the noetic image: `docker pull ros:noetic`
- Create a Network and a Volume:
```
docker network create mir_network
docker volume create mir_volume
```
- Start the docker container with the following command:
```
docker run -it \
  --name mirRobot \
  --net=ros_network \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume=ros_volume:/root/ros_workspace \
  ros:noetic
```
- Allow X11 Access on your host machine to use your display:
```
xhost +local:root
```
- Now access the docker container with the following cmd:
```
docker exec -it mirRobot bash
```
- Create a 'CATKINWS' dir in the root dir with 
```
mkdir CATKINWS
``` 
and then inside of that folder create a 'src' folder 
```
mkdir src
``` 
- Install the required packages inside CATKINWS/src: 
```
apt update
apt install python3-catkin-tools
catkin_create_pkg project1 sensor_msgs nav_msgs
```
- Now, navigate to `/project1` and then create the following dirs:
```
mkdir src
mkdir launch
mkdir maps
mkdir param
```
- now go to `/CATKINWS` and run:
```
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
- then run:
```
catkin init 
catkin_make
``` 
- Now, source the workspace:
```
source /CATKINWS/devel/setup.bash
echo "source /CATKINWS/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
- Test the Setup:
```
rospack list
```
- Install Vim:
```
apt update
apt install vim -y
```
- Install Gazebo:
```
apt update
apt install gazebo11 -y
gazebo --version
```
## Download the Required Dependencies to Start the Mir-Robot 100 
- To download the required packages for the robot run:
```
sudo apt install ros-noetic-mir-robot
```
- **Option 1**: the content of full_simulation.launch on the gazebo branch on github into a file by redirecting to `CATKINWS/src/project1/launch` and then running:
```
vim full_simulation.launch
```
--> now click (Esc+P) and then Esc again and then write `:wq` to save and quit the file

- **Option 2**: download the file from the github repo copy it with `docker cp` into the stated dir --> redirect to where the file is locally and then run:
```
docker cp full_simulation.launch mirRobot:/CATKINWS/src/project1/launch
```
- Now we need to run inside of one terminal inside of the docker container the following cmd: 
```
roscore
```
- Inside of another terminal, also inside of the docker container, run:
```
roslaunch project1 full_simulation.launch 
```
- Now inside of the Gazebo Window (at the bottom), you need to start the simulation and then open the rViz Window to see where the robot is positioned and open the Robot Steering Window to move the robot.

# Ingest Robot Motion Data and Generate Sequences for LSTM model
preprocessing.py offers a function **preprocess_data** that imports data from an input file and generates three output files that contain data sequences to be used by an LSTM model for training, validation, and test. The input file contains x, y, orientation_x, orientation_y, orientation_z and orientation_w.

The function takes these major parameters:  

```python
"""
input_file : str  
    path to the input CSV file containing raw motion data  
output_directory : str  
  directory where the preprocessed datasets will be saved
save_as_pt : bool, optional
    whether to save the datasets as .pt files for PyTorch, by default False - then the file is saved as CSV
sequence_length : int, optional
    number of timesteps to include in each input sequence, by default 10.
output_size : int, optional
    number of timesteps to predict for each output sequence, by default 1.
val_size : float, optional
    proportion of the data to use for validation, by default 0.1.
test_size : float, optional
    proportion of the data to use for testing, by default 0.2.
"""
```
You can save the generated sequence files either in PyTorch (.pt) or CSV format.

For this part of the code to run it is important to install torch using this command:
````bash
  pip install torch
