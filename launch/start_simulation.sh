#!/bin/bash

# Start Gazebo and RViz
roslaunch project1 gazebo_rviz.launch &
gazebo_pid=$!

# Wait for Gazebo and RViz to load
echo "Waiting for Gazebo and RViz to fully load..."
sleep 20 

# Start Nodes
roslaunch project1 nodes.launch &
nodes_pid=$!

# Wait for processes to complete
wait $gazebo_pid
wait $nodes_pid
