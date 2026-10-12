#!/usr/bin/env python3
"""
 Author: Kaan Ural
 move_robot_arm command code is taken from: Automatic Addison https://automaticaddison.com
 then heavily modified by me.
   
 Description: Support vector machine algorithm is used to find the joint angles to reach
		      the randomly selected point within the workspace range. Then the 
		      ROS action lib is used to move the robotic arm to the goal location.
"""
# Import the necessary libraries
from __future__ import print_function # Printing
import rospy # Python client library
import actionlib # ROS action library
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal # Controller messages
from std_msgs.msg import Float64 # 64-bit floating point numbers
from trajectory_msgs.msg import JointTrajectoryPoint # Robot trajectories

import pandas as pd
import numpy as np
import seaborn as sns
import numpy.typing as npt
import random 
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from math import sqrt
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline

from joblib import load
multi_output_svr_loaded = load('multi_output_svr.joblib')

df = pd.read_csv('gazebo_dataset_synced.csv')
y = df.iloc[:,4:]
X = df.iloc[:,1:4]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Scale the data
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)

def move_robot_arm(joint_values):
  """
  Function to move the robot arm to desired joint angles.
  :param: joint_values A list of desired angles for the joints of a robot arm 
  """
  # Create the SimpleActionClient, passing the type of the action to the constructor
  arm_client = actionlib.SimpleActionClient('/iiwa/PositionJointInterface_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
 
 
  # Wait for the server to start up and start listening for goals.
  arm_client.wait_for_server()
     
  # Create a new goal to send to the Action Server
  arm_goal = FollowJointTrajectoryGoal()
 
  # Store the names of each joint of the robot arm
  arm_goal.trajectory.joint_names = ["iiwa_joint_1","iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4", "iiwa_joint_5", "iiwa_joint_6","iiwa_joint_7"]
   
  # Create a trajectory point   
  point = JointTrajectoryPoint()
 
  # Store the desired joint values
  point.positions = joint_values
 
  # Set the time it should in seconds take to move the arm to the desired joint angles
  point.time_from_start = rospy.Duration(3)
 
  # Add the desired joint values to the goal
  arm_goal.trajectory.points.append(point)
 
  # Define timeout values
  exec_timeout = rospy.Duration(10)
  prmpt_timeout = rospy.Duration(5)
 
  # Send a goal to the ActionServer and wait for the server to finish performing the action
  arm_client.send_goal_and_wait(arm_goal, exec_timeout, prmpt_timeout)
 
 
if __name__ == '__main__':
  """
  Main method.
  """
  try:
    # Initialize a rospy node so that the SimpleActionClient can
    # publish and subscribe over ROS.
    rospy.init_node('send_goal_to_arm_py')


    x, y, z = np.random.uniform(-0.5,0.5), np.random.uniform(-0.5,0.5), np.random.uniform(0.2,0.9)
    target_position = np.array([x,y,z])
    
    # Use your model to predict joint angles from x, y, z coordinates
    x_new = x_scaler.transform([target_position])  # Normalize
    
    best_solution = multi_output_svr_loaded.predict(x_new)[0]

    #inv_angles = yscaler.inverse_transform(y_new)
    print('Predicted joint angles:\n', best_solution)    
   # print('inverse transformed predicted joint angles:\n', inv_angles[0])
    print(f"Target position: \n{target_position}\n")
#    print(f"Position with predicted angles:\n {get_end_tip_position(best_solution).round(4)}")


    # Move the joints of the robot arm to the desired angles in radians
    move_robot_arm(best_solution) 
    print("Robotic arm has successfully reached the goal!")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sys.stderr)
