#!/usr/bin/env python3
 
# Author: Automatic Addison https://automaticaddison.com
# Description: Uses the ROS action lib to move a robotic arm to a goal location
 
# Import the necessary libraries
from __future__ import print_function # Printing
import rospy # Python client library
import actionlib # ROS action library
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal # Controller messages
from std_msgs.msg import Float64 # 64-bit floating point numbers
from trajectory_msgs.msg import JointTrajectoryPoint # Robot trajectories


###################################################################################
###################################################################################

# Import modules
import numpy as np
import math
import random
 
lim1 = np.deg2rad(170)
lim2 = np.deg2rad(120)
lim3 = np.deg2rad(170)
lim4 = np.deg2rad(120)
lim5 = np.deg2rad(170)
lim6 = np.deg2rad(120)
lim7 = np.deg2rad(175)

constraints = (np.array([-lim1 , -lim2 , -lim3 , -lim4 , -lim5 , -lim6, -lim7]),
               np.array([ lim1  , lim2 ,  lim3 ,  lim4 ,  lim5 ,  lim6,  lim7]))

d2 = d4 = d4 = d6 = 0.00
d1 = 0.34
d3 = 0.4
d5 = 0.4
d7 = 0.126

def getTransformMatrix(theta, d, a, alpha):
    T = np.array([[np.cos(theta) , -np.sin(theta)*np.cos(alpha) ,  np.sin(theta)*np.sin(alpha) , a*np.cos(theta)],
                  [np.sin(theta) ,  np.cos(theta)*np.cos(alpha) , -np.cos(theta)*np.sin(alpha) , a*np.sin(theta)],
                  [0             ,  np.sin(alpha)               ,  np.cos(alpha)               , d              ],
                  [0             ,  0                           ,  0                           , 1              ]
                 ])
    return T

def get_end_tip_position(params):
    # Create the transformation matrices for the respective joints
    t_00 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    t_01 = getTransformMatrix(params[0] , d1  , 0 , -np.pi/2)
    t_12 = getTransformMatrix(params[1] , d2  , 0 ,  np.pi/2)
    t_23 = getTransformMatrix(params[2] , d3  , 0 ,  np.pi/2)
    t_34 = getTransformMatrix(params[3] , d4  , 0 , -np.pi/2)
    t_45 = getTransformMatrix(params[4] , d5  , 0 , -np.pi/2)
    t_56 = getTransformMatrix(params[5] , d6  , 0 ,  np.pi/2)
    t_67 = getTransformMatrix(params[6] , d7  , 0 ,        0)

    # Get the overall transformation matrix
    end_tip_m = t_00.dot(t_01).dot(t_12).dot(t_23).dot(t_34).dot(t_45).dot(t_56).dot(t_67)

    # The coordinates of the end tip are the 3 upper entries in the 4th column
    pos = np.array([end_tip_m[0,3],end_tip_m[1,3],end_tip_m[2,3]])
    return pos


def objective_function(joint_angles, target_position):
    # Compute the current position of the end-effector based on joint angles
    current_position = get_end_tip_position(joint_angles)  # Placeholder value
    # Calculate the Euclidean distance to the target position
    distance = np.linalg.norm(current_position - target_position)
    return distance

def simulated_annealing_for_ik(target_position, bounds, initial_temperature, n_iterations, step_size):
    num_joints = len(bounds)
    current_solution = np.array([random.uniform(low, high) for low, high in bounds])
    current_cost = objective_function(current_solution, target_position)
    best_solution, best_cost = current_solution.copy(), current_cost

    # Initialize history tracking
    cost_history = []
    temperature_history = []
    
    for i in range(n_iterations):
        fraction = i / float(n_iterations)
        T = max(0.01, min(1, 1 - fraction))  # Cooling schedule
        
        # Generate a neighbor solution (ensure it respects joint limits)
        candidate_solution = current_solution + np.random.uniform(-step_size, step_size, num_joints)
        candidate_solution = np.clip(candidate_solution, constraints[0], constraints[1])
        candidate_cost = objective_function(candidate_solution, target_position)
        
        if candidate_cost < best_cost or math.exp((current_cost - candidate_cost) / T) > random.random():
            current_solution, current_cost = candidate_solution, candidate_cost
            if candidate_cost < best_cost:
                best_solution, best_cost = candidate_solution.copy(), candidate_cost
                
        if n_iterations - i == 10:
            print("Last 10 iterations:\n")
        elif n_iterations - i < 10:
            print(f"Iteration {i}: T={T:.3f}, Current Cost={current_cost:.3f}, Best Cost={best_cost:.3f}")
     
        # Update history
        cost_history.append(best_cost)
        temperature_history.append(T)
        
        if candidate_cost < 0.02:
            print(f"Iteration {i}: T={T:.3f}, Current Cost={current_cost:.3f}, Best Cost={best_cost:.3f}")
            break
    return best_solution, best_cost, cost_history, temperature_history


###################################################################################
###################################################################################

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
 #######################################################
     # Parameters for the IK problem
    # Target position for the end-effector
    #target_position = np.array([-0.3, 0.3, 0.8])
    bounds = [(constraints[0][i], constraints[1][i]) for i in range(7)] 
    x, y, z = np.random.uniform(-0.5,0.5), np.random.uniform(-0.5,0.5), np.random.uniform(0.1,0.9)
#    target_position = np.array([x,y,z])
    target_position = np.array([-0.00232187, -0.25024734,  0.79152821])


    initial_temperature = 0.6
    n_iterations = 50000
    step_size = 0.1

    # Solving the IK problem
    print("Optimization started. ")
    print("Using simulated annealing for joint angle prediction.\n ")
    best_solution, best_cost, cost_history, temperature_history = simulated_annealing_for_ik(target_position, bounds, initial_temperature, n_iterations, step_size)

    print(f"\nBest Solution: {best_solution}, Cost: {best_cost}\n")
    print(f"Target position: \n{target_position}\n")
    print(f"Position with predicted angles:\n {get_end_tip_position(best_solution).round(4)}")

########################################

    # Move the joints of the robot arm to the desired angles in radians
    move_robot_arm(best_solution)
 
    print("\nRobotic arm has successfully reached the goal position!\n")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sys.stderr)
