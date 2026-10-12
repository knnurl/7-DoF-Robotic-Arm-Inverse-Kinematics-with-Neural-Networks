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

def initialize_population(population_size, constraints):
    population = []
    for _ in range(population_size):
        individual = [np.random.uniform(low, high) for low, high in zip(constraints[0], constraints[1])]
        population.append(individual)
    return np.array(population) 	

def fitness_function(solution, target_position):
    #n_particles = solution.shape[0]  # number of particles
    #n_particles = len(solution)  # number of particles

    # Calculate the forward kinematics of the solution(joint angles).
    current_position = get_end_tip_position(solution)
    #current_position = [get_end_tip_position(solution) for i in range(n_particles)]

    # Calculate the distance between the current and target positions.
    position_distance = np.linalg.norm(target_position - current_position)
    # The fitness could be the inverse of the error (since we want to minimize the error).
    fitness = 1 / (1 + position_distance)
    return fitness

def select_parents(population, fitness_scores, num_parents):
    parents = np.zeros((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness_scores == np.max(fitness_scores))
        parents[i, :] = population[max_fitness_idx[0][0], :]
        fitness_scores[max_fitness_idx[0][0]] = -99999999999  # Exclude this parent from being selected again
    return parents

def crossover(parents, offspring_size):
    offspring = np.zeros(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have half of its genes from each parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
    
def mutation(offspring_crossover, mutation_rate, joint_limits):
    for idx in range(offspring_crossover.shape[0]):
        for jdx in range(offspring_crossover.shape[1]):
            if np.random.rand() < mutation_rate:
                # Randomly change the jdx gene of the idx offspring.
                random_value = np.random.uniform(joint_limits[0][jdx], joint_limits[1][jdx])
                offspring_crossover[idx, jdx] = random_value
    return offspring_crossover
    
def optimize_ga():
    population = initialize_population(population_size, constraints)
    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness_function(individual, target_position) for individual in population])
        # Select parents
        parents = select_parents(population, fitness_scores, num_parents=population_size//2)
        # Crossover
        offspring_crossover = crossover(parents, offspring_size=(population_size - parents.shape[0], 7))
        # Mutation
        offspring_mutation = mutation(offspring_crossover, mutation_rate=0.1, joint_limits=constraints)
        # Create new population
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
        # Check for convergence or other termination criteria
        # (not shown here; could be based on fitness threshold, etc.)

    # Best solution
    best_idx = np.argmax(fitness_scores)
    best_solution = population[best_idx, :]
    return best_idx, best_solution
    

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
    population_size = 50
    num_generations = 1000

#    bounds = [(constraints[0][i], constraints[1][i]) for i in range(7)] 

    # Workspace range
    x, y, z = np.random.uniform(-0.5,0.5), np.random.uniform(-0.5,0.5), np.random.uniform(0.2,0.9)
    target_position = np.array([x,y,z])
#    target_position = np.array([-0.00232187, -0.25024734,  0.79152821])


    # Solving the IK problem
    print("Optimization started. ")
    print("Using genetic algorithm for joint angle prediction.\n ")

    # Calculate the joint configuration
    best_idx, best_solution = optimize_ga()
    
    print(f"\nBest Solution: {best_solution} @ {best_idx}\n")
    print(f"Target position: \n{target_position.round(4)}\n")
    print(f"Position with predicted angles:\n {get_end_tip_position(best_solution).round(4)}")

########################################

    # Move the joints of the robot arm to the desired angles in radians
    move_robot_arm(best_solution)
 
    print("\nRobotic arm has successfully reached the goal position!\n")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sys.stderr)
