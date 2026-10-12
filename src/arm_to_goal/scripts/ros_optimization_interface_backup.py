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

##################################################################################
##################################################################################

		###########  SIMULATED ANNEALING  ################

##################################################################################
##################################################################################

initial_temperature = 0.6
n_iterations = 50000
step_size = 0.1

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
###################################################################################



##################################################################################
##################################################################################

		###########  GENETIC ALGORITHM  ################

##################################################################################
##################################################################################

# Algorithm parameters, problem bounds and target position
population_size = 80
num_generations = 1200

# Define the objective function that calculates the distance between
# current position and target position according to the joint angles
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

# Define the fundamental functions for genetic algorithm
def initialize_population(population_size, constraints):
    population = []
    for _ in range(population_size):
        individual = [np.random.uniform(low, high) for low, high in zip(constraints[0], constraints[1])]
        population.append(individual)
    return np.array(population)

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

# Main genetic algorithm function where results are obtained
def optimize_ga():
    population = initialize_population(population_size, constraints)
    best_history = [0]
    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness_function(individual, target_position) for individual in population])
        best_history.append(np.argmax(fitness_scores))
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
    
    return best_idx, best_solution, best_history
    

###################################################################################
###################################################################################
###################################################################################



##################################################################################
##################################################################################

		###########  DIFFERENTIAL EVOLUTION  ################

##################################################################################
##################################################################################
    
pop_size = 100
max_gen = 600
bounds = np.array([(-lim, lim) for lim in [lim1, lim2, lim3, lim4, lim5, lim6, lim7]])

# Define the objective function that calculates the distance between
# current position and target position according to the joint angles

def objective_function(joint_angles, target_position):
    # Compute the current position of the end-effector based on joint angles
    current_position = get_end_tip_position(joint_angles)  # Placeholder value
    # Calculate the Euclidean distance to the target position
    distance = np.linalg.norm(current_position - target_position)
    return distance

# Define the differential evolution algorithm
def differential_evolution(objective, target_position, bounds, constraints, pop_size, max_gen, mutation_factor=0.8, crossover_prob=0.7):
    # Initialize population within bounds
    pop = np.random.rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    pop = np.clip(pop, constraints[0], constraints[1])  # Ensure initial population respects joint limits
    
    best_score = objective(pop[0], target_position)  # Assuming target_position is correctly defined
    best_sol = pop[0]
    score_history = [best_score]  # Initialize history tracking

    for gen in range(max_gen):
        for i in range(pop_size):
            # Mutation
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mutation_factor * (b - c), bounds[:, 0], bounds[:, 1])
            mutant = np.clip(mutant, constraints[0], constraints[1])  # Ensure mutant respects joint limits
            
            # Crossover
            cross_points = np.random.rand(len(bounds)) < crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, len(bounds))] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            # Selection
            trial_score = objective(trial, target_position)
            if trial_score < best_score:
                best_score, best_sol = trial_score, trial.copy()
                score_history.append(best_score)  # Update history

            if trial_score < objective(pop[i], target_position):
                pop[i] = trial.copy()

        if gen==5:
            print("...")
        # At first and last 5 iterations, print the gen score
        if max_gen - gen == 3:
            print("\nLast 5 iterations:\n")
        elif max_gen - gen < 3 or gen < 3:
            print(f"Gen: {gen}, Best Score: {best_score}")
     
        # Stop searcing further if score reaches below 0.02
        # as it is an almost exact solution
        if best_score < 0.02:
            print(f"\nGen: {gen}, Best Score: {best_score:.4f}")
            break

    return best_sol, best_score, score_history

###################################################################################
###################################################################################
###################################################################################



##################################################################################
##################################################################################

		###########  PARTICLE SWARM OPTIMIZATION  ################

##################################################################################
##################################################################################

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
#%load_ext autoreload
#%autoreload 2

swarm_size = 50
dim = 7        # Dimension of X
epsilon = 1.0
options = {'c1': 1.5, 'c2':1.5, 'w':0.5}

def distance(query, target):
    x_dist = (target[0] - query[0])**2
    y_dist = (target[1] - query[1])**2
    z_dist = (target[2] - query[2])**2
    dist = np.sqrt(x_dist + y_dist + z_dist)
    return dist

def opt_func(X):
    n_particles = X.shape[0]  # number of particles
    dist = [distance(get_end_tip_position(X[i]), target_position) for i in range(n_particles)]
    return np.array(dist)

# Call an instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                    dimensions=dim,
                                    options=options,
                                    bounds=constraints)


###################################################################################
###################################################################################
###################################################################################



##################################################################################
##################################################################################

		###########  GRADIENT DESCENT  ################

##################################################################################
##################################################################################

initial_joint_angles = [0.,0.,0.,0.,0.,0.,0.]

def cost_function(joint_angles, target_position, forward_kinematics):
    """
    Calculate the cost based on the current and target positions.
    """
    current_position = forward_kinematics(joint_angles)
    return np.linalg.norm(current_position - target_position)

def gradient_descent(target_position, initial_joint_angles, forward_kinematics, iterations=100, learning_rate=0.01):
    """
    Perform inverse kinematics by optimizing the joint angles to minimize the distance
    to the target position using gradient descent.
    """
    joint_angles = np.array(initial_joint_angles)
    cost_history = []  # Initialize an empty list to store the cost history
        
    for _ in range(iterations):
        # Calculate the gradient of the cost function
        gradient = np.zeros(joint_angles.shape)
        
        # Calculate the current cost
        current_cost = cost_function(joint_angles, target_position, forward_kinematics)
        cost_history.append(current_cost)  # Append the current cost to the history
    
        for i in range(len(joint_angles)):
            # Perturbation method for gradient approximation
            original_angle = joint_angles[i]
            joint_angles[i] += 0.01
            cost_plus = cost_function(joint_angles, target_position, forward_kinematics)
            joint_angles[i] = original_angle - 0.01
            cost_minus = cost_function(joint_angles, target_position, forward_kinematics)
            joint_angles[i] = original_angle
            
            # Approximate derivative
            gradient[i] = (cost_plus - cost_minus) / 0.02
        
        # Update the joint angles based on the gradient
        joint_angles -= learning_rate * gradient

        # Optional: Implement a convergence check

    return joint_angles, cost_history










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
 
 
def get_correct_int(prompt):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if value < 0 or value > 6:
            print("Sorry, choose from the given.")
            continue
        else:
            break
    return value

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
#    bounds = [(constraints[0][i], constraints[1][i]) for i in range(7)] 
    x, y, z = np.random.uniform(-0.5,0.5), np.random.uniform(-0.5,0.5), np.random.uniform(0.1,0.9)
    target_position = np.array([x,y,z])
#    target_position = np.array([-0.00232187, -0.25024734,  0.79152821])

    print('Random target position has been selected.')
    print('Which optimization algorithm to use?\n')
    print('1- Simulated Annealing')
    print('2- Differential Evolution')
    print('3- Genetic Algorithm')
    print('4- Particle Swarm Optimization')
    print('5- Gradient Descent')
    print('       or ')
    print('6- Go home position\n')
    
#    x = input()
    x = get_correct_int(" ")

    # Solving the IK problem
    if x==1:
        print("\n\nChosen Simulated Annealing. ")
        print("Optimization started. ")
        best_solution, best_cost, cost_history, temperature_history = simulated_annealing_for_ik(target_position, bounds, initial_temperature, n_iterations, step_size)
        print("Optimization started. ")
        print(f"Cost: {best_cost}\n")
    elif x==2:
        print("\n\nChosen Differential Evolution. ")
        print("Optimization started. ")
        best_solution, best_score, score_history = differential_evolution(objective_function, target_position, bounds, constraints, pop_size, max_gen)
        
    elif x==3:
        print("\n\nChosen Genetic Algorithm. ")
        print("Optimization started. ")
        best_idx, best_solution, fitness_score_history = optimize_ga()
        
    elif x==4:
        print("\n\nChosen Particle Swarm Optimization. ")
        print("Optimization started. ")
        cost, best_solution = optimizer.optimize(opt_func, iters=200)
        print(f"Cost: {cost}\n")
        
    elif x==5:
        print("\n\nChosen Gradient Descent. ")
        print("Optimization started. ")
        best_solution, cost_history = gradient_descent(target_position, initial_joint_angles, get_end_tip_position, iterations=1000, learning_rate=0.01)
       
    elif x==6:
        print("\n\nMoving to home position.")
        best_solution = [0,0,0,0,0,0,0]

    else:
        print("Incorrect input.")
    
    if x != 6:
        initial_joint_angles = best_solution
        print(f"\nBest Solution: {best_solution}\n")
        print(f"Target position: \n{target_position}\n")
        print(f"Position with predicted angles:\n {get_end_tip_position(best_solution).round(4)}")

########################################

    # Move the joints of the robot arm to the desired angles in radians
    move_robot_arm(best_solution)
 
    print("\nRobotic arm has successfully reached the goal position!\n")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sys.stderr)
