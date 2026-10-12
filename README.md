# 7-DoF Robotic Arm Inverse Kinematics with Neural Networks

This project solves inverse kinematics (IK) for the KUKA LBR iiwa 7 R800, a 7-DoF arm, in ROS and Gazebo. Given a target end-effector position (x, y, z), it finds the seven joint angles that reach it in two ways:

- **Learned models.** A model trained on a dataset of poses predicts the joint angles directly. The models are a recurrent neural network and an SVR.
- **Optimisation.** A search runs over joint space and minimises the distance between the target and the forward-kinematics position. The methods are simulated annealing, differential evolution, a genetic algorithm, particle swarm optimisation and gradient descent.

In each case the solution goes to the simulated arm through the `FollowJointTrajectory` action, so the arm moves to the goal in Gazebo.

The method and results are written up in [the paper](docs/Paper%20-%20Enhancing%207DoF%20Inverse%20Kinematics.pdf). It also covers models that are not in the ROS scripts: feedforward networks, LSTM, random forest and decision tree.

## Demonstrations

| Video | Shows |
|---|---|
| [IK_RNN_Demonstration.mp4](mp4s/IK_RNN_Demonstration.mp4) | The pre-trained RNN predicting joint angles for random targets, with the arm moving in Gazebo |
| [OptimizationInterfaceDemonstration.mp4](mp4s/OptimizationInterfaceDemonstration.mp4) | The interactive interface for choosing an optimisation algorithm |

## Repository layout

```
mp4s/                     demonstration videos
docs/                     paper, and rostopic commands for moving the arms by hand
src/                      catkin workspace src/ (ROS Noetic)
  arm_to_goal/            IK scripts, the pre-trained RNN and the dataset
  iiwa_stack/             KUKA iiwa description, Gazebo, control and MoveIt packages (based on IFL-CAMP/iiwa_stack)
  mob_man_7/              mobile manipulator: iiwa 7 on a mobile base
  mobile_manipulator_body/  mobile base description
  notebooks/              notebook that turns Gazebo recordings into the dataset
  .rosinstall             upstream MoveIt and related packages, pinned to the commits used
```

### Scripts in `src/arm_to_goal/scripts/`

| Script | What it does |
|---|---|
| `ros_optimization_interface.py` | Menu-driven: picks a random target, solves it with the algorithm you choose (SA, DE, GA, PSO or gradient descent) and moves the arm. It can also send the arm home. |
| `send_goal_to_kuka_RNN.py` | Predicts joint angles with `rnn_model.h5` and moves the arm |
| `send_goal_to_kuka_SVR.py` | The same with support vector regression. It loads `multi_output_svr.joblib` from the working directory, and that 13 MB model file is not in this repository. |
| `send_goal_to_kuka_SA.py`, `send_goal_to_kuka_GA.py` | The same with simulated annealing or a genetic algorithm on its own |
| `send_goal_to_kuka_home.py` | Sends the arm to its zero position |

### Dataset

`src/arm_to_goal/scripts/gazebo_dataset_synced.csv` has about 96,000 rows. Each row gives an end-effector position (`x, y, z`) and the joint angles that produced it (`theta0`–`theta6`). The positions come from the forward kinematics of joint angles recorded in Gazebo, and they were checked against the simulation. [`src/notebooks/gazebo_extract_xyz&angles.ipynb`](src/notebooks/gazebo_extract_xyz&angles.ipynb) builds the dataset from the recordings.

The raw rosbag recordings (about 700 MB) are not included because they exceed GitHub's file size limit.

## Setup

The workspace was built on Ubuntu 20.04 with ROS Noetic and Gazebo 11.

```bash
mkdir -p ~/ik_ws && cd ~/ik_ws
git clone https://github.com/knnurl/7-DoF-Robotic-Arm-Inverse-Kinematics-with-Neural-Networks.git repo
ln -s repo/src src

# upstream packages (MoveIt, geometric_shapes, srdfdom, ...)
wstool init src src/.rosinstall    # or: vcs import src < src/.rosinstall

rosdep install --from-paths src --ignore-src -r -y
catkin build                       # or catkin_make
source devel/setup.bash

pip3 install numpy pandas scikit-learn tensorflow keras pyswarms matplotlib
```

## Running

Start the iiwa 7 in Gazebo with the position trajectory controller:

```bash
roslaunch iiwa_gazebo iiwa_gazebo.launch model:=iiwa7
```

In a second terminal, run a solver from the `arm_to_goal` package directory. The scripts load the model and dataset by relative path (`scripts/...`), so they must be started from that directory:

```bash
cd ~/ik_ws/src/arm_to_goal
python3 scripts/ros_optimization_interface.py   # choose an optimisation algorithm
python3 scripts/send_goal_to_kuka_RNN.py        # pre-trained RNN
```

The scripts send goals to `/iiwa/PositionJointInterface_trajectory_controller/follow_joint_trajectory`. To move the arm by hand with `rostopic pub`, see [docs/arm_control_commands.md](docs/arm_control_commands.md).

## Credits

- The `move_robot_arm` action-client code is adapted from [Automatic Addison](https://automaticaddison.com).
- `iiwa_stack` is based on [IFL-CAMP/iiwa_stack](https://github.com/IFL-CAMP/iiwa_stack) (BSD-2-Clause, see `src/iiwa_stack/LICENSE`).

Author: Kaan Ural, University of Salford.
