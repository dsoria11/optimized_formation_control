"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move at the same altitude in the X-Y
plane, into a square formation
"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from fontTools.merge.util import current_time
from scipy.stats import halfcauchy

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

np.random.seed(0)

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 4
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 10
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
SAFE_DISTANCE = 0.3
H_STEP = .15

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):

    #### Standalone function for formation control ####
    def formation_control_gradient(positions, target_distances, num_drones):
        gradients = np.zeros_like(positions)
        for i in range(num_drones):
            for j in range(num_drones):
                if i != j:
                    delta = positions[i] - positions[j]
                    dist = np.linalg.norm(delta)
                    # Repulsive force for collision avoidance
                    if dist < SAFE_DISTANCE:
                        gradients[i] -= 5.0 * (SAFE_DISTANCE - dist) * (delta / dist)

                    # Formation control force
                    delta = positions[i] - positions[j]
                    dist_sq = np.dot(delta, delta)
                    desired_sq = target_distances[i, j] ** 2
                    gradients[i] += (dist_sq - desired_sq) * delta
        return -gradients  # negative gradient: descent direction

####randomized initial position####
    INIT_XYZS = np.array([[np.random.uniform(-1,1),np.random.uniform(-1,1),0.5] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range (num_drones)])
    TARGET_POS = np.array([[i % 2, i // 2, 0.5] for i in range(num_drones)])
    TARGET_DISTANCE = np.zeros((DEFAULT_NUM_DRONES, DEFAULT_NUM_DRONES))

    for i in range(num_drones):
        for j in range(num_drones):
            TARGET_DISTANCE[i, j] = np.linalg.norm(TARGET_POS[i] - TARGET_POS[j])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()

    #### Lists to store magnitude of gradient values and time iterations for each drone
    time_pts = []
    drone_objective_values = [[] for _ in range(num_drones)]

    #### Lists to store objective function values and time iterations for each drone
    time_pts_error = []
    objective_error_values = []

    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        # #### Make it rain rubber ducks #############################
        # if i/env.CTRL_FREQ>5 and i%10==0 and i/env.CTRL_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Current time ####
        current_time = i / env.CTRL_FREQ

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        # Get 2D positions for gradient calculation
        pos_2d = obs[:, :2]

        # Calculate the combined gradient for all drones
        gradient = formation_control_gradient(pos_2d, TARGET_DISTANCE, num_drones)

        #### Objective Function Error for Formation ####
        total_error = 0.0
        collision_avoidance_term = 0.0
        count = 0
        for m in range(num_drones):
            for n in range(m+1, num_drones):
                actual_dist = np.linalg.norm(obs[m, :3] - obs[n, :3])
                desired_dist = TARGET_DISTANCE[m, n]
                total_error += abs(actual_dist**2 - desired_dist**2)**2
                if actual_dist < SAFE_DISTANCE:
                    collision_avoidance_term += abs(actual_dist**2 - SAFE_DISTANCE**2)**2
                    total_error += collision_avoidance_term
                count += 1
        avg_formation_error = total_error / count if count > 0 else 0

        #### Storing error and time for plotting
        objective_error_values.append(avg_formation_error)
        time_pts_error.append(current_time)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            #### The control "objective" for this drone is the magnitude of its gradient
            combined_gradient = gradient[j]
            gradient_magnitude = np.linalg.norm(combined_gradient)

            #### Store the gradient magnitude for the current drone
            drone_objective_values[j].append(gradient_magnitude)

            #### Convert 2D gradient to 3D for control, keeping z-direction at 0
            new_gradient = np.array([combined_gradient[0], combined_gradient[1], 0])
            new_target_pos = obs[j, :3] + H_STEP * new_gradient

            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                 state=obs[j],
                                                                 target_pos = new_target_pos,
                                                                 )

        # Storing time for plotting
        time_pts.append(current_time)

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plotting objective function (gradient magnitude) over time for each drone and overall mean #################
    plt.figure()

    #### Convert the list of lists to a NumPy array
    drone_objective_array = np.array(drone_objective_values)

    #### Calculating mean of gradients for all drones at each time step
    mean_trajectory = np.mean(drone_objective_array, axis=0)

    #### Plot a line for each drone
    for j in range(num_drones):
        plt.plot(time_pts, drone_objective_values[j], alpha=0.6, label=f"Drone {j} Gradient Magnitude")

    #### Overall mean trajectory
    plt.plot(time_pts, mean_trajectory, color='r', linestyle='--', linewidth=2, label="Overall Mean Trajectory")

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Control Gradient Magnitude')
    plt.title('Control Gradient Magnitude Over Time')
    plt.grid(True)

    return time_pts_error, objective_error_values

#-----------------------------------------------------------------------------------------------------#

def run_without_collision_term(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):

    np.random.seed(0)

    #### Refactoring 'formation_control_gradient' as standalone function ####
    def formation_control_gradient_without_collision(positions, target_distances, num_drones):
        gradients = np.zeros_like(positions)
        for i in range(num_drones):
            for j in range(num_drones):
                if i != j:
                    # Formation control force only
                    delta = positions[i] - positions[j]
                    dist_sq = np.dot(delta, delta)
                    desired_sq = target_distances[i, j] ** 2
                    gradients[i] += (dist_sq - desired_sq) * delta
        return -gradients  # negative gradient: descent direction

    ####randomized initial position####
    INIT_XYZS = np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.5] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])
    TARGET_POS = np.array([[i % 2, i // 2, 0.5] for i in range(num_drones)])
    TARGET_DISTANCE = np.zeros((DEFAULT_NUM_DRONES, DEFAULT_NUM_DRONES))

    for i in range(num_drones):
        for j in range(num_drones):
            TARGET_DISTANCE[i, j] = np.linalg.norm(TARGET_POS[i] - TARGET_POS[j])

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui
                     )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()

    #### Lists to store magnitude of gradient values and time iterations for each drone
    time_pts = []
    drone_objective_values = [[] for _ in range(num_drones)]

    #### Lists to store objective function values and time iterations for each drone
    time_pts_error = []
    objective_error_values = []

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        # #### Make it rain rubber ducks #############################
        # if i/env.CTRL_FREQ>5 and i%10==0 and i/env.CTRL_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Current time ####
        current_time = i / env.CTRL_FREQ

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        # Get 2D positions for gradient calculation
        pos_2d = obs[:, :2]

        # Calculate the combined gradient for all drones
        gradient = formation_control_gradient_without_collision(pos_2d, TARGET_DISTANCE, num_drones)

        #### Objective Function Error for Formation ####
        total_error = 0.0
        collision_avoidance_term = 0.0
        count = 0
        for m in range(num_drones):
            for n in range(m + 1, num_drones):
                actual_dist = np.linalg.norm(obs[m, :3] - obs[n, :3])
                desired_dist = TARGET_DISTANCE[m, n]
                total_error += abs(actual_dist**2 - desired_dist**2)**2
                if actual_dist < SAFE_DISTANCE:
                    total_error += collision_avoidance_term
                count += 1
        avg_formation_error = total_error / count if count > 0 else 0

        #### Storing error and time for plotting
        objective_error_values.append(avg_formation_error)
        time_pts_error.append(current_time)

        #### Compute control for the current way point #############
        for j in range(num_drones):

            #### The control "objective" for this drone is the magnitude of its gradient
            combined_gradient = gradient[j]
            gradient_magnitude = np.linalg.norm(combined_gradient)

            #### Store the gradient magnitude for the current drone
            drone_objective_values[j].append(gradient_magnitude)

            #### Convert 2D gradient to 3D for control, keeping z-direction at 0
            new_gradient = np.array([combined_gradient[0], combined_gradient[1], 0])
            new_target_pos = obs[j, :3] + H_STEP * new_gradient

            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                 state=obs[j],
                                                                 target_pos=new_target_pos,
                                                                 )
        # Storing time for plotting
        time_pts.append(current_time)

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    return time_pts_error, objective_error_values

# #### Save the simulation results ###########################
    # logger.save()
    # logger.save_as_csv("pid") # Optional CSV save
    #
    # #### Plot the simulation results ###########################
    # if plot:
    #     logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    # Run simulation with collision term
    time_with, error_with = run(**vars(ARGS))

    # Run simulation without collision term
    time_without, error_without = run_without_collision_term(**vars(ARGS))

    # Plotting both objective function errors on the same graph for comparison
    plt.figure()
    plt.plot(time_with, error_with, label="With Collision Term")
    plt.plot(time_without, error_without, label="Without Collision Term")

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Objective Function Error')
    plt.title('Objective Function Error Over Time Comparison')
    plt.grid(True)
    plt.show()
