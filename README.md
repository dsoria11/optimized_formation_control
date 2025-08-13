# Drone Multi-Control Algorithm for Formation Control
This project focuses on developing and optimizing formation control algorithms for multi-UAV systems. Using distance-based formation and gradient descent optimization, our approach enables drones to maintain safe distances, avoid collisions, and achieve precise formations. 

## Dependencies 
This project tests its simulations on: `PyBullet` and uses `Python 3.10+`

For more information, please visit it's respective repository.
- [PyBullet](https://github.com/bulletphysics/bullet3)

## Virtual Environment Setup
1. Install necessary packages:
   ```bash
   pip install pybullet
   pip intall numpy
   pip install matplotlib
   ```
2. Important Notice
   
   If using Windows 11, you may need to install Microsoft's Visual Studio and disable Windows 11 SDK from the `Installation Details` section to get the simulator to work.

## How to Use the Code
1. Using a python terminal
   ```bash
   python pid_square_form.py
   ```

2. Output
   1. Initializes the PyBullet simulation environment and sets up drones at random starting positions.
   2. Calculates desired inter-drone distances based on a target square formation.
   3. Runs the formation control algorithm using gradient descent to minimize formation error.
   4. Applies a repulsive force for collision avoidance when drones are closer than the safe distance.
   5. Updates drone positions in real time using PID controllers to track gradient-based target points.
   6. Displays a live 3D simulation in the PyBullet viewer showing drones moving into formation.
   7. Logs and stores:
    - Gradient magnitude over time for each drone
    - Mean gradient magnitude across all drones
    - Objective function error over time
    - Collision avoidance term values
   8. Generates Matplotlib plots comparing:
    - Objective function error with and without collision avoidance
    - Collision avoidance term behavior during the simulation
