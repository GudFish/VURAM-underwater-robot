"""MuJoCo-trained VURAM policy deployment modules (ArduSub/MAVROS port).

Loads the exported TorchScript policy (deploy.pt), runs the 20 Hz closed
loop against the ArduSub stack (odom -> obs[21] + history -> a[5] -> /dof_cmd), and
visualizes the reference (green) vs actual (orange) trajectories in RViz.
Deploy only: all training code lives in ~/mujoco_auv_env.
"""
