"""MuJoCo-trained VURAM policy deployment modules (ArduSub/MAVROS port).

Loads the exported TorchScript policy (policy.pt), runs the 20 Hz closed
loop against the ArduSub stack (odom -> obs -> a[6] -> /thruster_cmd), and
visualizes the reference (green) vs actual (orange) trajectories in RViz.
Deploy only: all training code lives in ~/mujoco_auv_env.
"""
