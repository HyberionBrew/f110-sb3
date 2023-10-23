import subprocess
import os

# Define a list of arguments
args = [
    "progress_weight",
    "raceline_delta_weight",
    "velocity_weight",
    "steering_change_weight",
    "velocity_change_weight",
    "min_action_weight",
    "min_lidar_ray_weight",
]

# Ensure 'logs' directory exists
if not os.path.exists('logs102'):
    os.mkdir('logs102')

# Create directories for each argument under 'logs/'
for arg in args:
    directory_path = f"logs102/{arg}"
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

# Iterate over each argument
for arg in args:
    # Create a dictionary to store the default values (0.0) for all arguments
    weights = {a: 0.0 for a in args}
    
    # Set the current argument to 1.0
    weights[arg] = 1.0
    
    # Construct the command
    command = ["python", "train.py","--num_processes","4","--logdir", f"logs102/{arg}", "--total_timesteps", "300_000"]
    for name, value in weights.items():
        command.extend([f"--{name}", str(value)])
    
    # Execute the command
    subprocess.run(command)