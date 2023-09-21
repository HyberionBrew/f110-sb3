import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
import shutil

def extract_timestamp(file_name):
    """Extract the timestamp from the filename."""
    match = re.search(r"(\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2})", file_name)
    return match.group(1) if match else None

def run_command(model_path, model_name):
    """Function to run the command."""
    cmd = f"python collect_rollouts.py --model_path={model_path} --norender --record --timesteps=50_000 --model_name={model_name}"
    subprocess.run(cmd, shell=True)

base_dir = "/home/fabian/f110_rl/f110-sb3/logs102"

# Traverse the directory to get all models
models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

tasks = []

output_dir = os.path.join(base_dir, "models")
os.makedirs(output_dir, exist_ok=True)

for model in models:
    checkpoint_dir = os.path.join(base_dir, model, "checkpoints")
    
    # Filter out checkpoints with 200000_steps
    checkpoints = [f for f in os.listdir(checkpoint_dir) if "200000_steps" in f]
    
    # If multiple files, select one with the latest timestamp
    if len(checkpoints) > 1:
        checkpoints.sort(key=extract_timestamp, reverse=True)
    
    if checkpoints:
        
        selected_checkpoint = checkpoints[0]
        print("Used checkpoint:", selected_checkpoint)
        model_path = os.path.join(checkpoint_dir, selected_checkpoint)
        model_name = model
        
        tasks.append((model_path, model_name))
    destination_path = os.path.join(output_dir, model_name + ".zip")
    shutil.copy2(model_path, destination_path)
# Use a process pool to run commands in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    for task in tasks:
        executor.submit(run_command, task[0], task[1])

print("All commands submitted.")