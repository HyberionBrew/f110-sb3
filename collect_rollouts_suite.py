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
    cmd = f"python collect_rollouts.py --model_path={model_path} --norender --record --timesteps=100_000 --model_name={model_name}"
    subprocess.run(cmd, shell=True)

def extract_step(file_name):
    """Extract the step from the filename."""
    pattern = r"(\d+)_steps"
    match = re.search(pattern, file_name)
    #number_before_steps = int(match.group(1))
    return int(match.group(1)) if match else None
base_dir = "/home/fabian/msc/f110_dope/rollouts/f110-sb3/logs102"

# Traverse the directory to get all models
models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
#print(models)
tasks = []

output_dir = "/home/fabian/msc/f110_dope/rollouts/f110-sb3/models"
os.makedirs(output_dir, exist_ok=True)
#print(output_dir)
for model in models:
    checkpoint_dir = os.path.join(base_dir, model, "checkpoints")
    #print("-----")
    #print(checkpoint_dir)
    # Filter out checkpoints with 200000_steps
    checkpoints = [f for f in os.listdir(checkpoint_dir)] #if "299952_steps" in f]

    # If multiple files, select one with the latest timestamp
    if len(checkpoints) > 1:
        checkpoints.sort(key=extract_step, reverse=False)
        #print("Sorted checkpoints:", checkpoints)    
    #break
    if checkpoints:
        # now we extract 3 checkpoints one at the start one at the end and one in the middle
        checkpoints = [checkpoints[0], checkpoints[-1]]
        names = ["start", "end"]
        for selected_checkpoint, name in zip(checkpoints, names):
            #selected_checkpoint = checkpoints[0]
            #print("Used checkpoint:", selected_checkpoint)
            model_path = os.path.join(checkpoint_dir, selected_checkpoint)
            model_name = model + "_" + name
            
            tasks.append((model_path, model_name))
            destination_path = os.path.join(output_dir, model_name+ ".zip")
            shutil.copy2(model_path, destination_path)
    
print(tasks)
# Use a process pool to run commands in parallel
with ProcessPoolExecutor(max_workers=3) as executor:
    for task in tasks:
        executor.submit(run_command, task[0], task[1])

print("All commands submitted.")