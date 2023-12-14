#!/usr/bin/env python3

import subprocess
import concurrent.futures
import os

# Configuration
#assert(len(algo_values) == 1)
speed_multipliers = [0.5, 1.0, 3.0, 5.0]#,'min_lidar_ray_weight' 'steering_change_weig','velocity_change_weig', 'velocity_weight'

gap_blockers = [2, 5]

max_concurrent_processes = 2

# Create the logs directory if it doesn't exist
if not os.path.exists('logs_run_parallel'):
    os.makedirs('logs_run_parallel')

# Define a function to run a single job
def run_job(speed_multiplier, gap_blocker):
    num_timesteps = 50_000 # 100 rollouts
    command = (
        f"python collect_rollouts_ftg.py --speed_multiplier={speed_multiplier} --gap_blocker={gap_blocker} --norender --record --timesteps={num_timesteps}"
        f" --dataset=datasets1112"
    )
    
    # Modify log file path to include discount value and run number
    log_file_path = f"logs_parallel/output_algo_{speed_multiplier}_{gap_blocker}.log"
    
    with open(log_file_path, 'w') as log_file:
        subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)

# Using concurrent.futures to run multiple processes in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
    # Launch all the jobs
    futures = [
        executor.submit(run_job,speed_multiplier, gap_blocker)
        for speed_multiplier in speed_multipliers
        for gap_blocker in gap_blockers
    ]
    
    # Wait for all jobs to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Job raised an exception: {e}")
