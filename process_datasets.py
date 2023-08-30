
import zarr
import os
import argparse
import pickle as pkl
import numpy as np
# for now we do not! recompute the reward
parser = argparse.ArgumentParser(description='Your script description')

parser.add_argument('--input_folder', type=str, default='datasets', help='Logging directory')
parser.add_argument('--file', type=str, default=None, help='Specific File to process')
parser.add_argument('--noappend', action='store_false', 
                    default=True,dest='append', help='Whether to append to existing file')
args = parser.parse_args()

def main(args):
    all_files = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join( args.input_folder, f))]

    if args.file is not None:
        all_files = [args.file]
    print(f"Available files: {all_files}")
    #exit()
    #root = zarr.open('trajectories.zarr', mode='w')

    chunks_size = 1000 
    # Create an extendible array named "actions"
    if not(args.append):
        # write new
        root = zarr.open('trajectories.zarr', mode='w')
        actions_array = root.zeros("actions", shape=(0, 2), chunks=(chunks_size, 2), dtype='float32',
                                    overwrite=False, maxshape=(None, 2))
        rewards_array = root.zeros("rewards", shape=(0, 1), chunks=(chunks_size, 1), dtype='float32',
                                    overwrite=False, maxshape=(None, 1))
        done_array = root.zeros("done", shape=(0, 1), chunks=(chunks_size, 1), dtype='bool',
                                    overwrite=False, maxshape=(None, 1))
        truncated_array = root.zeros("truncated", shape=(0, 1), chunks=(chunks_size, 1), dtype='bool',
                                    overwrite=False, maxshape=(None, 1))
        timesteps_array = root.zeros("timesteps", shape=(0, 1), chunks=(chunks_size, 1), dtype='int32',
                                    overwrite=False, maxshape=(None, 1))
        model_name_array = root.zeros("model_name", shape=(0, 1), chunks=(chunks_size, 1), dtype='str',
                                    overwrite=False, maxshape=(None, 1))
        obs_group = root.create_group('observations')
    else:
        store = zarr.DirectoryStore('trajectories.zarr')
        root = zarr.group(store=store)
        actions_array = root['actions']
        rewards_array = root['rewards']
        done_array = root['done']
        truncated_array = root['truncated']
        timesteps_array = root['timesteps']
        model_name_array = root['model_name']
        obs_group = root['observations']
    # Create an extendible array named "observations"

    
    inital = True
    for file in all_files:
        print(f"Processing {file}")
        try:
            with open(os.path.join(args.input_folder, file), 'rb') as f:
                while True:
                    action, obs, reward, done, truncated, info, timesteps, model_name = pkl.load(f)
                    # append actions to actions group
                    actions_array.append(action)
                    # initalize obs group
                    if inital and not(args.append):
                        inital=False
                        for key, value in obs.items():
                            max_shape = tuple([None] + list(np.array(value).shape[1:]))
                            chunk_shape = tuple([chunks_size] + list(np.array(value).shape[1:]))
                            obs_group.zeros(key, shape=tuple([0] + list(np.array(value).shape[1:])), dtype=np.array(value).dtype,
                                            maxshape=max_shape, chunks=chunk_shape, overwrite=False)
                    elif inital and args.append:
                        inital=False
                        for key, value in obs.items():
                            if key in obs_group:  # Check if the dataset exists
                                # No need to create the dataset. Just open and resize it.
                                obs_array = obs_group[key]
                            else:
                                # assertion error
                                raise AssertionError(f"Dataset {key} does not exist")
                    # append values
                    for key, value in obs.items():
                        obs_array = obs_group[key]
                        obs_array.append(np.array(value))
                    # append rewards
                    #print(reward)
                    rewards_array.append([[reward]])
                    # append done
                    done_array.append([[done]])
                    truncated_array.append([[truncated]])
                    timesteps_array.append([[timesteps]])
                    model_name_array.append([[model_name]])

                    #print(obs)
                    # print(action)
                    if done or truncated:
                        print(done)
                        print(truncated)
                        print("-------")

        except EOFError:
            print("Done")
            
if __name__ == "__main__":
    main(args)