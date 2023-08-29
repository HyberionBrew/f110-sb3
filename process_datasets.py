
import zarr
import os
import argparse

# for now we do not! recompute the reward
parser = argparse.ArgumentParser(description='Your script description')


args = parser.parse_args()

def main(args):
    all_files = [f for f in os.listdir( FLAGS.input_folder) if os.path.isfile(os.path.join( args.input_folder, f))]
    print(f"Available files: {all_files}")
    

if __name__ == "__main__":
    main(args)