"""
Multiprocess rendering script to render images on multiple GPUs at a time.

Saves commands as scripts named "procs-#.txt".
"""

from collections import defaultdict
import argparse

PER_RUN_NUM = 50

def main(args):
    gpu_idx = 0
    commands = defaultdict(list)
    for start_idx in range(args.start, args.num_images, PER_RUN_NUM):
        gpu = args.gpus[gpu_idx]
        
        dir = f"/dfs/user/tailin/.results/CLEVR_relation/mpi-{args.start}-{args.num_images}"
        cmdline = f"env CUDA_VISIBLE_DEVICES={gpu} ./blender/blender --background --python render_images.py -- --start_idx {start_idx} --num_images {PER_RUN_NUM} --use_gpu 1 --min_objects 2 --max_objects 6 --output_image_dir {dir}/images/ --output_scene_dir {dir}/scenes/ --output_scene_file {dir}/CLEVR_scenes.json"

        print(cmdline)

        commands[gpu].append(cmdline + "\n")

        gpu_idx += 1
        gpu_idx %= len(args.gpus)

    for gpu, cmds in commands.items():
        with open(f"procs-{gpu}.txt", "w") as f:
            f.writelines(cmds)

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', nargs="+", required=True, help="CUDA device IDs to render on")
parser.add_argument('--start', type=int, required=True, help="Start index to render at")
parser.add_argument('--num-images', type=int, required=True, help="Total number of images to generate")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)