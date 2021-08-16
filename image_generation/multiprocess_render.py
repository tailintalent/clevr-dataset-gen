"""
Multiprocess rendering script to render images on multiple GPUs at a time.

Saves commands as scripts named "procs-#.txt".
"""

from collections import defaultdict

NUM_IMAGES = 2000
NUM_GPUS = 4
PER_RUN_NUM = 50

gpu = 0
commands = defaultdict(list)
for start_idx in range(0, NUM_IMAGES, PER_RUN_NUM):
    dir = f"/dfs/user/tailin/.results/CLEVR_relation/mpi/gpu{gpu}-{start_idx}-{start_idx + PER_RUN_NUM}"
    cmdline = f"env CUDA_VISIBLE_DEVICES={gpu} ./blender/blender --background --python render_images.py -- --start_idx {start_idx} --num_images {PER_RUN_NUM} --use_gpu 1 --min_objects 2 --max_objects 6 --output_image_dir {dir}/images/ --output_scene_dir {dir}/scenes/ --output_scene_file {dir}/CLEVR_scenes.json"

    print(cmdline)
    
    commands[gpu].append(cmdline + "\n")

    gpu += 1
    gpu %= NUM_GPUS

for gpu, cmds in commands.items():
    with open(f"procs-{gpu}.txt", "w") as f:
        f.writelines(cmds)