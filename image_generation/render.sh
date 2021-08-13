#!/bin/sh

output=/dfs/user/tailin/.results/CLEVR_relation

for i in $(seq 0 3); do
export CUDA_VISIBLE_DEVICES="$i"
./blender/blender --background --python render_images.py -- --num_images 15000 --use_gpu 1 --min_objects 2 --max_objects 6 \
	--output_image_dir $output/output$i/images/ --output_scene_dir $output/output$i/scenes/ --output_scene_file $output/CLEVR_scenes.json &
done
