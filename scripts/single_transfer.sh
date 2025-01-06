

export CUDA_VISIBLE_DEVICES=2
# python train_style.py --ip 0.0.0.0 -s data/$1/ --model_path output/$1 --style_image styles/$2
# if [ $# -eq 3 ]; then
#     port=6009
# else
#     port=$4
# fi

# python train_style.py --ip 0.0.0.0 \
#     -s /data3/lwj/preprocessed_data/DTU/scan106 \
#     --model_path /home/lwj/data/TAT-GS/output/blender/scan106 \
#     --style_image /home/lwj/data/TAT-GS/texture_refinement/test_img_list \
#     -r 2 \
#     --port 6012 \
#     --color_transfer \

python train_style.py --ip 0.0.0.0 \
    -s /data3/lwj/preprocessed_data/llff/flower \
    --model_path /data3/lwj/ckpt/3dgs/origin/llff/flower \
    --style_image /home/lwj/data/TAT-GS/texture_refinement/test_img_list \
    --port 6012 \
    --color_transfer \

# python train_style.py --ip 0.0.0.0 \
#     -s /data3/lwj/original_data/nerf_synthetic/drums \
#     --model_path /data3/lwj/ckpt/3dgs/origin/nerf_synthetic/drums \
#     --style_image /home/lwj/data/TAT-GS/texture_refinement/tex \
#     --port 6012 \
#     --color_transfer \
