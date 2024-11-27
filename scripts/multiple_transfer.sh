# export CUDA_VISIBLE_DEVICES=$6
# python train_style.py --ip 0.0.0.0 \
#     -s /data3/lwj/preprocessed_data/llff/$1/\
#     --model_path /data3/lwj/ckpt/3dgs/origin_0/llff/$1 \
#     --scene_prompt $2 \
#     --style_image styles/$3.jpg styles/$4.jpg styles/$5.jpg \
#     --color_transfer --port $7 \

export CUDA_VISIBLE_DEVICES=$5
python train_style.py --ip 0.0.0.0 \
    -s /data3/lwj/preprocessed_data/llff/$1/\
    --model_path /data3/lwj/ckpt/3dgs/origin_0/llff/$1 \
    --scene_prompt $2 \
    --style_image styles/$3.jpg styles/$4.jpg \
    --color_transfer --port $6 \
