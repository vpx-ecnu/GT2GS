export CUDA_VISIBLE_DEVICES=$3
# python train_style.py --ip 0.0.0.0 -s data/$1/ --model_path output/$1 --style_image styles/$2

python train_style.py --ip 0.0.0.0 \
    -s /data3/lwj/preprocessed_data/tnt/$1 \
     --model_path /data3/lwj/ckpt/3dgs/origin_0/tnt/$1 \
     --style_image styles/$2.jpg \
     --port $4 \
     --color_transfer \
     --stage_one 1000 \
     --stage_two 2000 \

