

export CUDA_VISIBLE_DEVICES=2
# python train_style.py --ip 0.0.0.0 -s data/$1/ --model_path output/$1 --style_image styles/$2
# if [ $# -eq 3 ]; then
#     port=6009
# else
#     port=$4
# fi

python train_style.py --ip 0.0.0.0 \
    -s /data3/lwj/preprocessed_data/llff/trex \
    --model_path /data3/lwj/ckpt/3dgs/origin_0/llff/trex \
    --style_image /home/lwj/data/TAT-GS/texture_refinement/test_img_list \
    --port 6012 \
    --color_transfer \

