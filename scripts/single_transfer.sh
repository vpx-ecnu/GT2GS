

export CUDA_VISIBLE_DEVICES=$3
# python train_style.py --ip 0.0.0.0 -s data/$1/ --model_path output/$1 --style_image styles/$2
if [ $# -eq 3 ]; then
    port=6009
else
    port=$4
fi

python train_style.py --ip 0.0.0.0 \
    -s /Datasets/preprocessed_data/llff/$1 \
    --model_path /Datasets/ckpt/3dgs/origin_0/llff/$1 \
    --style_image /Datasets/styles/$2 \
    --port $port \
    --color_transfer \

