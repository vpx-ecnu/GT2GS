export CUDA_VISIBLE_DEVICES=$5
python train_style.py --ip 0.0.0.0 \
    -s /Datasets/preprocessed_data/llff/$1/\
    --model_path /Datasets/ckpt/3dgs/origin_0/llff/$1 \
    --style_image /Datasets/styles/$2.jpg \
    --scene_prompt $3 --style_prompt $4 \
    --color_transfer \
    --isolate --erode \
    --port $6\
