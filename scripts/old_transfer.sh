python train_style.py --ip 0.0.0.0 \
    -s /Datasets/preprocessed_data/llff/$1 \
    -m /Datasets/radegs_0/llff/$1 \
    --style_image /Datasets/styles/texture/$2 \
    --port 6012 \
    --color_transfer \