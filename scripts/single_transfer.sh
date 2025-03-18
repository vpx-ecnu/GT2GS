# python style_main.py --ip 0.0.0.0 \
#     -s /data3/lwj/preprocessed_data/llff/$1 \
#     -o /data3/lwj/ckpt/radegs_0/llff/$1 \
#     --style_image /home/lwj/data/ARF-svox2/data/styles/$2 \
#     --port 6012 \
#     --color_transfer \

# 143.jpg

python style_main.py \
    -s /Datasets/preprocessed_data/llff/$1 \
    -o /Datasets/radegs_0/llff/$1 \
    --style_image /Datasets/styles/texture/$2 \
    --color_transfer \
    --gta_type clip \
    --prior \
    --density \
    --round 60 \
    --style_iter 10

