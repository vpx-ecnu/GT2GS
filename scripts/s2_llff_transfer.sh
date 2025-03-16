python style_main.py --ip 0.0.0.0 \
    -s /data3/lwj/preprocessed_data/llff/$1 \
    -o /data3/lwj/ckpt/radegs_0/llff/$1 \
    --style_image /home/lwj/data/ARF-svox2/data/styles/$2 \
    --port 6012 \
    --color_transfer \
    --name $3 \
    --gta_type clip \
    --prior \
    --density \

# 143.jpg