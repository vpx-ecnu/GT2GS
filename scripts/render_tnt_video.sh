export CUDA_VISIBLE_DEVICES=$3

sequence=(0)

# Loop through each element in the sequence
for num in "${sequence[@]}"
do
    python train_style.py \
            -s /data3/lwj/preprocessed_data/llff/$1  \
            --model_path /data3/lwj/ckpt/3dgs/origin_0/llff/$1 \
            --style_image styles/$2.jpg --port 6010 \
            --method fast --color_transfer \
            --content_hyper $num \

    mv output/style/$1/$2fast/ output/style/$1/$2fast_c$num/
done
    
/data3/lzl/ArtGaussian/output/style/horns/132fast_c0.01