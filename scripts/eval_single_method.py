import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--device", required=True, type=int)
parser.add_argument("--method", required=True, type=str)
parser.add_argument("--port", required=True, type=int)

args = parser.parse_args()

lst = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

for i, scene in enumerate(lst):
    data_path = f"/data3/lwj/preprocessed_data/llff/{scene}"
    model_path = f"/data3/lwj/ckpt/3dgs/origin_0/llff/{scene}"
    
    for k in range(0, 141):
        
        output_path = f"output/style/{scene}/{k}{args.method}"
        output_render_path = os.path.join(output_path, "render")
        if os.path.exists(output_render_path):
            print(f"Already Exists {scene} {k}")
            continue
        
        cmd = f"CUDA_VISIBLE_DEVICES={args.device} python train_style.py -s {data_path}  --model_path {model_path} --style_image styles/{k}.jpg --port {args.port} --method {args.method}"
        print(cmd)
        os.system(cmd)
        
        
        if not os.path.exists(output_render_path):
            print("Error")
            exit()
            
        cmd = f"rm {output_path}/point_cloud/iteration_1399/*"
        print(cmd)
        os.system(cmd)
        
        
         
        
        
        
