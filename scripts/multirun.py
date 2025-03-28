
import os

def train(scene, style):
    output_path = f"output/2025-3-25/{scene}/{os.path.basename(style)}/"
    if os.path.exists(os.path.join(output_path, "render/0000.png")):
        return
    cmd = f"\
    python style_main.py \
        -s /Datasets/preprocessed_data/llff/{scene} \
        -o /Datasets/radegs_0/llff/{scene} \
        --style_image /Datasets/styles/GT2GS/{style} \
        -m {output_path} \
        --color_transfer \
        --gta_type clip \
        --density"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    
    scene = ["trex", "horns"]
    styles = os.listdir("/Datasets/styles/GT2GS")
    
    for a in scene:
        for b in styles:
            train(a, b)
    