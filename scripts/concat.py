from PIL import Image

import os

def resize_and_concat(image1_path, image2_path, output_path, target_height=300):
    """
    将两张图片的高度调整为相同尺寸，然后水平拼接。

    参数:
        image1_path (str): 第一张图片的路径。
        image2_path (str): 第二张图片的路径。
        output_path (str): 拼接后图片的输出路径。
        target_height (int): 目标高度，默认为 300。
    """
    # 打开图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 计算调整后的宽度，保持宽高比
    width1 = int(target_height * img1.width / img1.height)
    width2 = int(target_height * img2.width / img2.height)

    # 调整图片大小
    img1_resized = img1.resize((width1, target_height))
    img2_resized = img2.resize((width2, target_height))

    # 创建新图片，宽度为两张图片宽度之和
    new_img = Image.new('RGB', (width1 + width2, target_height))

    # 将两张图片粘贴到新图片中
    new_img.paste(img1_resized, (0, 0))
    new_img.paste(img2_resized, (width1, 0))

    # 保存拼接后的图片
    new_img.save(output_path)

# 示例用法
# resize_and_concat("image1.jpg", "image2.jpg", "output.jpg")

if __name__ == "__main__":
    
    date = "2025-3-25"
    scene = ["trex", "horns"]
    styles = os.listdir("/Datasets/styles/GT2GS")
    os.makedirs(f"/workspace/test/{date}")
    
    for a in scene:
        for b in styles:
            
            output_path = f"output/2025-3-25/{a}/{os.path.basename(b)}/"
            scene_image = os.path.join(output_path, "render/0000.png")
            style_image = os.path.join("/Datasets/styles/GT2GS", b)
            
            resize_and_concat(style_image, scene_image, f"/workspace/test/{date}/{a}{b}")