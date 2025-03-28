import cv2
import numpy as np
import os

def create_nx_n_image(image, n):
    """
    输入图像，输出由n*n个原图像拼接而成的新图像。
    
    @param image: 输入图像 (NumPy array)。
    @param n: 拼接网格的大小 (n x n)。
    @return: 拼接后的图像 (NumPy array)。
    """
    # 获取输入图像的高度和宽度
    h, w = image.shape[:2]
    
    # 首先按行拼接
    row_images = [np.hstack([image] * n) for _ in range(n)]  # 创建每一行，水平拼接n个图像
    
    # 然后按列拼接
    full_image = np.vstack(row_images)  # 垂直拼接n行图像
    
    return full_image

# 示例
if __name__ == "__main__":
    # 读取图像
    image_path = "/home/lwj/data/ARF-svox2/data/styles/143.jpg"
    image = cv2.imread(image_path)

    # 设置拼接的网格大小，4x4为例
    n = 3

    # 获取拼接后的图像
    result_image = create_nx_n_image(image, n)
    
    base_name = os.path.basename(image_path)
    file_name_without_ext, ext = os.path.splitext(base_name)
    save_path = f"/home/lwj/data/MM2025/TAT-GS/texture_refinement/{file_name_without_ext}_{n}x{n}{ext}"

    # 保存或显示结果
    cv2.imwrite(save_path, result_image)
