from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms
import os

def downsample_and_rotate(image_path, save_path):

    img = Image.open(image_path)
    img = img.resize((640, 640), Image.Resampling.LANCZOS)
    # img_downsampled = img.resize((img.width // 4, img.height // 4), Image.Resampling.LANCZOS)
    img_downsampled = img
    rotated_images = []

    # crop scale(texture scale control)
    H_crop, W_crop = img.height // 2, img.width // 2
    width, height = (W_crop, H_crop)
    
    crop_op = torchvision.transforms.CenterCrop((W_crop, H_crop)) 

    for i in range(0, 16):
        img_rotated = img_downsampled.rotate(22.5 * i, expand=True)
        img_rotated = crop_op(img_rotated)
        new_img = Image.new('RGB', (2 * width, 2 * height))
        for i in range(0, 4):
            x = (i % 2) * width
            y = (i // 2) * height
            new_img.paste(img_rotated, (x, y))
        rotated_images.append(new_img)
    
    for i in range(len(rotated_images)):
        rotated_images[i].save(os.path.join(save_path, str(i) + '.jpg'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    downsample_and_rotate(args.image_path, args.save_path)
    
# /home/lwj/data/ARF-svox2/data/styles/145.jpg