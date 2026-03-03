from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms
import os

def downsample_and_rotate(image_path, save_path):

    img = Image.open(image_path)
    # crop_size = min(img.width, img.height)
    # pre_crop = torchvision.transforms.CenterCrop((crop_size, crop_size))
    # img = pre_crop(img) 
    img = img.resize((640, 640), Image.Resampling.LANCZOS)
    # img_downsampled = img.resize((img.width // 4, img.height // 4), Image.Resampling.LANCZOS)
    img_downsampled = img
    rotated_images = []

    # crop scale(texture scale control)
    # k=4
    H_crop, W_crop = img.height // 2, img.width // 2
    width, height = (W_crop, H_crop)
    
    crop_op = torchvision.transforms.CenterCrop((W_crop, H_crop)) 

    for i in range(0, 16):
        img_rotated = img_downsampled.rotate(22.5 * i, expand=False)
        # img_rotated = crop_op(img_rotated)
        new_img = img_rotated.resize((640, 640), Image.Resampling.LANCZOS)
        # new_img = Image.new('RGB', (k * width, k * height))
        # for i in range(0, k*k):
        #     x = (i % k) * width
        #     y = (i // k) * height
        #     new_img.paste(img_rotated, (x, y))
        rotated_images.append(new_img)
    
    for i in range(len(rotated_images)):
        rotated_images[i].save(os.path.join(save_path, str(i) + '.jpg'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='/home/lwj/data/TAT-GS/texture_refinement/test_img_list')
    args = parser.parse_args()
    downsample_and_rotate(args.image_path, args.save_path)
    
# /home/lwj/data/ARF-svox2/data/styles/145.jpg