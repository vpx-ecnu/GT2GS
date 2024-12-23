from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms

def downsample_and_rotate(image_path, save_path):

    img = Image.open(image_path)
    img = img.resize((640, 640), Image.Resampling.LANCZOS)
    # img_downsampled = img.resize((img.width // 4, img.height // 4), Image.Resampling.LANCZOS)
    img_downsampled = img
    rotated_images = []

    # crop scale(texture scale control)
    H_crop, W_crop = img.height // 2, img.width // 2
    crop_op = torchvision.transforms.CenterCrop((W_crop, H_crop))

    for i in range(0, 16):
        img_rotated = img_downsampled.rotate(22.5 * i, expand=True)
        # img_rotated = crop_op(img_rotated)
        rotated_images.append(img_rotated)
    
    width, height = (W_crop, H_crop)
    combined_img = Image.new('RGB', (4 * width, 4 * height))
    
    for i, img in enumerate(rotated_images):
        x = (i % 4) * width
        y = (i // 4) * height
        combined_img.paste(img, (x, y))
    
    combined_img.save(save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    downsample_and_rotate(args.image_path, args.save_path)
    
# /home/lwj/data/ARF-svox2/data/styles/145.jpg