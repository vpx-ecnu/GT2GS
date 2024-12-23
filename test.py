import os

import cv2
import numpy as np

from utils.camera_utils import sort_cameras_by_angle, load_json_to_cameras,get_homography_matrix,compute_epipolar_constrains


def overlay_warped_image(base_image, warped_image):

    mask = (warped_image != 0).any(axis=-1).astype(np.uint8)
    base_image[mask == 1] = warped_image[mask == 1]

    return base_image

def warp_image_with_homography(image, H, target_height, target_width):
    H_np = H.cpu().numpy()
    warped_image = cv2.warpPerspective(image, H_np, (target_width, target_height))
    return warped_image

def main():

    camera_json = "cameras.json"
    ori_image_path = "images"

    cameras = load_json_to_cameras(camera_json,ori_image_path)

    sorted_cameras = sort_cameras_by_angle(cameras)

    output_folder = "sorted_images"
    os.makedirs(output_folder, exist_ok=True)
    
    for i, cam in enumerate(sorted_cameras):
        img_name = cam.image_name
        image_path = f"images/{img_name}.JPG"

        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            output_filename = os.path.join(output_folder, f"sorted_images_{i}.jpg")
            cv2.imwrite(output_filename, image)
            print(f"Saved sorted image {output_filename}")
        else:
            print(f"Image for camera {img_name} not found.")

    image_path = os.path.join(output_folder,"sorted_images_0.jpg")

    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_path} not found.")
        else:
            for i in range(1,len(cameras)):
                cam1 = cameras[i]
                cam2 = cameras[i-1]

                H = get_homography_matrix(cam1, cam2)
                
                # mask = compute_epipolar_constrains(cam1,cam2)
                # ...

                warped_image = warp_image_with_homography(image, H, cam2.image_height, cam2.image_width)

                temp_filename = f"temp_images/temp_image_{i}.jpg"
                cv2.imwrite(temp_filename, warped_image)
                print(f"Saved warped image {temp_filename}")

                image = overlay_warped_image(image, warped_image)

                output_filename = f"output_images/output_image_{i}.jpg"
                cv2.imwrite(output_filename, image)
                print(f"Saved output image {output_filename}")

if __name__ == "__main__":
    main()