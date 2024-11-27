import numpy as np
import os
import cv2

def read_and_resize_image(image_path):
    
    image = cv2.imread(image_path) 
    return image

def resize(image, h, w):
    resized_image = cv2.resize(image, (h, w), interpolation=cv2.INTER_AREA)
    return resized_image

def main():
    
    lst = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    
    
    for i, scene in enumerate(lst):
        for j in range(141):
            
            print(i, j)
            
            
            goal_path = os.path.join("debug/ablation", scene, f"{j}")
            if os.path.exists(goal_path):
                print(f"Already Exist {scene} {j}.jpg")
                continue
            original_image_path = os.path.join("/data3/lwj/preprocessed_data/llff", scene, "images")
            original_images = []
            for f in os.listdir(original_image_path):
                original_images.append(read_and_resize_image(os.path.join(original_image_path, f)))
            num_images = len(original_images)
            
            fast_image_path = os.path.join("output/style/", scene, f"{j}fast/render")
            fast_images = []
            for f in os.listdir(fast_image_path):
                fast_images.append(read_and_resize_image(os.path.join(fast_image_path, f)))
                
            nnfm_image_path = os.path.join("output/style/", scene, f"{j}nnfm/render")
            nnfm_images = []
            for f in os.listdir(nnfm_image_path):
                nnfm_images.append(read_and_resize_image(os.path.join(nnfm_image_path, f)))

                
            assert len(original_images) == len(fast_images)
            assert len(original_images) == len(nnfm_images)
            assert len(original_images) != 0
            
            style_image_path = os.path.join("styles", f"{j}.jpg")
            style_image = read_and_resize_image(style_image_path)
            tmp_w, tmp_h, _ = original_images[0].shape
            style_image = resize(style_image, tmp_h, tmp_w)
            
            
            
            os.makedirs(goal_path, exist_ok=True)
            for k, original_image in enumerate(original_images):
                w, h, _ = original_image.shape
                fast_images[i] = resize(fast_images[i], h, w)
                nnfm_images[i] = resize(nnfm_images[i], h, w)
                
                tmp_image1 = np.concatenate([original_image, style_image], axis=1)
                tmp_image2 = np.concatenate([fast_images[i], nnfm_images[i]], axis=1)
                final_image = np.concatenate([tmp_image1, tmp_image2], axis=0)
                
                # print(os.path.join(goal_path, f"{k}.jpg"))
                cv2.imwrite(os.path.join(goal_path, f"{k}.jpg"), final_image)
                
            # exit()
if __name__ == "__main__":
    main()
            
            
            
            
            