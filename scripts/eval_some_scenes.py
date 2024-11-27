import os
from multiprocessing import Process

def execute_command(device, scenes, styles, methods):
    for scene, style, method in zip(scenes, styles, methods):
        
        cmd = f"CUDA_VISIBLE_DEVICES={device} python train_style.py \
                --ip 0.0.0.0 \
                -s /data3/lwj/preprocessed_data/llff/{scene}  \
                --model_path /data3/lwj/ckpt/3dgs/origin_0/llff/{scene} \
                --style_image styles/{style}.jpg --port {6015 + int(device)} \
                --method {method} --color_transfer"        
        os.system(cmd)
        cmd = f"CUDA_VISIBLE_DEVICES={device} python render_llff_video.py \
                -s /data3/lwj/preprocessed_data/llff/{scene}  \
                -m output/style/{scene}/{style}fast"
        
        os.system(cmd)
        cmd = f"mv output/style/{scene}/{style}fast/{style}fast.mp4 /data3/lzl/RefNPR/mp4"
        print(cmd)
        os.system(cmd)
        
        # cmd = f"find output/style/{scene}/{style}{method}/render -type f -name \"*.jpg\" -exec rm -f {{}} \;"
        # print(cmd)
        # os.system(cmd)
        

devices = ['1', '2', '3']  # 固定四张显卡
# devices = ['3']  # 固定四张显卡

# scenes = ["fern", "fern", "flower", "flower", "flower", "flower", "fortress", "fortress", "horns", "horns", "leaves", "leaves", "orchids", "orchids", "room", "room", "trex", "trex", "trex", "trex"]

# styles = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 126, 133, 140]
# scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "trex"]
# styles = [32, 14, 120, 12, 140, 91, 119]
# scenes = ["trex", "horns"]
# styles = [38, 62]
# scenes = ["flower", "flower"]
# styles = [21, 28]
scenes = ["horns"]
styles = [17]
# 分配任务给每个设备
tasks = []
for scene, style in zip(scenes, styles):
# for scene in scenes:
#     for style in styles:
    for method in ["fast"]:
        tasks.append((scene, style, method))
    # break
print(tasks)
# exit()

tasks_per_device = (len(tasks) + len(devices) - 1) // len(devices)
tasks_for_devices = [tasks[i * tasks_per_device:min(len(tasks), (i + 1) * tasks_per_device)] for i in range(len(devices))]

# 创建并启动每个设备的进程
processes = []
for i, device in enumerate(devices):
    scenes_for_device = [task[0] for task in tasks_for_devices[i]]
    styles_for_device = [task[1] for task in tasks_for_devices[i]]
    methods_for_device = [task[2] for task in tasks_for_devices[i]]
    p = Process(target=execute_command, args=(device, scenes_for_device, styles_for_device, methods_for_device))
    processes.append(p)
    p.start()

# 等待所有进程完成
for p in processes:
    p.join()
