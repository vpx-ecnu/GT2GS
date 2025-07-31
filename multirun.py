import subprocess
import os
import random
import time
import math
import multiprocessing

def gpu_worker_process(gpu_id: int, tasks_for_this_gpu: list):
    
    log_file_path = f"gpu_worker_{gpu_id}.log"
    log_file_handle = open(log_file_path, "w")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[{time.time():.2f}s] Worker for GPU {gpu_id} started. PID: {os.getpid()} - Tasks: {len(tasks_for_this_gpu)}")

    for i, (scene_type, scene_name, style_type, style_name) in enumerate(tasks_for_this_gpu):
        
        print(f"[{time.time():.2f}s] GPU {gpu_id} - Task {i+1}/{len(tasks_for_this_gpu)}: Processing {scene_name}-{style_name}...")

        command = [
            "python", "run.py",
            f"scene_type={scene_type}",
            f"scene_name={scene_name}",
            f"style_type={style_type}",
            f"style_name={style_name}"
        ]
        
        try:
            print(command)
            result = subprocess.run(
                command,
                check=True,
                stdout=log_file_handle,
                stderr=log_file_handle,
                text=True
            )
            # print(command)
            print(f"[{time.time():.2f}s] GPU {gpu_id} - Task {i+1}/{len(tasks_for_this_gpu)}: Completed {scene_name}-{style_name}.")
        except subprocess.CalledProcessError as e:
            print(f"[{time.time():.2f}s] GPU {gpu_id} - Task {i+1}/{len(tasks_for_this_gpu)}: FAILED {scene_name}-{style_name}.")
        except Exception as e:
            print(f"[{time.time():.2f}s] GPU {gpu_id} - Task {i+1}/{len(tasks_for_this_gpu)}: Unexpected error for {scene_name}-{style_name}: {e}")
            
            
    print(f"[{time.time():.2f}s] Worker for GPU {gpu_id} finished all {len(tasks_for_this_gpu)} tasks.")
    
def get_data_list():
    data_list = []
    
    scene = ["fern", "flower", "fortress", "orchids", "trex", "horns"]
    styles = os.listdir("/Datasets/original_data/styles")
    styles = sorted(styles)
    
    for i, s in enumerate(styles):
        if s.endswith(".jpg"):
            data_list.append(("llff", scene[i % len(scene)], "style", s))
    
    styles = os.listdir("/Datasets/original_data/new_tex")
    styles = sorted(styles)
        
    for i, s in enumerate(styles):
        if s.endswith(".jpg"):
            data_list.append(("llff", scene[i % len(scene)], "texture", s))
    
    # scene = ["trex"]
    # styles = ["grid_0066.jpg"]
    # styles = ['new_tex/' + s for s in styles]
        
    # for i, s in enumerate(styles):
    #     if s.endswith(".jpg"):
    #         train(scene[i % len(scene)], s)
            # break
        
    # scene = ['M60', 'truck']
    scene = ["family", "horse", "m60", "playground", "train", "truck"]
    styles = [
        "0.jpg", "15.jpg", "26.jpg", "31.jpg", "42.jpg", "64.jpg", "86.jpg", "97.jpg", 
        "103.jpg", "109.jpg", "114.jpg", "125.jpg", "16.jpg", "54.jpg", "92.jpg", 
        "10.jpg", "22.jpg", "66.jpg", "88.jpg", "110.jpg", "121.jpg", "127.jpg", 
        "25.jpg", "90.jpg", "23.jpg", "3.jpg", "40.jpg", "46.jpg", "73.jpg", 
        "107.jpg", "123.jpg"
    ]
    
    for i, s in enumerate(styles):
        if s.endswith(".jpg"):
            data_list.append(("tnt", scene[i % len(scene)], "style", s))
    
    styles = os.listdir("/Datasets/original_data/new_tex")
    styles = sorted(styles)
        
    for i, s in enumerate(styles):
        if s.endswith(".jpg"):
            data_list.append(("tnt", scene[i % len(scene)], "texture", s))
            
    return data_list
    
if __name__ == "__main__":
    # Crucial for CUDA in multiprocessing: 'spawn' or 'forkserver'
    multiprocessing.set_start_method('spawn', force=True)
    data_list = get_data_list()
    # print(data_list)
    # exit(0)
    
    # processed_data_list = []
    # for scene_name, style_name in data_list:
    #     style_type = "texture"
    #     if style_name and style_name[0].isdigit():
    #         style_type = "style"
    #     processed_data_list.append((scene_name, style_name, style_type))

    # Randomly shuffle the tasks
    random.shuffle(data_list)
    print(f"Total {len(data_list)} tasks shuffled.")

    gpu_ids_to_use = [0, 1, 2] # As requested: 1 ~ 3
    num_gpus = len(gpu_ids_to_use)

    num_tasks = len(data_list)
    tasks_per_gpu = math.floor((num_tasks + num_gpus - 1) / num_gpus)
    assigned_tasks_per_gpu = []
    for i in range(num_gpus):
        start_idx = i * tasks_per_gpu
        end_idx = min((i + 1) * tasks_per_gpu, num_tasks)
        assigned_tasks_per_gpu.append(data_list[start_idx:end_idx])
        print(f"GPU {gpu_ids_to_use[i]} will process {len(assigned_tasks_per_gpu[i])} tasks.")
    # exit(0)

    worker_processes = []
    start_time = time.time()

    for i, gpu_id in enumerate(gpu_ids_to_use):
        if not assigned_tasks_per_gpu[i]:
            print(f"No tasks assigned to GPU {gpu_id}. Skipping process creation.")
            continue

        p = multiprocessing.Process(
            target=gpu_worker_process,
            args=(gpu_id, assigned_tasks_per_gpu[i]) # Pass the list of tasks for this GPU
        )
        p.daemon = True
        worker_processes.append(p)
        p.start() # Asynchronously start the worker process
        print(f"[{time.time():.2f}s] Main: Launched process for GPU {gpu_id} (PID: {p.pid}).")

    print(f"\n[{time.time():.2f}s] All GPU processes launched asynchronously. Main process will now wait for them to finish.")

    # Wait for all worker processes to complete
    for p in worker_processes:
        p.join() # Wait for each worker process to complete

    print(f"\n[{time.time():.2f}s] All worker processes have finished. Total execution time: {time.time() - start_time:.2f}s.")
