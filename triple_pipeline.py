import torch
import cv2
import time
import psutil
import numpy as np
import os
from ultralytics import YOLO
from torchvision import models
import tensorflow as tf

# 1. GPU Memory Growth for TensorFlow (prevents it from hogging all RAM)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# 2. Hardware Monitoring Function
def get_gpu_load():
    try:
        with open("/sys/devices/gpu.0/load", "r") as f:
            return float(f.read().strip()) / 10.0
    except:
        return 0.0

# 3. Initialize Tiered Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tier 1: YOLO Detection (PyTorch)
yolo_model = YOLO('yolov8n.pt').to(device)

# Tier 2: ResNet Classification (PyTorch)
resnet_model = models.resnet18()
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.load_state_dict(torch.load('weights/resnet_crack_v1.pth'))
resnet_model.to(device).eval()

# Tier 3: U-Net Segmentation (TensorFlow/Keras)
unet_model = tf.keras.models.load_model('weights/unet_crack_weights.weights.h5')

# 4. Video Setup
input_video = './videos/train_video.mp4'
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Saving at half FPS because of Stride 2
out = cv2.VideoWriter('triple_pipeline_results.avi', cv2.VideoWriter_fourcc(*'XVID'), fps/2, (width, height))

# 5. Processing Loop with Stride 2
stats_path = "triple_model_performance_stats.csv"
with open(stats_path, "w") as f:
    f.write("Frame,Total_Inf(ms),YOLO(ms),ResNet(ms),UNet(ms),CPU(%),GPU_Load(%),RAM_Usage(MB)\n")
    
    frame_idx = 0
    try:
        while cap.isOpened():
            # STRIDE 2 LOGIC
            cap.grab() # Skip the odd frame
            ret, frame = cap.retrieve() # Get the even frame
            if not ret: break
            
            frame_idx += 1
            t_start = time.time()
            
            # TIER 1: YOLO
            t0 = time.time()
            yolo_results = yolo_model(frame, verbose=False)
            t_yolo = (time.time() - t0) * 1000
            
            # TIER 2: ResNet
            t1 = time.time()
            img_r = cv2.resize(frame, (224, 224))
            img_r = torch.from_numpy(img_r).permute(2,0,1).float().unsqueeze(0).to(device)
            with torch.no_grad():
                _ = resnet_model(img_r)
            t_resnet = (time.time() - t1) * 1000
            
            # TIER 3: U-Net
            t2 = time.time()
            img_u = cv2.resize(frame, (256, 256)) / 255.0
            img_u = np.expand_dims(img_u, axis=0)
            _ = unet_model.predict(img_u, verbose=0)
            t_unet = (time.time() - t2) * 1000
            
            total_inf = (time.time() - t_start) * 1000
            
            # Hardware Stats
            cpu = psutil.cpu_percent()
            gpu = get_gpu_load()
            ram = psutil.virtual_memory().used / (1024 * 1024)
            
            # Save stats
            f.write(f"{frame_idx},{total_inf:.2f},{t_yolo:.2f},{t_resnet:.2f},{t_unet:.2f},{cpu},{gpu},{ram:.2f}\n")
            out.write(frame)
            
            if frame_idx % 20 == 0:
                print(f"Frame {frame_idx} | Pipeline Latency: {total_inf:.1f}ms | GPU: {gpu}%")
                f.flush() # Secure the data in case of crash

    finally:
        cap.release()
        out.release()
        print(f"\n[SUCCESS] Results saved: triple_pipeline_results.avi and {stats_path}")
