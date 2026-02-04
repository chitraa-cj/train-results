import torch
import cv2
import time
import psutil
import numpy as np
import os
from ultralytics import YOLO
from torchvision import models as torch_models
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. GPU Memory Growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# 2. U-Net Architecture Definition
def build_unet(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)
    return models.Model(inputs=[inputs], outputs=[outputs])

# 3. Initialize Tiered Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tier 1: YOLO
yolo_model = YOLO('yolov8n.pt').to(device)

# Tier 2: ResNet
resnet_model = torch_models.resnet18()
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.load_state_dict(torch.load('weights/resnet_crack_v1.pth'))
resnet_model.to(device).eval()

# Tier 3: U-Net (Building then loading weights)
unet_model = build_unet()
unet_model.load_weights('weights/unet_crack_weights.weights.h5')
print("Triple-Model Pipeline Initialized Successfully.")

# 4. Video Setup
cap = cv2.VideoCapture('./videos/train_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('triple_pipeline_results.avi', cv2.VideoWriter_fourcc(*'XVID'), fps/2, (width, height))

# 5. Processing Loop
stats_path = "triple_model_performance_stats.csv"
with open(stats_path, "w") as f:
    f.write("Frame,Total_Inf(ms),YOLO(ms),ResNet(ms),UNet(ms),RAM(MB)\n")
    frame_idx = 0
    while cap.isOpened():
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret: break
        
        frame_idx += 1
        t_start = time.time()
        
        # YOLO Detection
        t0 = time.time()
        _ = yolo_model(frame, verbose=False)
        t_yolo = (time.time() - t0) * 1000
        
        # ResNet Classification
        t1 = time.time()
        img_r = cv2.resize(frame, (224, 224))
        img_r = torch.from_numpy(img_r).permute(2,0,1).float().unsqueeze(0).to(device)
        with torch.no_grad(): _ = resnet_model(img_r)
        t_resnet = (time.time() - t1) * 1000
        
        # UNet Segmentation
        t2 = time.time()
        img_u = cv2.resize(frame, (256, 256)) / 255.0
        img_u = np.expand_dims(img_u, axis=0)
        _ = unet_model.predict(img_u, verbose=0)
        t_unet = (time.time() - t2) * 1000
        
        total_inf = (time.time() - t_start) * 1000
        ram = psutil.virtual_memory().used / (1024 * 1024)
        
        f.write(f"{frame_idx},{total_inf:.2f},{t_yolo:.2f},{t_resnet:.2f},{t_unet:.2f},{ram:.2f}\n")
        out.write(frame)
        
        if frame_idx % 20 == 0:
            print(f"Frame {frame_idx} | Latency: {total_inf:.1f}ms | RAM: {ram:.0f}MB")
            f.flush()

cap.release()
out.release()
