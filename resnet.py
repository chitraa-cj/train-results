import torch
import time
import psutil
import os
from torchvision import models
import cv2

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2) 
model.load_state_dict(torch.load('weights/resnet_crack_v1.pth'))
model.to(device).eval()

# Video Setup
video_path = './videos/train_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get original video properties for the writer
frame_width = int(cap.get(cv2.get(3)))
frame_height = int(cap.get(cv2.get(4)))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize Video Writer (using XVID for .avi compatibility)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('resnet_inspection_results.avi', fourcc, fps, (frame_width, frame_height))

# Log File Setup
with open("resnet_performance_stats.txt", "w") as f:
    f.write("Frame, Inference_Time(ms), CPU(%), RAM(MB), GPU_Mem(MB)\n")
    
    frame_count = 0
    start_bench = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Preprocessing for Model
        img_resized = cv2.resize(frame, (224, 224))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Inference & Timing
        t1 = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        t2 = time.time()
        
        # Resource Monitoring
        inf_time = (t2 - t1) * 1000
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / (1024 * 1024)
        gpu_mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
        
        # Annotate Frame (Optional: adds visual proof of speed)
        label = f"ResNet Inf: {inf_time:.1f}ms"
        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save Frame to Output Video
        out.write(frame)
        
        frame_count += 1
        f.write(f"{frame_count}, {inf_time:.2f}, {cpu}, {ram:.2f}, {gpu_mem:.2f}\n")
        
        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}...")

    total_time = time.time() - start_bench
    f.write(f"\nTotal Processing Time: {total_time:.2f}s\n")

# Cleanup
cap.release()
out.release()
print("Results saved: resnet_inspection_results.avi and resnet_performance_stats.txt")
