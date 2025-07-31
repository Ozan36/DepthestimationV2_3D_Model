# inference/models/depth_estimation/depthestimation.py

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class DepthEstimator:
    def __init__(self):
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        
    def predict(self, image_path):
        # 1. Resmi aç ve RGB'ye çevir (alpha kanalı varsa kaldırır)
        img = Image.open(image_path).convert("RGB")
    
        # 2. PIL Image → NumPy array (RGB formatında, [H, W, 3])
        img_np = np.array(img)
    
        # 3. Transformu uygula (artık NumPy array olduğu için /255 işlemi çalışır)
        input_data = self.transform(img_np)
    
        # 4. Transform çıktısı dict ise "image" anahtarını al, değilse direkt kullan
        input_batch = input_data["image"] if isinstance(input_data, dict) else input_data
    
        # 5. Model ile derinlik haritası oluştur
        with torch.no_grad():
            prediction = self.model(input_batch)
    
        # 6. Sonucu normalize et (0-255 aralığına getir)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    
        return depth_map_normalized.astype(np.uint8)