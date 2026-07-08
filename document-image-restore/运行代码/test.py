import torch
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

print("=== 环境检测开始 ===")
print("PyTorch 版本:", torch.__version__)
print("OpenCV 版本:", cv2.__version__)
print("Numpy 版本:", np.__version__)

# 测试PSNR计算
test_img = np.ones((100, 100), dtype=np.uint8) * 128
score = psnr(test_img, test_img)
print("PSNR 测试值:", score)

print("=== 全部检测通过，环境配置成功！ ===")