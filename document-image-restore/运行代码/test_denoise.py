import torch
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from dncnn_model import DnCNN

# ============ 只需修改这里的图片名 ============
test_img_name = "1.jpg"
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "denoise_model.pth"

# 图片路径
path_noisy = f"denoise_dataset/test/noisy/{test_img_name}"
path_clear = f"denoise_dataset/test/clear/{test_img_name}"

# 加载模型
model = DnCNN(channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 读取灰度图
img_noisy = cv2.imread(path_noisy, 0)
img_clear = cv2.imread(path_clear, 0)
if img_noisy is None or img_clear is None:
    print("错误：图片不存在，请修改test_img_name为你文件夹内真实图片名称")
else:
    # 预处理
    noisy_tensor = torch.from_numpy(img_noisy).float() / 255.0
    noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0).to(device)
    # 推理去噪
    with torch.no_grad():
        pred_noise = model(noisy_tensor)
        denoise_tensor = noisy_tensor - pred_noise
    # 转回图片
    denoise_img = denoise_tensor.squeeze().cpu().numpy()
    denoise_img = np.clip(denoise_img * 255, 0, 255).astype(np.uint8)

    # 计算指标
    psnr_noisy = psnr(img_clear, img_noisy)
    ssim_noisy = ssim(img_clear, img_noisy)
    psnr_denoise = psnr(img_clear, denoise_img)
    ssim_denoise = ssim(img_clear, denoise_img)

    # 打印结果
    print("===== 量化指标对比 =====")
    print(f"带噪原图 PSNR:{psnr_noisy:.2f}dB SSIM:{ssim_noisy:.4f}")
    print(f"去噪结果 PSNR:{psnr_denoise:.2f}dB SSIM:{ssim_denoise:.4f}")
    print(f"PSNR提升 {psnr_denoise - psnr_noisy:.2f}dB")

    # 保存对比图
    cv2.imwrite("01_带噪原图.jpg", img_noisy)
    cv2.imwrite("02_去噪结果.jpg", denoise_img)
    cv2.imwrite("03_清晰标准图.jpg", img_clear)
    print("对比图片已保存至项目根目录")