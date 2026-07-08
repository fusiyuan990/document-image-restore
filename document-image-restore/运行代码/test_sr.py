import torch
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from srresnet_model import SRResNet

# 文件夹里文件是 2.jpg，必须带后缀
test_img_name = "2.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
model.load_state_dict(torch.load("sr_model.pth", map_location=device))
model.eval()

lr_path = f"sr_dataset/test/lr/{test_img_name}"
hr_path = f"sr_dataset/test/hr/{test_img_name}"
lr_img = cv2.imread(lr_path, 0)
hr_img = cv2.imread(hr_path, 0)

if lr_img is None or hr_img is None:
    print("❌ 图片不存在，请修改test_img_name为文件夹内真实图片名称")
else:
    tensor = torch.from_numpy(lr_img).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_hr = model(tensor)
    out_img = pred_hr.squeeze().cpu().numpy()
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)

    # 适配高清图尺寸，消除维度不一致报错
    lr_baseline = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]))
    psnr_base = peak_signal_noise_ratio(hr_img, lr_baseline)
    psnr_sr = peak_signal_noise_ratio(hr_img, out_img)
    ssim_base = structural_similarity(hr_img, lr_baseline)
    ssim_sr = structural_similarity(hr_img, out_img)

    print("=====超分量化指标对比=====")
    print(f"简单放大低清图 PSNR:{psnr_base:.2f}dB  SSIM:{ssim_base:.4f}")
    print(f"SR超分模型输出 PSNR:{psnr_sr:.2f}dB  SSIM:{ssim_sr:.4f}")

    cv2.imwrite("01_低清原图放大.jpg", lr_baseline)
    cv2.imwrite("02_超分修复结果.jpg", out_img)
    cv2.imwrite("03_高清标准原图.jpg", hr_img)
    print("对比图片已保存到项目根目录")