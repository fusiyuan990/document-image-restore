import torch
import cv2
import numpy as np
from dncnn_model import DnCNN
from srresnet_model import SRResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载两个训练完成的模型
# 去噪模型
denoise_net = DnCNN(channels=1).to(device)
denoise_net.load_state_dict(torch.load("denoise_model.pth", map_location=device))
denoise_net.eval()
# 超分模型
sr_net = SRResNet().to(device)
sr_net.load_state_dict(torch.load("sr_model.pth", map_location=device))
sr_net.eval()

def restore_old_document(img_path, save_name="完整修复输出.jpg"):
    raw_img = cv2.imread(img_path, 0)
    if raw_img is None:
        print("图片读取失败，请核对文件名")
        return None, None
    tensor = torch.from_numpy(raw_img).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    # 第一步：去除噪声
    with torch.no_grad():
        noise = denoise_net(tensor)
        clean_tensor = tensor - noise
    # 第二步：4倍超分辨率放大
    with torch.no_grad():
        sr_out = sr_net(clean_tensor)
    result = sr_out.squeeze().cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(save_name, result)
    print(f"文档修复完成，输出文件：{save_name}")
    return raw_img, result

# 修改引号内为根目录图片，例如 "2.jpg"
origin_img, final_img = restore_old_document("2.jpg")