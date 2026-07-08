import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from dncnn_model import DnCNN

# -------------------------- 配置参数 --------------------------
batch_size = 4        # 每次训练喂给模型的图片数，电脑配置差就改2
epochs = 50           # 训练总轮数
learning_rate = 1e-4  # 学习率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动用显卡/CPU
data_root = "denoise_dataset"  # 数据集路径
save_path = "denoise_model.pth" # 模型保存路径
# -------------------------------------------------------------

# 1. 定义数据集读取类
# 1. 定义数据集读取类
class DenoiseDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.clear_dir = os.path.join(root_dir, split, "clear")
        self.noisy_dir = os.path.join(root_dir, split, "noisy")
        self.img_list = os.listdir(self.clear_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        # 读取灰度图
        clear_img = cv2.imread(os.path.join(self.clear_dir, img_name), 0)
        noisy_img = cv2.imread(os.path.join(self.noisy_dir, img_name), 0)
        
        # 统一缩放到 256x256，解决尺寸不一致报错
        clear_img = cv2.resize(clear_img, (256, 256))
        noisy_img = cv2.resize(noisy_img, (256, 256))
        
        # 归一化到0-1，转成Tensor
        clear_img = torch.from_numpy(clear_img).float() / 255.0
        noisy_img = torch.from_numpy(noisy_img).float() / 255.0
        
        # 增加通道维度 [H,W] → [1,H,W]
        clear_img = clear_img.unsqueeze(0)
        noisy_img = noisy_img.unsqueeze(0)
        
        return noisy_img, clear_img

# 2. 加载数据集
train_dataset = DenoiseDataset(data_root, "train")
val_dataset = DenoiseDataset(data_root, "val")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 3. 初始化模型、损失函数、优化器
model = DnCNN(channels=1).to(device)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 训练循环
best_psnr = 0
print(f"开始训练，使用设备：{device}")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    # 训练阶段
    for noisy, clear in tqdm(train_loader, desc=f"第{epoch+1}轮训练"):
        noisy = noisy.to(device)
        clear = clear.to(device)
        
        # 计算真实噪声
        real_noise = clear - noisy
        
        # 前向传播：模型预测噪声
        pred_noise = model(noisy)
        
        # 计算损失
        loss = criterion(pred_noise, real_noise)
        
        # 反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # 验证阶段
    model.eval()
    val_psnr = 0
    with torch.no_grad():
        for noisy, clear in val_loader:
            noisy = noisy.to(device)
            clear = clear.to(device)
            
            pred_noise = model(noisy)
            denoised = noisy - pred_noise  # 带噪图 - 预测噪声 = 去噪图
            
            # 转成numpy计算PSNR
            denoised_np = denoised.squeeze().cpu().numpy()
            clear_np = clear.squeeze().cpu().numpy()
            val_psnr += psnr(clear_np, denoised_np)
    
    avg_val_psnr = val_psnr / len(val_loader)
    
    # 打印结果
    print(f"【第{epoch+1}/{epochs}轮】 训练损失：{avg_train_loss:.6f}  |  验证集PSNR：{avg_val_psnr:.2f} dB")
    
    # 保存效果最好的模型
    if avg_val_psnr > best_psnr:
        best_psnr = avg_val_psnr
        torch.save(model.state_dict(), save_path)
        print(f"  → 模型已更新，当前最佳PSNR：{best_psnr:.2f} dB")

print("✅ 训练全部完成！最优模型已保存为 denoise_model.pth")