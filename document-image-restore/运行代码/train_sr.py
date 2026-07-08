import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2, os, numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from srresnet_model import SRResNet

batch_size = 4
epochs = 60
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "sr_dataset"
save_path = "sr_model.pth"

class SRDataset(Dataset):
    def __init__(self, root, split="train"):
        self.hr_dir = os.path.join(root, split, "hr")
        self.lr_dir = os.path.join(root, split, "lr")
        self.names = os.listdir(self.hr_dir)
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        hr = cv2.imread(os.path.join(self.hr_dir, name), 0)
        lr = cv2.imread(os.path.join(self.lr_dir, name), 0)
        # 强制统一尺寸，杜绝尺寸堆叠报错
        hr = cv2.resize(hr, (256, 256))
        lr = cv2.resize(lr, (64, 64))
        hr = torch.from_numpy(hr).float()/255.0
        lr = torch.from_numpy(lr).float()/255.0
        hr = hr.unsqueeze(0)
        lr = lr.unsqueeze(0)
        return lr, hr

train_set = SRDataset(data_root, "train")
val_set = SRDataset(data_root, "val")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

model = SRResNet().to(device)
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=lr)
best_psnr = 0

print("开始训练超分模型")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch{epoch+1}"):
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        pred = model(lr_img)
        loss = criterion(pred, hr_img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)

    model.eval()
    val_psnr = 0
    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            pred = model(lr_img)
            pred_np = pred.squeeze().cpu().numpy()
            hr_np = hr_img.squeeze().cpu().numpy()
            val_psnr += psnr(hr_np, pred_np)
    avg_psnr = val_psnr / len(val_loader)
    print(f"第{epoch+1}轮 损失:{avg_loss:.6f} 验证PSNR:{avg_psnr:.2f}dB")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), save_path)
        print(f"更新最优超分模型，最佳PSNR:{best_psnr:.2f}")
print("✅ 超分训练完成，权重sr_model.pth已保存")