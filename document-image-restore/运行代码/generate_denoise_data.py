import cv2
import numpy as np
import os
from tqdm import tqdm

# -------------------------- 可修改参数 --------------------------
# 清晰原图所在的文件夹路径
clear_img_dir = "dataset_raw"
# 生成的去噪数据集保存位置
output_dir = "denoise_dataset"
# 噪声强度：数值越大噪声越多，新手先用默认值即可
gaussian_var = 0.008        # 高斯噪声强度（模拟颗粒感）
salt_pepper_prob = 0.015    # 椒盐噪声比例（模拟霉点斑点）
# ---------------------------------------------------------------

# 自动创建所有需要的文件夹
os.makedirs(os.path.join(output_dir, "train", "clear"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "noisy"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "clear"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "noisy"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "clear"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "noisy"), exist_ok=True)

# 1. 定义两个加噪声的函数
def add_gaussian_noise(img, var=0.008):
    """添加高斯噪声，模拟老照片整体的颗粒感"""
    img_normalized = img / 255.0          # 把像素值从0-255缩放到0-1
    noise = np.random.normal(0, var ** 0.5, img.shape)
    img_noisy = img_normalized + noise
    img_noisy = np.clip(img_noisy, 0, 1)  # 把像素值限制在合法范围内
    return (img_noisy * 255).astype(np.uint8)

def add_salt_pepper_noise(img, prob=0.015):
    """添加椒盐噪声，模拟老照片上的黑白霉点"""
    out = img.copy()
    # 生成白色斑点（盐噪声）
    num_salt = int(prob * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    out[coords[0], coords[1]] = 255
    # 生成黑色斑点（椒噪声）
    num_pepper = int(prob * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    out[coords[0], coords[1]] = 0
    return out

# 2. 获取所有清晰图片的文件名
img_list = [f for f in os.listdir(clear_img_dir) if f.endswith(('.jpg', '.png'))]
print(f"找到 {len(img_list)} 张清晰原图")

# 3. 划分数据集：80%训练、10%验证、10%测试
np.random.shuffle(img_list)  # 打乱图片顺序，避免分布不均
train_num = int(len(img_list) * 0.8)
val_num = int(len(img_list) * 0.1)

train_imgs = img_list[:train_num]
val_imgs = img_list[train_num:train_num+val_num]
test_imgs = img_list[train_num+val_num:]

# 4. 批量生成退化图片
def process_dataset(img_names, split_name):
    print(f"正在生成{split_name}集...")
    for img_name in tqdm(img_names):
        # 读取图片，并转成灰度图（老文档多为黑白，还能减少计算量）
        img_path = os.path.join(clear_img_dir, img_name)
        img_clear = cv2.imread(img_path, 0)  # 参数0 = 灰度模式读取
        
        if img_clear is None:
            print(f"跳过无法读取的图片：{img_name}")
            continue
        
        # 添加混合噪声，模拟真实老照片效果
        img_noisy = add_gaussian_noise(img_clear, gaussian_var)
        img_noisy = add_salt_pepper_noise(img_noisy, salt_pepper_prob)
        
        # 分别保存清晰图和带噪图到对应文件夹
        cv2.imwrite(os.path.join(output_dir, split_name, "clear", img_name), img_clear)
        cv2.imwrite(os.path.join(output_dir, split_name, "noisy", img_name), img_noisy)

# 依次处理训练集、验证集、测试集
process_dataset(train_imgs, "train")
process_dataset(val_imgs, "val")
process_dataset(test_imgs, "test")

print("✅ 去噪数据集全部生成完成！")
print(f"训练集：{len(train_imgs)} 张")
print(f"验证集：{len(val_imgs)} 张")
print(f"测试集：{len(test_imgs)} 张")