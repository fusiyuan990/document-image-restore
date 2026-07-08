import cv2
import numpy as np
import os
from tqdm import tqdm

clear_img_dir = "dataset_raw"
output_dir = "sr_dataset"
scale = 4

# 自动创建文件夹
os.makedirs(os.path.join(output_dir, "train", "hr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train", "lr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "hr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val", "lr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "hr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test", "lr"), exist_ok=True)

def make_low_res(img, scale=4):
    h, w = img.shape
    lr = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    return lr

img_list = [f for f in os.listdir(clear_img_dir) if f.endswith(("jpg","png"))]
np.random.shuffle(img_list)
train_num = int(len(img_list)*0.8)
val_num = int(len(img_list)*0.1)
train_imgs = img_list[:train_num]
val_imgs = img_list[train_num:train_num+val_num]
test_imgs = img_list[train_num+val_num:]

def process(img_names, split):
    print(f"正在生成{split}集")
    for name in tqdm(img_names):
        path = os.path.join(clear_img_dir, name)
        hr = cv2.imread(path, 0)
        if hr is None:
            continue
        # 统一裁剪256尺寸
        h,w = hr.shape
        x_start = max(0, (w-256)//2)
        y_start = max(0, (h-256)//2)
        hr = hr[y_start:y_start+256, x_start:x_start+256]
        lr = make_low_res(hr, scale)
        cv2.imwrite(os.path.join(output_dir, split, "hr", name), hr)
        cv2.imwrite(os.path.join(output_dir, split, "lr", name), lr)

process(train_imgs, "train")
process(val_imgs, "val")
process(test_imgs, "test")
print("✅ 超分数据集生成完成，文件夹：sr_dataset")