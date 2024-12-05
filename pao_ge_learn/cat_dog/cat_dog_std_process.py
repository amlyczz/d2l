import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 数据目录
data_dir = "../data/data_cat_dog"

# 动态获取所有子目录名称作为类别
categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# 初始化变量
sum_pixels = np.zeros(3)  # 用于累计 RGB 像素总和
sum_squared_pixels = np.zeros(3)  # 用于累计 RGB 像素平方和
total_pixels = 0  # 累计总像素点数

print("加载图片中...")
for category in categories:
    category_path = os.path.join(data_dir, category)
    print(f"正在处理类别: {category}")
    for filename in tqdm(os.listdir(category_path)):
        img_path = os.path.join(category_path, filename)
        try:
            # 打开图片，转为 RGB
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img) / 255.0  # 归一化到 [0, 1]

            # 累加像素总和与平方和（按通道计算）
            sum_pixels += img_array.sum(axis=(0, 1))  # 对高度和宽度求和
            sum_squared_pixels += (img_array ** 2).sum(axis=(0, 1))  # 像素平方和
            total_pixels += img_array.shape[0] * img_array.shape[1]  # 累计像素点数
        except Exception as e:
            print(f"跳过文件: {img_path}，错误: {e}")

# 计算每个通道的均值和标准差
print("计算均值和标准差...")
mean = sum_pixels / total_pixels
std = np.sqrt(sum_squared_pixels / total_pixels - mean ** 2)

print(f"均值 (RGB): {mean}")
print(f"标准差 (RGB): {std}")
