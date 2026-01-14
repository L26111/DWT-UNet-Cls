import os
import numpy as np
from paddle.io import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, txt_file, transform=None, target_mode="RGB"):
        """
        Args:
            txt_file (str): 存储图片路径和标签的txt文件路径
            transform (callable, optional): 数据预处理变换
            target_mode (str): 图像转换模式，"RGB"或"L"（灰度图），确保所有图像通道一致
        """
        self.data = self._load_data(txt_file)
        self.transform = transform
        self.target_mode = target_mode  # 统一图像模式

    def __getitem__(self, index):
        # 解析图片路径和标签
        img_path, label = self.data[index].split(' ')
        label = int(label)  # 提前转换标签为int，避免重复操作

        # 处理图片文件
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"找不到图片文件：{img_path}")  # 主动抛错，避免后续逻辑异常
        
        # 打开图片并统一模式（如转为RGB）
        image = Image.open(img_path).convert(self.target_mode)
        
        # 关键修复：将PIL图像转为numpy数组，并将uint8转为float32且归一化到[0,1]
        image = np.array(image, dtype=np.float32) / 255.0  # 转换为float32并归一化
        
        # 应用预处理变换（此时输入已是float32，避免ToTensor报错）
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.data)

    def _load_data(self, txt_file):
        if not os.path.isfile(txt_file):
            raise FileNotFoundError(f"数据集索引文件不存在：{txt_file}")
        
        with open(txt_file, 'r') as f:
            data = f.read().splitlines()
        
        # 简单校验数据格式（每行应为"路径 标签"）
        for line in data:
            if len(line.split(' ')) != 2:
                raise ValueError(f"txt文件格式错误，应为'路径 标签'：{line}")
        
        return data
    