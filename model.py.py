import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

# ----------------------------
# 1. 小波分解基础模块（多尺度支持）
# ----------------------------
def haar_wavelet_filters():
    """Haar小波滤波器定义"""
    low_filter = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.float32)
    high_filter = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=np.float32)
    return low_filter, high_filter


class MultiScaleWaveletDecompose(nn.Layer):
    """多尺度小波分解（支持1-3级分解）"""
    def __init__(self, channels, levels=3):
        super().__init__()
        self.levels = levels  # 分解尺度（1-3级）
        L, H = haar_wavelet_filters()
        
        # 构建2D小波滤波器
        self.L1_2D = np.outer(L, L)  # 低频滤波器
        self.H1_2D = np.outer(H, L)  # 垂直高频
        self.H2_2D = np.outer(L, H)  # 水平高频
        self.H3_2D = np.outer(H, H)  # 对角高频
        
        # 为每个尺度初始化卷积层
        self.conv_layers = nn.LayerList()
        for _ in range(levels):
            convs = nn.LayerDict({
                'low': self._build_wavelet_conv(channels, self.L1_2D),
                'high1': self._build_wavelet_conv(channels, self.H1_2D),
                'high2': self._build_wavelet_conv(channels, self.H2_2D),
                'high3': self._build_wavelet_conv(channels, self.H3_2D)
            })
            self.conv_layers.append(convs)

    def _build_wavelet_conv(self, channels, kernel):
        """构建小波卷积层（固定滤波器）"""
        conv = nn.Conv2D(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias_attr=False,
            groups=channels
        )
        # 初始化卷积核（每个通道共享相同滤波器）
        kernel_tensor = np.zeros((channels, 1, 2, 2), dtype=np.float32)
        for i in range(channels):
            kernel_tensor[i, 0] = kernel
        conv.weight.set_value(kernel_tensor)
        return conv

    def forward(self, x):
        """返回多尺度分解结果：[(low, high1, high2, high3), ...]"""
        scales = []
        current = x
        for level in range(self.levels):
            convs = self.conv_layers[level]
            low = convs['low'](current)
            high1 = convs['high1'](current)
            high2 = convs['high2'](current)
            high3 = convs['high3'](current)
            scales.append((low, high1, high2, high3))
            current = low  # 下一级分解基于当前低频分量
        return scales


# ----------------------------
# 2. MSDWA（多尺度小波注意力）模块
# ----------------------------
class MSDWA(nn.Layer):
    """多尺度小波注意力模块：融合多尺度小波特征并生成注意力权重"""
    def __init__(self, channels, levels=3):
        super().__init__()
        self.levels = levels
        self.wavelet_decompose = MultiScaleWaveletDecompose(channels, levels)
        
        # 注意力权重生成器（跨尺度信息融合）
        self.attention = nn.Sequential(
            nn.Conv2D(channels * 4 * levels, channels, kernel_size=1),  # 融合所有尺度特征
            nn.BatchNorm(channels),
            nn.ReLU(),
            nn.Conv2D(channels, channels, kernel_size=1),
            nn.Sigmoid()  # 输出0-1注意力权重
        )

    def forward(self, x):
        b, c, h, w = x.shape
        scales = self.wavelet_decompose(x)  # 多尺度分解结果
        
        # 收集所有尺度的特征并上采样到原始尺寸
        all_features = []
        for (low, high1, high2, high3) in scales:
            # 上采样到输入尺寸
            low_up = F.interpolate(low, size=(h, w), mode='nearest')
            high1_up = F.interpolate(high1, size=(h, w), mode='nearest')
            high2_up = F.interpolate(high2, size=(h, w), mode='nearest')
            high3_up = F.interpolate(high3, size=(h, w), mode='nearest')
            all_features.extend([low_up, high1_up, high2_up, high3_up])
        
        # 拼接所有特征并计算注意力权重
        fused_features = paddle.concat(all_features, axis=1)  # [b, c*4*levels, h, w]
        attn_weights = self.attention(fused_features)  # [b, c, h, w]
        
        # 注意力加权原始特征
        return x * attn_weights + x  # 残差连接


# ----------------------------
# 3. 主干网络：Baseline UNet + MSDWA模块
# ----------------------------
class RoadDiseaseClassifier(nn.Layer):
    """路面病害分类器：UNet + 多尺度小波注意力(MSDWA)"""
    def __init__(self, num_classes=4, use_msdwa=True):
        super().__init__()
        self.use_msdwa = use_msdwa
        
        # UNet Encoder
        self.down1 = Encoder(num_channels=3, num_filters=64)
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)
        
        # MSDWA模块（多尺度小波注意力）
        if self.use_msdwa:
            self.msdwa = MSDWA(channels=512, levels=3)  # 3级尺度分解
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # UNet Encoder特征提取
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        _, x = self.down4(x)
        
        # 应用多尺度小波注意力
        if self.use_msdwa:
            x = self.msdwa(x)
        
        # 分类输出
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def get_features(self, x):
        """提取分类前的特征，用于t-SNE可视化"""
        # UNet Encoder特征提取
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        _, x = self.down4(x)
        
        # 应用多尺度小波注意力
        if self.use_msdwa:
            x = self.msdwa(x)
        
        return x  # 返回全局池化前的特征


# ----------------------------
# 4. UNet Encoder模块（保持Baseline结构）
# ----------------------------
class Encoder(nn.Layer):
    """UNet编码器基本单元：两次卷积+批归一化+ReLU+最大池化"""
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm(num_filters, act="relu")
        
        self.conv2 = nn.Conv2D(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm(num_filters, act="relu")
        
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding="SAME")

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_conv = x  # 跳跃连接特征
        x_pool = self.pool(x)
        return x_conv, x_pool