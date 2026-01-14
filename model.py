import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


def haar_wavelet_filters():
    low_filter = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.float32)
    high_filter = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.float32)
    return low_filter, high_filter


class MultiScaleWaveletDecompose(nn.Layer):
    def __init__(self, channels, levels=3):
        super().__init__()
        self.levels = levels
        L, H = haar_wavelet_filters()

        self.L1_2D = np.outer(L, L)
        self.H1_2D = np.outer(H, L)
        self.H2_2D = np.outer(L, H)
        self.H3_2D = np.outer(H, H)

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
        conv = nn.Conv2D(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias_attr=False,
            groups=channels
        )
        kernel_tensor = np.zeros((channels, 1, 2, 2), dtype=np.float32)
        for i in range(channels):
            kernel_tensor[i, 0] = kernel
        conv.weight.set_value(kernel_tensor)
        return conv

    def forward(self, x):
        scales = []
        current = x
        for level in range(self.levels):
            convs = self.conv_layers[level]
            low = convs['low'](current)
            high1 = convs['high1'](current)
            high2 = convs['high2'](current)
            high3 = convs['high3'](current)
            scales.append((low, high1, high2, high3))
            current = low
        return scales


class MSDWA(nn.Layer):
    def __init__(self, channels, levels=3):
        super().__init__()
        self.levels = levels
        self.wavelet_decompose = MultiScaleWaveletDecompose(channels, levels)

        self.attention = nn.Sequential(
            nn.Conv2D(channels * 4 * levels, channels, kernel_size=1),
            nn.BatchNorm(channels),
            nn.ReLU(),
            nn.Conv2D(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        scales = self.wavelet_decompose(x)

        all_features = []
        for (low, high1, high2, high3) in scales:
            low_up = F.interpolate(low, size=(h, w), mode='nearest')
            high1_up = F.interpolate(high1, size=(h, w), mode='nearest')
            high2_up = F.interpolate(high2, size=(h, w), mode='nearest')
            high3_up = F.interpolate(high3, size=(h, w), mode='nearest')
            all_features.extend([low_up, high1_up, high2_up, high3_up])

        fused_features = paddle.concat(all_features, axis=1)
        attn_weights = self.attention(fused_features)

        return x * attn_weights + x


class RoadDiseaseClassifier(nn.Layer):
    def __init__(self, num_classes=4, use_msdwa=True):
        super().__init__()
        self.use_msdwa = use_msdwa

        self.down1 = Encoder(num_channels=3, num_filters=64)
        self.down2 = Encoder(num_channels=64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)

        if self.use_msdwa:
            self.msdwa = MSDWA(channels=512, levels=3)

        self.global_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        _, x = self.down4(x)

        if self.use_msdwa:
            x = self.msdwa(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def get_features(self, x):
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        _, x = self.down4(x)

        if self.use_msdwa:
            x = self.msdwa(x)

        return x


class Encoder(nn.Layer):
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
        x_conv = x
        x_pool = self.pool(x)
        return x_conv, x_pool