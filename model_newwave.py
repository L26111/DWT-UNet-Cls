import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class SiLU(nn.Layer):
    def forward(self, x):
        return x * F.sigmoid(x)


class ConvBNSiLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELANBlock(nn.Layer):
    def __init__(self, in_c, out_c, num_layers=4):
        super().__init__()
        c_half = out_c // 2

        self.conv1 = ConvBNSiLU(in_c, c_half, 1)
        self.conv2 = ConvBNSiLU(in_c, c_half, 1)

        layers = []
        for i in range(num_layers):
            layers.append(ConvBNSiLU(c_half, c_half, 3, padding=1))
        self.layers = nn.LayerList(layers)

        total_in_channels = c_half * (2 + num_layers)
        self.concat_conv = ConvBNSiLU(total_in_channels, out_c, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        outs = [x1, x2]

        y = x1
        for conv in self.layers:
            y = conv(y)
            outs.append(y)

        concatenated = paddle.concat(outs, axis=1)
        return self.concat_conv(concatenated)


class DownSample(nn.Layer):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mp = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv = ConvBNSiLU(in_c, out_c, 1)

    def forward(self, x):
        return self.conv(self.mp(x))


class ChannelAttention(nn.Layer):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc = nn.Sequential(
            nn.Conv2D(channels, channels // ratio, 1, bias_attr=False),
            SiLU(),
            nn.Conv2D(channels // ratio, channels, 1, bias_attr=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x_cat = paddle.concat([avg_out, max_out], axis=1)
        return self.sigmoid(self.conv(x_cat))


def haar_wavelet_filters():
    low_filter = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.float32)
    high_filter = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.float32)
    return low_filter, high_filter


class WaveletDecompose(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        L, H = haar_wavelet_filters()
        L2D = np.outer(L, L)
        H2D = np.outer(H, H)

        low_kernel = np.zeros((channels, 1, 2, 2), dtype=np.float32)
        high_kernel = np.zeros((channels, 1, 2, 2), dtype=np.float32)
        for i in range(channels):
            low_kernel[i, 0] = L2D
            high_kernel[i, 0] = H2D

        self.low_conv = nn.Conv2D(
            channels, channels, kernel_size=2, stride=2, bias_attr=False, groups=channels
        )
        self.high_conv = nn.Conv2D(
            channels, channels, kernel_size=2, stride=2, bias_attr=False, groups=channels
        )

        self.low_conv.weight.set_value(low_kernel)
        self.high_conv.weight.set_value(high_kernel)

        self.low_conv.weight.stop_gradient = True
        self.high_conv.weight.stop_gradient = True

    def forward(self, x):
        low_freq = self.low_conv(x)
        high_freq = self.high_conv(x)
        return low_freq, high_freq


class WaveletAttentionV2(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.wavelet = WaveletDecompose(channels)
        self.ca_low = ChannelAttention(channels)
        self.ca_high = ChannelAttention(channels)
        self.sa_low = SpatialAttention()
        self.sa_high = SpatialAttention()

    def forward(self, x):
        low, high = self.wavelet(x)

        low = self.ca_low(low) * low
        low = self.sa_low(low) * low

        high = self.ca_high(high) * high
        high = self.sa_high(high) * high

        out = F.interpolate(low, size=x.shape[2:], mode='nearest') + \
              F.interpolate(high, size=x.shape[2:], mode='nearest')
        return out


class CSPDarknetXWithWavelet(nn.Layer):
    def __init__(self, num_classes=4, use_wavelet=True):
        super().__init__()
        self.use_wavelet = use_wavelet

        self.stem = ConvBNSiLU(3, 80, 3, stride=1, padding=1)

        self.down1 = DownSample(80, 160)
        self.stage1 = ELANBlock(160, 160, num_layers=2)

        self.down2 = DownSample(160, 320)
        self.stage2 = ELANBlock(320, 320, num_layers=2)

        self.down3 = DownSample(320, 640)
        self.stage3 = ELANBlock(640, 640, num_layers=2)

        self.down4 = DownSample(640, 1280)
        self.stage4 = ELANBlock(1280, 1280, num_layers=2)

        if self.use_wavelet:
            self.wavelet_attn = WaveletAttentionV2(1280)

        self.fc1 = nn.Sequential(nn.AdaptiveAvgPool2D(1), nn.Flatten())
        self.fc2 = nn.Linear(1280, num_classes)

        self.y_fc1 = None

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        self.down1_out = x
        x = self.stage1(x)
        x = self.down2(x)
        self.down2_out = x
        x = self.stage2(x)
        x = self.down3(x)
        self.down3_out = x
        x = self.stage3(x)
        x = self.down4(x)
        self.down4_out = x
        x = self.stage4(x)

        if self.use_wavelet:
            x = self.wavelet_attn(x)

        x = self.fc1(x)
        self.y_fc1 = x.clone()
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CSPDarknetXWithWavelet(num_classes=8, use_wavelet=True)
    paddle.summary(model, (1, 3, 224, 224))