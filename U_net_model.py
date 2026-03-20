import torch
import torch.nn as nn

# 简易卷积块：Conv2d -> ReLU -> BatchNorm
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

# U-Net 模型
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256]):
        super().__init__()
        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_ch = in_channels
        for feat in features:
            self.enc_blocks.append(ConvBlock(prev_ch, feat))
            prev_ch = feat
        # 解码器
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for feat in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(feat * 2, feat))
            prev_ch = feat
        # 最终输出
        self.final_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)
    def forward(self, x):
        # 编码路径
        enc_features = []
        for enc in self.enc_blocks:
            x = enc(x)
            enc_features.append(x)
            x = self.pool(x)
        # 底部
        x = enc_features[-1]
        # 解码路径
        for up, dec, enc_feat in zip(self.up_convs, self.dec_blocks, reversed(enc_features)):
            x = up(x)
            # 拼接跳连特征
            if x.shape[2] != enc_feat.shape[2] or x.shape[3] != enc_feat.shape[3]:
                H = min(x.shape[2], enc_feat.shape[2])
                W = min(x.shape[3], enc_feat.shape[3])

                x = x[:, :, :H, :W]
                enc_feat = enc_feat[:, :, :H, :W]
            x = torch.cat([x, enc_feat], dim=1)
            x = dec(x)
        # 输出
        out = self.final_conv(x)
        return out  # 可视为输出“去噪后谱”或“噪声残差”
