"""경량 캡차 인식 모델.

분할(segmentation) 없이, CNN 특징맵을 가로로 6등분해 위치별 26-클래스 분류.
파라미터 약 0.6M, CPU 추론 수 ms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class LightCaptchaNet(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, num_pos=config.NUM_POS,
                 widths=(32, 64, 128, 128)):
        super().__init__()
        self.num_pos = num_pos
        chans = 3
        blocks = []
        for w in widths:
            blocks.append(ConvBlock(chans, w))
            chans = w
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(chans, num_classes, 1)

    def forward(self, x):
        x = self.features(x)                              # B x C x H' x W'
        x = self.dropout(x)
        x = F.adaptive_avg_pool2d(x, (1, self.num_pos))   # B x C x 1 x P
        x = self.head(x)                                  # B x n_cls x 1 x P
        return x.squeeze(2).permute(0, 2, 1)              # B x P x n_cls


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = LightCaptchaNet()
    dummy = torch.randn(2, 3, config.IMG_H, config.IMG_W)
    out = net(dummy)
    print("output shape:", tuple(out.shape))     # (2, 6, 26)
    print(f"params: {param_count(net):,}")
