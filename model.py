"""경량 캡차 인식 모델.

CNN 특징맵을 가로로 (num_pos*over) 칸으로 나눠 위치별 26-클래스 로짓을 뽑고,
학습 가능한 1D conv 로 인접 칸을 섞은 뒤 num_pos 로 모은다. 고정 6등분보다
글자 위치 변동(간격/여백 차이)에 조금 더 강하다. 파라미터 약 0.7M.
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
                 widths=(32, 64, 128, 128), over=4):
        super().__init__()
        self.num_pos = num_pos
        self.num_classes = num_classes
        self.over = over                                     # 자리당 세분 칸 수
        chans = 3
        blocks = []
        for w in widths:
            blocks.append(ConvBlock(chans, w))
            chans = w
        self.features = nn.Sequential(*blocks)
        self.drop2d = nn.Dropout2d(0.25)
        self.mix = nn.Sequential(                            # 인접 칸 섞기 (학습형)
            nn.Conv2d(chans, chans, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(chans, num_classes, 1)

    def forward(self, x):
        x = self.features(x)                                 # B x C x H' x W'
        x = self.drop2d(x)
        w = self.num_pos * self.over
        x = F.adaptive_avg_pool2d(x, (1, w))                 # B x C x 1 x (P*over)
        x = self.mix(x)
        x = self.head(x)                                     # B x n_cls x 1 x (P*over)
        x = x.view(-1, self.num_classes, self.num_pos, self.over).mean(-1)
        return x.permute(0, 2, 1)                            # B x P x n_cls


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = LightCaptchaNet()
    dummy = torch.randn(2, 3, config.IMG_H, config.IMG_W)
    out = net(dummy)
    print("output shape:", tuple(out.shape))     # (2, 6, 26)
    print(f"params: {param_count(net):,}")
