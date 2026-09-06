"""합성 캡차 Dataset.

- SynthCaptcha : 매 __getitem__ 마다 새 캡차를 즉석 생성 (디스크 불필요, 사실상 무한).
- build_val    : 재현 가능한 고정 검증셋을 메모리에 만든다.
"""

import random

import torch
from torch.utils.data import Dataset

import config
from preprocess import to_tensor
from synth import CaptchaRenderer


def _augment(x, rng):
    """텐서 단계 augmentation (과적합 방지). x: 3xHxW, [-1,1]."""
    C, H, W = x.shape
    # 밝기 / 대비 지터
    if rng.random() < 0.8:
        x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.15, 0.15)
    # 채널별 미세 색조 이동
    if rng.random() < 0.5:
        x = x + torch.tensor([rng.uniform(-0.1, 0.1) for _ in range(3)]).view(3, 1, 1)
    # 랜덤 가림 (cutout) — 간섭선/점에 대한 강인성. 글자를 통째로 지우지 않도록 작게
    for _ in range(rng.randint(0, 2)):
        bw = rng.randint(W // 16, W // 7)
        bh = rng.randint(H // 10, H // 3)
        x0 = rng.randint(0, max(1, W - bw))
        y0 = rng.randint(0, max(1, H - bh))
        x[:, y0:y0 + bh, x0:x0 + bw] = rng.uniform(-1, 1)
    return x.clamp(-1.5, 1.5)


class SynthCaptcha(Dataset):
    def __init__(self, length: int, seed=None, fonts=None, augment=True):
        self.length = length
        self._seed = seed
        self._fonts = fonts
        self._augment = augment
        self._renderer = None            # 워커별로 지연 생성
        self._arng = random.Random(seed)

    def _renderer_lazy(self):
        if self._renderer is None:
            seed = self._seed
            info = torch.utils.data.get_worker_info()
            if seed is not None and info is not None:
                seed = seed + info.id * 100003
            self._renderer = CaptchaRenderer(fonts=self._fonts, seed=seed)
        return self._renderer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        r = self._renderer_lazy()
        text = r.random_text()
        x = to_tensor(r.render(text))
        if self._augment:
            x = _augment(x, self._arng)
        y = torch.tensor(config.encode(text), dtype=torch.long)
        return x, y


def build_val(n: int, seed: int = 1234, fonts=None):
    """(x: n x 3 x H x W, y: n x NUM_POS) 텐서 쌍을 반환."""
    r = CaptchaRenderer(fonts=fonts, seed=seed)
    xs, ys = [], []
    for _ in range(n):
        text = r.random_text()
        xs.append(to_tensor(r.render(text)))
        ys.append(torch.tensor(config.encode(text), dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)
