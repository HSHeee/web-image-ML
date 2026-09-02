"""합성 캡차 Dataset.

- SynthCaptcha : 매 __getitem__ 마다 새 캡차를 즉석 생성 (디스크 불필요, 사실상 무한).
- build_val    : 재현 가능한 고정 검증셋을 메모리에 만든다.
"""

import torch
from torch.utils.data import Dataset

import config
from preprocess import to_tensor
from synth import CaptchaRenderer


class SynthCaptcha(Dataset):
    def __init__(self, length: int, seed=None, fonts=None):
        self.length = length
        self._seed = seed
        self._fonts = fonts
        self._renderer = None            # 워커별로 지연 생성

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
