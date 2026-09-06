"""합성 캡차 생성기.

실제 타깃 캡차 관찰 결과에 맞춰 튜닝:
- 영대문자 6글자, 넓고 균일한 간격, 좌우 여백 큼
- 거의 수직 (회전 ±5°), 균일한 크기, 캡차당 폰트 1종(산세리프)
- 캡차당 글자색 1개 (채도 높음), 배경은 단색의 어둡고 채도 높은 색
- 가로로 가로지르는 얇은 물결선 1개(가끔 2개)
- 촘촘한 연노랑 계열 1px 점 노이즈

미리보기:  python synth.py   ->  samples.png
"""

import colorsys
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import CHARS, IMG_H, IMG_W, NUM_POS

_WIN_FONTS = r"C:\Windows\Fonts"
# 실제 캡차와 비슷한 산세리프만
_FONT_CANDIDATES = [
    "arial.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf", "trebuc.ttf",
    "segoeui.ttf", "arialbd.ttf", "verdanab.ttf", "calibrib.ttf", "segoeuib.ttf",
]

RESAMPLE = Image.Resampling.LANCZOS
BICUBIC = Image.Resampling.BICUBIC


def discover_fonts(extra_dirs=None):
    """설치된 TTF 폰트 경로 목록을 반환한다."""
    dirs = [_WIN_FONTS] + list(extra_dirs or [])
    found = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in _FONT_CANDIDATES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                found.append(p)
    if not found:
        for d in dirs:
            found += sorted(glob.glob(os.path.join(d, "*.ttf")))[:12]
    seen, uniq = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


class CaptchaRenderer:
    def __init__(self, fonts=None, seed=None):
        self.fonts = fonts or discover_fonts()
        if not self.fonts:
            raise RuntimeError("TTF 폰트를 찾지 못했습니다. fonts=[...] 로 직접 지정하세요.")
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        self._font_cache = {}
        self._bg_h = 0.0
        self._text_rgb = (255, 255, 255)

    # ------------------------------------------------------------------ helpers
    def _font(self, path, size):
        key = (path, size)
        f = self._font_cache.get(key)
        if f is None:
            f = ImageFont.truetype(path, size)
            self._font_cache[key] = f
        return f

    @staticmethod
    def _hsv(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0), 1), min(max(v, 0), 1))
        return (int(r * 255), int(g * 255), int(b * 255))

    @staticmethod
    def _luma(c):
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

    def _contrast_color(self, c):
        return (20, 20, 20) if self._luma(c) > 128 else (235, 235, 235)

    def _pick_text_color(self, bg_rgb):
        r = self.rng
        for _ in range(8):
            h = (self._bg_h + r.uniform(0.25, 0.75)) % 1.0     # 배경과 다른 색상각
            c = self._hsv(h, r.uniform(0.4, 0.95), r.uniform(0.7, 1.0))
            if abs(self._luma(c) - self._luma(bg_rgb)) > 55:   # 최소 대비 보장
                return c
        return c

    # --------------------------------------------------------------- components
    def _make_background(self, W, H):
        r = self.rng
        self._bg_h = r.random()
        s = r.uniform(0.45, 0.9)
        v = r.uniform(0.22, 0.5)
        c1 = self._hsv(self._bg_h, s, v)
        self._bg_rgb = c1
        if r.random() < 0.15:                              # 드물게 그라디언트
            c2 = self._hsv(self._bg_h + r.uniform(-0.05, 0.05),
                           s * r.uniform(0.8, 1.1), v * r.uniform(0.8, 1.2))
            t = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
            base = (1 - t) * np.array(c1, np.float32) + t * np.array(c2, np.float32)
            base = np.broadcast_to(base, (H, W, 3)).astype(np.uint8)
            img = Image.fromarray(base, "RGB").convert("RGBA")
        else:
            img = Image.new("RGBA", (W, H), c1 + (255,))
        if r.random() < 0.7:                               # 배경 텍스처 노이즈
            a = np.asarray(img).astype(np.float32)
            a[..., :3] += self.nrng.normal(0, r.uniform(3, 12), a[..., :3].shape)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
        return img

    def _wave(self, img, amp_frac=(0.006, 0.02)):
        """세로 방향 사인파 (아주 약하게)."""
        r = self.rng
        a = np.asarray(img)
        H, W = a.shape[:2]
        amp = r.uniform(*amp_frac) * H
        freq = r.uniform(0.6, 1.8)
        phase = r.uniform(0, 2 * np.pi)
        shift = (amp * np.sin(2 * np.pi * freq * np.arange(W) / W + phase)).astype(int)
        rows = (np.arange(H)[:, None] - shift[None, :]) % H
        cols = np.arange(W)[None, :]
        return Image.fromarray(a[rows, cols])

    def _add_lines(self, img):
        """글자 밴드 한가운데를 관통하는 얇은 물결선 (반투명)."""
        r = self.rng
        W, H = img.size
        a = np.asarray(img).astype(np.float32)
        n = 1 if r.random() < 0.9 else (0 if r.random() < 0.5 else 2)
        xs = np.arange(W)
        for _ in range(n):
            y0 = r.uniform(0.40, 0.60) * H                 # 글자 세로 중앙 근처
            amp = r.uniform(0.015, 0.08) * H
            slope = r.uniform(-0.12, 0.12)
            k = r.randint(2, 4)
            kinks = np.interp(xs, np.linspace(0, W, k),
                              self.nrng.uniform(-1, 1, k)) * amp
            phase = r.uniform(0, 2 * np.pi)
            freq = r.uniform(0.4, 1.4)
            thick = r.choice([1, 1, 2])
            alpha = r.uniform(0.55, 0.9)
            col = np.array(self._hsv(r.random(), r.uniform(0.15, 0.6),
                                     r.uniform(0.4, 0.8)), np.float32)
            ys = (y0 + slope * (xs - W / 2)
                  + amp * np.sin(2 * np.pi * freq * xs / W + phase) + kinks).astype(int)
            for t in range(-thick, thick + 1):
                yy = np.clip(ys + t, 0, H - 1)
                a[yy, xs, :3] = (1 - alpha) * a[yy, xs, :3] + alpha * col
        img.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))

    def _add_speckles(self, img):
        """점 노이즈. 색은 글자색 계열(밝기 지터), 1~3px 덩어리."""
        r = self.rng
        a = np.asarray(img).copy()
        H, W = a.shape[:2]
        n = int(W * H * r.uniform(0.0025, 0.008))
        ys = self.nrng.integers(0, H, n)
        xs = self.nrng.integers(0, W, n)
        base = np.array(self._text_rgb, np.float32)
        jit = self.nrng.uniform(0.55, 1.15, (n, 1))       # 밝기만 흔들기
        cols = np.clip(base[None, :] * jit, 0, 255).astype(np.uint8)
        if r.random() < 0.3:                               # 일부는 배경 밝은 톤
            k = n // 4
            cols[:k] = np.clip(self.nrng.uniform(150, 255, (k, 3)), 0, 255)
        a[ys, xs, :3] = cols
        for dy, dx in [(1, 0), (0, 1), (1, 1)]:            # 덩어리로 번지게
            m = self.nrng.random(n) < 0.35
            a[np.clip(ys[m] + dy, 0, H - 1), np.clip(xs[m] + dx, 0, W - 1), :3] = cols[m]
        img.paste(Image.fromarray(a))

    def _photo_noise(self, img):
        if self.rng.random() < 0.5:
            return img
        a = np.asarray(img).astype(np.float32)
        a += self.nrng.normal(0, self.rng.uniform(3, 10), a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    # -------------------------------------------------------------------- public
    def random_text(self, n=NUM_POS):
        return "".join(self.rng.choice(CHARS) for _ in range(n))

    def render(self, text, out_size=(IMG_W, IMG_H)):
        r = self.rng
        W0, H0 = out_size
        ss = 2
        W, H = W0 * ss, H0 * ss

        img = self._make_background(W, H)
        self._text_rgb = self._pick_text_color(self._bg_rgb)

        n = len(text)
        # 글자가 차지하는 전체 폭과 시작점을 변동 (위치가 k/6 에 고정되지 않게)
        spread = r.uniform(0.72, 0.92) * W
        start = r.uniform(0.04, 1.0 - 0.04 - spread / W) * W
        slot = spread / n
        fs = int(H * r.uniform(0.40, 0.56))               # 실제는 글자가 작음
        font_path = r.choice(self.fonts)                   # 캡차당 폰트 1개
        stroke_w = ss if r.random() < 0.10 else 0
        stroke_fill = self._contrast_color(self._text_rgb)
        xscale = r.uniform(0.82, 1.05)                     # 살짝 폭 좁게 (tall-narrow)
        jit = r.uniform(0.06, 0.16)                        # 자리별 흔들림 크기

        for i, ch in enumerate(text):
            font = self._font(font_path, max(10, int(fs * r.uniform(0.94, 1.06))))
            layer = Image.new("RGBA", (int(slot * 2.6), H), (0, 0, 0, 0))
            cd = ImageDraw.Draw(layer)
            bbox = cd.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = (layer.width - tw) / 2 - bbox[0]
            cy = (H - th) / 2 - bbox[1] + r.uniform(-H * 0.09, H * 0.09)
            cd.text((cx, cy), ch, font=font, fill=self._text_rgb + (255,),
                    stroke_width=stroke_w, stroke_fill=stroke_fill + (255,))
            if xscale < 0.99:
                layer = layer.resize((max(1, int(layer.width * xscale)), layer.height),
                                     RESAMPLE)
            layer = layer.rotate(r.uniform(-4, 4), resample=BICUBIC, expand=True)
            cpos = start + slot * (i + 0.5) + r.uniform(-slot * jit, slot * jit)
            px = int(cpos - layer.width / 2)
            py = int((H - layer.height) / 2)
            img.alpha_composite(layer, (px, py))

        if r.random() < 0.30:
            img = self._wave(img)
        self._add_lines(img)                               # 선은 글자 위에
        self._add_speckles(img)

        img = img.convert("RGB").resize(out_size, RESAMPLE)
        if r.random() < 0.85:                              # 실제처럼 부드러운 엣지
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.4, 1.1)))
        return self._photo_noise(img)


if __name__ == "__main__":
    renderer = CaptchaRenderer(seed=0)
    cols, rows, pad = 6, 8, 6
    grid = Image.new("RGB",
                     (cols * (IMG_W + pad) + pad, rows * (IMG_H + pad) + pad),
                     (255, 255, 255))
    for i in range(cols * rows):
        text = renderer.random_text()
        im = renderer.render(text)
        gx = pad + (i % cols) * (IMG_W + pad)
        gy = pad + (i // cols) * (IMG_H + pad)
        grid.paste(im, (gx, gy))
    grid.save("samples.png")
    print("wrote samples.png  (6x8 grid)")
