"""합성 캡차 생성기.

웹페이지에 뜨는 캡차(컬러 배경 + 스페클 노이즈 + 회전된 영대문자 6글자)를
흉내 내는 이미지를 무한히 생성한다. 파라미터를 넓게 랜덤화해서 실제 스타일이
조금 달라도 견디도록 한다.

미리보기:  python synth.py   ->  samples.png
"""

import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import CHARS, IMG_H, IMG_W, NUM_POS

_WIN_FONTS = r"C:\Windows\Fonts"
_FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "ariblk.ttf", "arialn.ttf",
    "calibri.ttf", "calibrib.ttf", "verdana.ttf", "verdanab.ttf",
    "tahoma.ttf", "tahomabd.ttf", "trebuc.ttf", "trebucbd.ttf",
    "georgia.ttf", "georgiab.ttf", "times.ttf", "timesbd.ttf",
    "cour.ttf", "courbd.ttf", "comic.ttf", "comicbd.ttf",
    "impact.ttf", "segoeui.ttf", "segoeuib.ttf",
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
    # 후보를 못 찾으면 아무 ttf나 긁어온다
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

    # ------------------------------------------------------------------ helpers
    def _font(self, path, size):
        key = (path, size)
        f = self._font_cache.get(key)
        if f is None:
            f = ImageFont.truetype(path, size)
            self._font_cache[key] = f
        return f

    def _rand_col(self, lo=0, hi=255):
        r = self.rng
        return (r.randint(lo, hi), r.randint(lo, hi), r.randint(lo, hi))

    def _text_color(self):
        """배경과 대비되도록 밝거나 / 어둡거나 / 채도 높은 색."""
        r = self.rng
        mode = r.random()
        if mode < 0.45:                       # 밝은 글자
            v = r.randint(205, 255)
            return (v - r.randint(0, 45), v - r.randint(0, 45), v - r.randint(0, 45))
        if mode < 0.8:                        # 어두운 글자
            v = r.randint(0, 55)
            return (v + r.randint(0, 45), v + r.randint(0, 45), v + r.randint(0, 45))
        return self._rand_col(0, 255)         # 컬러 글자

    @staticmethod
    def _luma(c):
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]

    def _contrast_color(self, c):
        return (20, 20, 20) if self._luma(c) > 128 else (235, 235, 235)

    # --------------------------------------------------------------- components
    def _make_background(self, W, H):
        r = self.rng
        mode = r.choice(["solid", "solid", "vgrad", "hgrad"])
        c1 = self._rand_col(35, 215)
        if mode == "solid":
            img = Image.new("RGBA", (W, H), c1 + (255,))
        else:
            c2 = self._rand_col(35, 215)
            if mode == "vgrad":
                t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
            else:
                t = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
            base = (1 - t) * np.array(c1, np.float32) + t * np.array(c2, np.float32)
            base = np.broadcast_to(base, (H, W, 3)).astype(np.uint8)
            img = Image.fromarray(base, "RGB").convert("RGBA")
        if r.random() < 0.6:                  # 배경 텍스처 노이즈
            a = np.asarray(img).astype(np.float32)
            a[..., :3] += self.nrng.normal(0, r.uniform(4, 18), a[..., :3].shape)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
        return img

    def _wave(self, img):
        """세로 방향 사인파로 글자를 물결지게 만든다 (벡터화)."""
        r = self.rng
        a = np.asarray(img)
        H, W = a.shape[:2]
        amp = r.uniform(H * 0.012, H * 0.05)
        freq = r.uniform(0.6, 2.2)
        phase = r.uniform(0, 2 * np.pi)
        shift = (amp * np.sin(2 * np.pi * freq * np.arange(W) / W + phase)).astype(int)
        rows = (np.arange(H)[:, None] - shift[None, :]) % H     # (H, W)
        cols = np.arange(W)[None, :]
        return Image.fromarray(a[rows, cols])

    def _add_lines(self, img):
        r = self.rng
        if r.random() < 0.35:                 # 상당수는 간섭선 없음 (타깃과 유사)
            return
        draw = ImageDraw.Draw(img)
        W, H = img.size
        for _ in range(r.randint(1, 2)):
            pts = [(r.randint(0, W), r.randint(0, H)) for _ in range(r.randint(2, 4))]
            draw.line(pts, fill=self._rand_col(0, 255) + (r.randint(70, 160),),
                      width=r.choice([1, 2, 3]))

    def _add_speckles(self, img):
        """무작위 컬러 점 노이즈 (numpy 로 한 번에 찍는다)."""
        r = self.rng
        a = np.asarray(img).copy()
        H, W = a.shape[:2]
        density = r.uniform(0.0006, 0.0026)
        n = int(W * H * density)
        ys = self.nrng.integers(0, H, n)
        xs = self.nrng.integers(0, W, n)
        if r.random() < 0.5:                                   # 팔레트 제한
            pal = np.array([self._rand_col(0, 255) for _ in range(r.randint(2, 4))])
            cols = pal[self.nrng.integers(0, len(pal), n)]
        else:
            cols = self.nrng.integers(0, 256, (n, 3))
        a[ys, xs, :3] = cols
        # 점 일부를 2px 로 (아래/오른쪽 픽셀도 칠함)
        big = self.nrng.random(n) < 0.4
        a[np.clip(ys[big] + 1, 0, H - 1), xs[big], :3] = cols[big]
        a[ys[big], np.clip(xs[big] + 1, 0, W - 1), :3] = cols[big]
        img.paste(Image.fromarray(a))

    def _photo_noise(self, img):
        if self.rng.random() < 0.5:
            return img
        a = np.asarray(img).astype(np.float32)
        a += self.nrng.normal(0, self.rng.uniform(3, 12), a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    # -------------------------------------------------------------------- public
    def random_text(self, n=NUM_POS):
        return "".join(self.rng.choice(CHARS) for _ in range(n))

    def render(self, text, out_size=(IMG_W, IMG_H)):
        r = self.rng
        W0, H0 = out_size
        ss = 2                                # 슈퍼샘플링 후 축소 -> 회전 계단현상 감소
        W, H = W0 * ss, H0 * ss

        img = self._make_background(W, H)

        n = len(text)
        margin = int(W * 0.04)
        avail = W - 2 * margin
        slot = avail / n
        font_size = int(H * r.uniform(0.58, 0.76))

        per_char_color = r.random() < 0.18
        base_color = self._text_color()
        stroke_w = r.choice([1, 2]) * ss if r.random() < 0.3 else 0
        stroke_fill = self._contrast_color(base_color)

        x = margin
        for ch in text:
            font = self._font(r.choice(self.fonts),
                              max(10, int(font_size * r.uniform(0.9, 1.1))))
            layer = Image.new("RGBA", (int(slot * 2.4), H), (0, 0, 0, 0))
            cd = ImageDraw.Draw(layer)
            color = self._text_color() if per_char_color else base_color
            bbox = cd.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = (layer.width - tw) / 2 - bbox[0]
            cy = (H - th) / 2 - bbox[1] + r.uniform(-H * 0.07, H * 0.07)
            cd.text((cx, cy), ch, font=font, fill=color + (255,),
                    stroke_width=stroke_w, stroke_fill=stroke_fill + (255,))
            layer = layer.rotate(r.uniform(-20, 20), resample=BICUBIC, expand=True)
            px = int(x - (layer.width - slot) / 2 + r.uniform(-slot * 0.08, slot * 0.08))
            py = int((H - layer.height) / 2)
            img.alpha_composite(layer, (max(px, -layer.width + 1), py))
            x += slot * r.uniform(0.82, 1.02)      # 살짝 겹치기 허용

        if r.random() < 0.7:
            img = self._wave(img)
        self._add_lines(img)
        self._add_speckles(img)

        img = img.convert("RGB").resize(out_size, RESAMPLE)
        if r.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.9)))
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
