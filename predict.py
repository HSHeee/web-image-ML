"""추론: 이미지 -> 6글자 문자열.

CLI
  python predict.py img1.png img2.png
  python predict.py --show-conf img.png
  type img.png | python predict.py            # stdin 으로 바이트 1장

코드에서
  from predict import load_model, predict_image, predict_bytes
  model = load_model("checkpoints/best.pt")
  text, conf, per_char = predict_image(model, PIL_image)
"""

import argparse
import io
import sys

import torch
from PIL import Image

import config
from model import LightCaptchaNet
from preprocess import to_tensor


def load_model(ckpt_path="checkpoints/best.pt", device="cpu"):
    model = LightCaptchaNet()
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_image(model, img: Image.Image, device="cpu"):
    """returns (text, mean_confidence, [per_char_confidence])"""
    x = to_tensor(img).unsqueeze(0).to(device)
    probs = model(x)[0].softmax(-1)                 # P x n_cls
    conf, idx = probs.max(-1)
    return config.decode(idx.tolist()), conf.mean().item(), conf.tolist()


def predict_bytes(model, data: bytes, device="cpu"):
    return predict_image(model, Image.open(io.BytesIO(data)), device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="이미지 파일들 (생략 시 stdin 1장)")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show-conf", action="store_true")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)

    if not args.images:
        text, mconf, cc = predict_bytes(model, sys.stdin.buffer.read(), args.device)
        print(f"{text}\t{mconf:.3f}" if args.show_conf else text)
        return

    for path in args.images:
        text, mconf, cc = predict_image(model, Image.open(path), args.device)
        if args.show_conf:
            print(f"{path}\t{text}\t{mconf:.3f}\t{[round(c, 2) for c in cc]}")
        else:
            print(text)


if __name__ == "__main__":
    main()
