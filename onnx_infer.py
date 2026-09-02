"""ONNX Runtime 로 추론 (torch 불필요, CPU 수 ms).

  python onnx_infer.py img1.png img2.png --model captcha.onnx
"""

import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image

import config

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(path):
    img = (Image.open(path).convert("RGB")
           .resize((config.IMG_W, config.IMG_H), Image.Resampling.BILINEAR))
    a = (np.asarray(img, np.float32) / 255.0 - _MEAN) / _STD
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="captcha.onnx")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    for path in args.images:
        logits = sess.run(None, {"image": preprocess(path)})[0][0]   # P x n_cls
        e = np.exp(logits - logits.max(-1, keepdims=True))
        probs = e / e.sum(-1, keepdims=True)
        idx = probs.argmax(-1)
        conf = probs.max(-1).mean()
        print(f"{path}\t{config.decode(idx.tolist())}\t{conf:.3f}")


if __name__ == "__main__":
    main()
