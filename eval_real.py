"""실제 캡차 폴더로 현재 모델 정확도 측정. 파일명이 정답이어야 함.

  python eval_real.py                    # Screenshot/ 폴더
  python eval_real.py 다른폴더 --show
"""

import argparse
import glob
import os

from PIL import Image

import config
from predict import load_model, predict_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="Screenshot")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--show", action="store_true", help="한 장씩 출력")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    files = sorted(glob.glob(os.path.join(args.folder, "*.png"))
                   + glob.glob(os.path.join(args.folder, "*.jpg")))
    ok = ok_ch = tot_ch = 0
    miss = []
    for f in files:
        true = os.path.splitext(os.path.basename(f))[0].strip().upper()
        if len(true) != config.NUM_POS:
            continue
        pred, conf, _ = predict_image(model, Image.open(f))
        cc = sum(a == b for a, b in zip(pred, true))
        ok += pred == true
        ok_ch += cc
        tot_ch += config.NUM_POS
        if pred != true:
            miss.append((true, pred))
        if args.show:
            mark = "OK" if pred == true else "  "
            print(f"{mark} {true} -> {pred}  ({cc}/6, conf {conf:.2f})")

    n = tot_ch // config.NUM_POS
    print(f"\nfull-string {ok}/{n} ({ok/n*100:.1f}%)   "
          f"per-char {ok_ch}/{tot_ch} ({ok_ch/tot_ch*100:.1f}%)")
    if miss:
        print("오답:", ", ".join(f"{t}->{p}" for t, p in miss))


if __name__ == "__main__":
    main()
