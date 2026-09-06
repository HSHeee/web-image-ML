"""학습 스크립트.

예)
  python train.py                              # 기본값 (steps 4000, batch 128)
  python train.py --steps 12000 --workers 8    # 더 오래 / 더 빠르게
  python train.py --resume checkpoints/last.pt

체크포인트: checkpoints/last.pt, checkpoints/best.pt
"""

import argparse
import glob
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

import config
from dataset import SynthCaptcha, build_val
from model import LightCaptchaNet, param_count
from preprocess import to_tensor
from synth import discover_fonts


@torch.no_grad()
def evaluate(model, x_val, y_val, device, batch=256):
    model.eval()
    n = x_val.size(0)
    ok_str = ok_ch = 0
    for i in range(0, n, batch):
        xb = x_val[i:i + batch].to(device)
        yb = y_val[i:i + batch].to(device)
        pred = model(xb).argmax(-1)                 # B x P
        ok_ch += (pred == yb).sum().item()
        ok_str += (pred == yb).all(dim=1).sum().item()
    model.train()
    return ok_str / n, ok_ch / (n * config.NUM_POS)


def load_real(real_dir):
    """파일명이 정답인 실제 캡차 폴더 -> (x, y, names). 없으면 None."""
    files = sorted(glob.glob(os.path.join(real_dir, "*.png"))
                   + glob.glob(os.path.join(real_dir, "*.jpg")))
    xs, ys, names = [], [], []
    for f in files:
        label = os.path.splitext(os.path.basename(f))[0].strip().upper()
        if len(label) != config.NUM_POS or any(c not in config.CHARS for c in label):
            continue
        xs.append(to_tensor(Image.open(f)))
        ys.append(torch.tensor(config.encode(label), dtype=torch.long))
        names.append(label)
    if not xs:
        return None
    return torch.stack(xs), torch.stack(ys), names


@torch.no_grad()
def evaluate_real(model, x, y, device):
    model.eval()
    pred = model(x.to(device)).argmax(-1).cpu()
    model.train()
    ok_str = (pred == y).all(dim=1).sum().item()
    ok_ch = (pred == y).sum().item()
    return ok_str / len(y), ok_ch / (len(y) * config.NUM_POS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", default="")
    ap.add_argument("--fonts-dir", default="", help="추가 폰트 폴더 (선택)")
    ap.add_argument("--real-dir", default="Screenshot",
                    help="파일명=정답 인 실제 캡차 폴더. val 시점마다 정확도 같이 출력")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6,
                    help="CPU 연산 스레드 수. DataLoader 워커와 코어를 나눠 쓰도록 제한 (0=자동)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    fonts = discover_fonts([args.fonts_dir]) if args.fonts_dir else None
    print(f"device={device}  fonts={len(fonts) if fonts else 'auto'}  "
          f"steps={args.steps}  batch={args.batch_size}")

    train_ds = SynthCaptcha(length=args.steps * args.batch_size, seed=None, fonts=fonts)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, drop_last=True,
        persistent_workers=args.workers > 0, pin_memory=(device == "cuda"),
    )

    print(f"building validation set ({args.val_size}) ...")
    x_val, y_val = build_val(args.val_size, seed=1234, fonts=fonts)

    real = load_real(args.real_dir) if args.real_dir else None
    if real:
        rx, ry, _ = real
        rx = rx.to(device)
        print(f"real eval set: {len(ry)} images from {args.real_dir}/")
    else:
        print(f"real eval set: (없음 - {args.real_dir}/ 에 파일명=정답 png 두면 활성화)")

    model = LightCaptchaNet().to(device)
    print(f"params: {param_count(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    crit = nn.CrossEntropyLoss()

    step, best = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best = ck.get("step", 0), ck.get("best", 0.0)
        print(f"resumed from {args.resume} @ step {step} (best {best*100:.2f}%)")

    model.train()
    running, t0 = 0.0, time.time()
    for xb, yb in train_dl:
        step += 1
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)                                   # B x P x n_cls
        loss = crit(out.reshape(-1, config.NUM_CLASSES), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if step % 50 == 0:
            ips = 50 * args.batch_size / (time.time() - t0)
            print(f"step {step:>6}/{args.steps}  loss {running/50:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {ips:.0f} img/s")
            running, t0 = 0.0, time.time()

        if step % args.val_every == 0 or step >= args.steps:
            s_acc, c_acc = evaluate(model, x_val, y_val, device)
            line = f"  [synth] full {s_acc*100:.2f}%  char {c_acc*100:.2f}%"
            if real:
                r_str, r_ch = evaluate_real(model, rx, ry, device)
                line += f"   [real] full {r_str*100:.1f}%  char {r_ch*100:.1f}%"
                score = r_ch                       # 실제 per-char 로 best 선정
            else:
                score = s_acc
            print(line)
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step,
                  "best": max(best, score),
                  "meta": {"chars": config.CHARS, "num_pos": config.NUM_POS,
                           "img_h": config.IMG_H, "img_w": config.IMG_W}}
            torch.save(ck, os.path.join(args.out, "last.pt"))
            if score >= best:
                best = score
                torch.save(ck, os.path.join(args.out, "best.pt"))
                print(f"  -> saved best.pt (score {best*100:.2f}%)")

        if step >= args.steps:
            break

    print(f"done. best full-string acc {best*100:.2f}%")


if __name__ == "__main__":
    main()
