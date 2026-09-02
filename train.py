"""학습 스크립트.

예)
  python train.py                              # 기본값 (steps 4000, batch 128)
  python train.py --steps 12000 --workers 8    # 더 오래 / 더 빠르게
  python train.py --resume checkpoints/last.pt

체크포인트: checkpoints/last.pt, checkpoints/best.pt
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import SynthCaptcha, build_val
from model import LightCaptchaNet, param_count
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
            print(f"  [val] full-string {s_acc*100:.2f}%   per-char {c_acc*100:.2f}%")
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step,
                  "best": max(best, s_acc),
                  "meta": {"chars": config.CHARS, "num_pos": config.NUM_POS,
                           "img_h": config.IMG_H, "img_w": config.IMG_W}}
            torch.save(ck, os.path.join(args.out, "last.pt"))
            if s_acc >= best:
                best = s_acc
                torch.save(ck, os.path.join(args.out, "best.pt"))
                print(f"  -> saved best.pt ({best*100:.2f}%)")

        if step >= args.steps:
            break

    print(f"done. best full-string acc {best*100:.2f}%")


if __name__ == "__main__":
    main()
