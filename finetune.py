"""실제 캡차 소량으로 파인튜닝.

합성으로 사전학습된 checkpoints/best.pt 에서 출발해, 실제 이미지(파일명=정답)를
합성과 섞어 낮은 LR 로 짧게 학습한다. 실제 일부는 홀드아웃해서 검증에만 쓴다.

  python finetune.py --steps 800 --real-ratio 0.4
  python finetune.py --from checkpoints/best.pt --out checkpoints/finetuned.pt

결과: checkpoints/finetuned.pt  (홀드아웃 per-char 기준 best)
"""

import argparse
import glob
import os
import random

import torch
import torch.nn as nn
from PIL import Image

import config
from dataset import SynthCaptcha, _augment
from model import LightCaptchaNet, param_count
from preprocess import to_tensor


def load_real_split(folder, holdout_every=4):
    files = sorted(glob.glob(os.path.join(folder, "*.png"))
                   + glob.glob(os.path.join(folder, "*.jpg")))
    items = []
    for f in files:
        lab = os.path.splitext(os.path.basename(f))[0].strip().upper()
        if len(lab) == config.NUM_POS and all(c in config.CHARS for c in lab):
            items.append((to_tensor(Image.open(f)),
                          torch.tensor(config.encode(lab), dtype=torch.long)))
    hold = items[::holdout_every]
    train = [it for i, it in enumerate(items) if i % holdout_every != 0]
    return train, hold


@torch.no_grad()
def eval_set(model, items, device):
    if not items:
        return 0.0, 0.0
    x = torch.stack([a for a, _ in items]).to(device)
    y = torch.stack([b for _, b in items])
    pred = model(x).argmax(-1).cpu()
    return ((pred == y).all(1).float().mean().item(),
            (pred == y).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="checkpoints/best.pt")
    ap.add_argument("--out", default="checkpoints/finetuned.pt")
    ap.add_argument("--real-dir", default="Screenshot")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--real-ratio", type=float, default=0.4,
                    help="배치에서 실제 이미지 비율 (나머지는 합성)")
    ap.add_argument("--holdout-every", type=int, default=4,
                    help="정렬된 실제 이미지 중 N장마다 1장을 검증용으로 제외")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    real_train, real_hold = load_real_split(args.real_dir, args.holdout_every)
    print(f"real: train {len(real_train)}  holdout {len(real_hold)}")
    if len(real_train) < 4:
        raise SystemExit("실제 학습 이미지가 너무 적습니다.")

    rng = random.Random(args.seed)
    arng = random.Random(args.seed + 1)

    n_real = max(1, round(args.batch_size * args.real_ratio))
    n_syn = args.batch_size - n_real
    syn = SynthCaptcha(length=args.steps * n_syn, seed=None, augment=True)
    syn_dl = iter(torch.utils.data.DataLoader(
        syn, batch_size=n_syn, num_workers=args.workers, drop_last=True,
        persistent_workers=args.workers > 0))

    model = LightCaptchaNet().to(device)
    ck = torch.load(args.src, map_location=device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    print(f"loaded {args.src}  params {param_count(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    crit = nn.CrossEntropyLoss()

    f0, c0 = eval_set(model, real_hold, device)
    print(f"start  holdout full {f0*100:.1f}%  char {c0*100:.1f}%")

    best = c0
    model.train()
    for step in range(1, args.steps + 1):
        xs, ys = next(syn_dl)
        # 실제 배치 구성: 복원추출 + 텐서 augmentation
        rb = [random.choice(real_train) for _ in range(n_real)]
        rx = torch.stack([_augment(a.clone(), arng) for a, _ in rb])
        ry = torch.stack([b for _, b in rb])
        xb = torch.cat([xs, rx]).to(device)
        yb = torch.cat([ys, ry]).to(device)

        out = model(xb)
        loss = crit(out.reshape(-1, config.NUM_CLASSES), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % 100 == 0 or step == args.steps:
            fh, chh = eval_set(model, real_hold, device)
            model.train()
            print(f"step {step:>4}/{args.steps}  loss {loss.item():.3f}  "
                  f"holdout full {fh*100:.1f}%  char {chh*100:.1f}%")
            if chh >= best:
                best = chh
                torch.save({"model": model.state_dict(),
                            "meta": {"chars": config.CHARS, "num_pos": config.NUM_POS,
                                     "img_h": config.IMG_H, "img_w": config.IMG_W}},
                           args.out)
                print(f"  -> saved {args.out} (holdout char {best*100:.1f}%)")

    print(f"done. best holdout char {best*100:.1f}%  ({args.out})")


if __name__ == "__main__":
    main()
