"""학습된 체크포인트를 ONNX 로 내보낸다.

  python export_onnx.py --ckpt checkpoints/best.pt --out captcha.onnx

배포 추론은 onnx_infer.py 참고 (onnxruntime, CPU 수 ms).
"""

import argparse

import torch

import config
from model import LightCaptchaNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="captcha.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = LightCaptchaNet()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    dummy = torch.randn(1, 3, config.IMG_H, config.IMG_W)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,          # 레거시 exporter: 단순 정적 그래프, Windows 콘솔 인코딩 이슈 회피
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
