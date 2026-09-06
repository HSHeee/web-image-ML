"""학습된 체크포인트를 단일 파일 ONNX 로 내보낸다.

  python export_onnx.py --ckpt checkpoints/finetuned.pt --out captcha.onnx

배포 추론은 onnx_infer.py 참고 (onnxruntime, CPU 수 ms).
"""

import argparse
import os
import tempfile

import onnx
import torch

import config
from model import LightCaptchaNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/finetuned.pt")
    ap.add_argument("--out", default="captcha.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    model = LightCaptchaNet()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    dummy = torch.randn(1, 3, config.IMG_H, config.IMG_W)
    # dynamo exporter: 새 헤드의 adaptive_avg_pool 업샘플을 지원. 가중치는 .data 로 분리됨
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "m.onnx")
        torch.onnx.export(
            model, dummy, tmp,
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset, dynamo=True,
        )
        m = onnx.load(tmp)                       # 외부 가중치까지 메모리로 로드
    onnx.save(m, args.out, save_as_external_data=False)   # 단일 파일로 저장
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
