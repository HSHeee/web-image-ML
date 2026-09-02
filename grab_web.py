"""웹페이지에 뜬 캡차 이미지를 캡처해서 바로 인식하는 예시.

Playwright 사용 (pip install playwright && playwright install chromium).
CSS 선택자로 캡차 <img> 또는 컨테이너 요소를 지정한다.

  python grab_web.py https://example.com/login  "img#captcha"
  python grab_web.py https://example.com/login  ".captcha-box"  --ckpt checkpoints/best.pt
"""

import argparse
import io

from PIL import Image

from predict import load_model, predict_bytes


def capture(url: str, selector: str, timeout_ms: int = 15000) -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        el = page.wait_for_selector(selector, timeout=timeout_ms)
        png = el.screenshot()          # 요소 영역만 캡처
        browser.close()
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("selector", help='캡차 요소 CSS 선택자 (예: "img#captcha")')
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--save", default="", help="캡처 이미지를 저장할 경로 (선택)")
    args = ap.parse_args()

    png = capture(args.url, args.selector)
    if args.save:
        Image.open(io.BytesIO(png)).save(args.save)

    model = load_model(args.ckpt)
    text, conf, per_char = predict_bytes(model, png)
    print(f"{text}\t(conf {conf:.3f})")


if __name__ == "__main__":
    main()
