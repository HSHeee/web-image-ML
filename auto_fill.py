"""접속 -> 요소 스크린샷 -> 캡차 인식 -> input 입력 -> 버튼 클릭.

서버 없음. 단일 스크립트. 모델은 이 프로세스 안에서 바로 구동.
본인 소유 사이트 / 자동화가 허용된 테스트 환경에서만 사용하세요.

설치:
    pip install playwright
    playwright install chromium

기본 사용:
    python auto_fill.py ^
      --url https://example.com/login ^
      --captcha "img#captcha" ^
      --input  "input[name=captcha]" ^
      --submit "button[type=submit]" ^
      --headed --confirm-submit

캡차가 깔끔한 요소가 아니면 좌표로 잘라서 캡처:
    --clip 320,180,150,50        # x,y,width,height  (--captcha 대신)

낮은 confidence 면 자동으로 새 캡차 요청하고 재시도:
    --refresh "a#reloadCaptcha" --min-conf 0.9 --max-attempts 5
"""

import argparse
import io

from PIL import Image

from predict import load_model, predict_image


def _shot(page, args):
    """캡차 이미지를 PIL 로 반환."""
    if args.clip:
        x, y, w, h = (float(v) for v in args.clip.split(","))
        png = page.screenshot(clip={"x": x, "y": y, "width": w, "height": h})
    else:
        el = page.wait_for_selector(args.captcha, timeout=args.timeout, state="visible")
        png = el.screenshot()
    return Image.open(io.BytesIO(png))


def run(args):
    from playwright.sync_api import sync_playwright

    model = load_model(args.ckpt)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        page = browser.new_context().new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout)
        if args.wait_ms:
            page.wait_for_timeout(args.wait_ms)

        solved = False
        for attempt in range(1, args.max_attempts + 1):
            img = _shot(page, args)
            text, conf, per_char = predict_image(model, img)
            print(f"[{attempt}/{args.max_attempts}] pred={text}  conf={conf:.3f}  "
                  f"per_char={[round(c, 2) for c in per_char]}")
            if args.save_shots:
                img.save(f"shot_{attempt:02d}_{text}.png")

            if conf < args.min_conf:
                if args.refresh and attempt < args.max_attempts:
                    print(f"  conf<{args.min_conf} -> 새 캡차 요청 후 재시도")
                    page.click(args.refresh)
                    page.wait_for_timeout(args.refresh_wait_ms)
                    continue
                print(f"  conf<{args.min_conf} -> 수동 확인 필요")
                if not args.force:
                    break

            box = page.wait_for_selector(args.input, timeout=args.timeout)
            box.fill("")
            box.type(text, delay=args.type_delay)

            if not args.submit:
                print("  입력 완료 (제출 안 함)")
                solved = True
                break

            if args.confirm_submit:
                input(f"  '{text}' 입력함. Enter 누르면 제출 (Ctrl+C 취소): ")
            page.click(args.submit)
            page.wait_for_timeout(args.after_submit_ms)
            print("  제출함")

            if not args.success_text:
                solved = True
                break
            if page.get_by_text(args.success_text).count() > 0:
                print(f"  성공 감지: '{args.success_text}'")
                solved = True
                break
            print("  성공 문구 미검출")
            if args.refresh and attempt < args.max_attempts:
                page.click(args.refresh)
                page.wait_for_timeout(args.refresh_wait_ms)

        print("결과:", "성공" if solved else "실패/미완료")
        if args.headed and args.hold:
            input("Enter 를 누르면 브라우저 종료: ")
        browser.close()
        return 0 if solved else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--captcha", help="캡차 이미지 요소 CSS 선택자")
    ap.add_argument("--clip", help="요소 대신 좌표로 캡처: x,y,width,height")
    ap.add_argument("--input", required=True, help="정답 입력란 CSS 선택자")
    ap.add_argument("--submit", default="", help="제출 버튼 선택자 (생략 시 입력만)")
    ap.add_argument("--refresh", default="", help="새 캡차 버튼 선택자 (재시도용)")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")

    ap.add_argument("--min-conf", type=float, default=0.90,
                    help="이 미만이면 재시도/중단")
    ap.add_argument("--max-attempts", type=int, default=1)
    ap.add_argument("--success-text", default="",
                    help="이 문구가 페이지에 나타나면 성공으로 판정")
    ap.add_argument("--force", action="store_true",
                    help="confidence 낮아도 그냥 입력 진행")

    ap.add_argument("--headed", action="store_true", help="브라우저 창 표시")
    ap.add_argument("--confirm-submit", action="store_true",
                    help="제출 직전 Enter 확인 (권장)")
    ap.add_argument("--hold", action="store_true", help="종료 전 대기 (headed)")
    ap.add_argument("--save-shots", action="store_true", help="캡처 이미지 저장")

    ap.add_argument("--timeout", type=int, default=15000)
    ap.add_argument("--wait-ms", type=int, default=0, help="goto 후 추가 대기")
    ap.add_argument("--type-delay", type=int, default=40, help="타이핑 간격 ms")
    ap.add_argument("--refresh-wait-ms", type=int, default=600)
    ap.add_argument("--after-submit-ms", type=int, default=1200)
    args = ap.parse_args()

    if not args.captcha and not args.clip:
        ap.error("--captcha 또는 --clip 중 하나는 필요합니다")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
