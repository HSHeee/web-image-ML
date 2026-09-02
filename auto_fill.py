"""접속 -> 요소 스크린샷 -> 캡차 인식 -> input 입력 -> 버튼 클릭.

서버 없음. 단일 스크립트. 모델은 이 프로세스 안에서 바로 구동.
본인 소유 사이트 / 자동화가 허용된 테스트 환경에서만 사용하세요.

설치:
    pip install playwright
    playwright install chromium

기본 사용 (PowerShell, 한 줄):
    python auto_fill.py --url https://example.com/login --captcha "img[alt='캡챠 이미지']" --from-src --input "input[placeholder*='문자를 입력']" --submit "text=입력완료" --headed --confirm-submit

옵션:
    --from-src                  <img src>(data URI 등)를 직접 디코딩 (스크린샷 대신, 더 깨끗)
    --clip 320,180,150,50       요소 대신 좌표로 캡처: x,y,width,height (--captcha 대신)
    --pre-click "text=시작하기"   캡차가 뜨기 전 눌러야 할 요소 (여러 번 지정 가능, 순서대로)
    --date "2026년 10월 3일"      시작 전 날짜 선택 클릭. 생략하면 건너뜀 (on/off)
    --refresh "button[aria-label='새 문자 불러오기']" --min-conf 0.9 --max-attempts 5
    --submit 생략               입력까지만 하고 멈춤 (처음 테스트할 때 권장)
    --save-shots                캡처한 캡차를 shot_XX_예측.png 로 저장

브라우저 선택:
    --browser whale             네이버 웨일로 실행 (설치 경로 자동 탐색)
    --browser chrome            설치된 Chrome / --browser msedge  Edge
    --browser-path "C:\\...\\whale.exe"   실행파일 직접 지정
    --user-data-dir "C:\\Users\\me\\pw-profile"   로그인 유지되는 별도 프로필
    --cdp http://localhost:9222 이미 로그인해 띄워둔 브라우저에 붙기
                               (웨일을 --remote-debugging-port=9222 로 직접 실행 후)
"""

import argparse
import base64
import io
import os

from PIL import Image

from predict import load_model, predict_image

_WHALE_CANDIDATES = [
    r"C:\Program Files\Naver\Naver Whale\Application\whale.exe",
    r"C:\Program Files (x86)\Naver\Naver Whale\Application\whale.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Naver\Naver Whale\Application\whale.exe"),
]


def _find_whale():
    for c in _WHALE_CANDIDATES:
        if os.path.isfile(c):
            return c
    raise SystemExit("whale.exe 를 못 찾음. --browser-path 로 직접 지정하세요.")


def _open_browser(p, args):
    """(page, close_fn) 반환. --browser / --browser-path / --user-data-dir / --cdp 처리."""
    kw = {}
    if args.browser_path:
        kw["executable_path"] = args.browser_path
    elif args.browser == "chrome":
        kw["channel"] = "chrome"
    elif args.browser == "msedge":
        kw["channel"] = "msedge"
    elif args.browser == "whale":
        kw["executable_path"] = _find_whale()

    if args.cdp:                                    # 이미 떠 있는 브라우저에 붙기
        browser = p.chromium.connect_over_cdp(args.cdp)
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        return ctx.new_page(), browser.close

    if args.user_data_dir:                          # 로그인 유지되는 영구 프로필
        ctx = p.chromium.launch_persistent_context(
            args.user_data_dir, headless=not args.headed, **kw)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        return page, ctx.close

    browser = p.chromium.launch(headless=not args.headed, **kw)
    return browser.new_context().new_page(), browser.close


def _grab(page, args):
    """캡차 이미지를 PIL 로 반환."""
    if args.clip:
        x, y, w, h = (float(v) for v in args.clip.split(","))
        png = page.screenshot(clip={"x": x, "y": y, "width": w, "height": h})
        return Image.open(io.BytesIO(png))

    el = page.wait_for_selector(args.captcha, timeout=args.timeout, state="visible")

    if args.from_src:                       # <img src> 를 직접 디코딩 (렌더링/스케일 없음)
        src = el.get_attribute("src") or ""
        if src.startswith("data:"):
            data = base64.b64decode(src.split(",", 1)[1])
        else:
            data = page.request.get(src).body()
        return Image.open(io.BytesIO(data))

    return Image.open(io.BytesIO(el.screenshot()))


def run(args):
    from playwright.sync_api import sync_playwright

    model = load_model(args.ckpt)

    with sync_playwright() as p:
        page, close_browser = _open_browser(p, args)
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout)

        steps = list(args.pre_click)        # 캡차가 뜨기 전 눌러야 할 것들 (순서대로)
        if args.date:                       # 날짜 선택 on/off: 값 있으면 맨 앞에 끼움
            steps.insert(0, f"abbr[aria-label='{args.date}']")
        for sel in steps:
            print(f"pre-click: {sel}")
            page.click(sel, timeout=args.timeout)
            page.wait_for_timeout(400)
        if args.wait_ms:
            page.wait_for_timeout(args.wait_ms)

        solved = False
        for attempt in range(1, args.max_attempts + 1):
            img = _grab(page, args)
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
        if args.hold:
            input("Enter 를 누르면 종료: ")
        if not args.cdp:                    # 붙은 브라우저는 닫지 않음
            close_browser()
        return 0 if solved else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--captcha", help="캡차 이미지 요소 CSS 선택자")
    ap.add_argument("--clip", help="요소 대신 좌표로 캡처: x,y,width,height")
    ap.add_argument("--from-src", action="store_true",
                    help="스크린샷 대신 <img src>(data URI 등)를 직접 디코딩")
    ap.add_argument("--input", required=True, help="정답 입력란 CSS 선택자")
    ap.add_argument("--submit", default="", help="제출 버튼 선택자 (생략 시 입력만)")
    ap.add_argument("--refresh", default="", help="새 캡차 버튼 선택자 (재시도용)")
    ap.add_argument("--pre-click", action="append", default=[], metavar="SELECTOR",
                    help="캡차가 뜨기 전 눌러야 할 요소 (반복 지정 가능, 준 순서대로)")
    ap.add_argument("--date", default="", metavar="ARIA_LABEL",
                    help="시작 전 날짜 선택 클릭. 예: \"2026년 10월 3일\". "
                         "생략하면 이 단계를 건너뜀 (on/off)")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")

    ap.add_argument("--browser", default="chromium",
                    choices=["chromium", "chrome", "msedge", "whale"],
                    help="chromium=번들(기본), chrome/msedge=설치본, whale=네이버 웨일")
    ap.add_argument("--browser-path", default="",
                    help="브라우저 실행파일 직접 지정 (whale.exe 등). --browser 보다 우선")
    ap.add_argument("--user-data-dir", default="",
                    help="영구 프로필 폴더. 한 번 로그인하면 이후 유지됨 "
                         "(그 프로필이 다른 창에서 열려 있으면 안 됨)")
    ap.add_argument("--cdp", default="",
                    help="이미 떠 있는 브라우저에 붙기. 예: http://localhost:9222 "
                         "(브라우저를 --remote-debugging-port=9222 로 직접 실행해 둘 것)")

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
