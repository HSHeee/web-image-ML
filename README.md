# captcha_ocr — 경량 캡차(6글자 영대문자) 인식기

웹페이지에 뜨는 `MSCQZD` 같은 문자 이미지를 텍스트로 변환한다.

- **입력**: 캡차 이미지 (파일 / 바이트 / PIL)
- **출력**: 6글자 문자열 (`A`–`Z`) + 글자별 confidence
- **모델**: 분할 없는 경량 CNN + 1D-conv-mix 헤드(글자 위치 변동에 강함). **약 636K 파라미터**
- **데이터**: 합성 생성(즉석 무한) + 실제 캡차 소량 파인튜닝
- **배포**: ONNX(fp32) → onnxruntime CPU **약 2 ms/이미지**, 모델 파일 2.6 MB

### 결과

| 모델 | 합성 검증셋 | 실제 캡차 (홀드아웃 26장) |
|---|---|---|
| 합성만 `best.pt` | per-char 98.8% | full-string ~70%, per-char ~93% |
| + 실제 76장 파인튜닝 `finetuned.pt` | — | **full-string 96~100%, per-char 99~100%** |

> 홀드아웃 = 파인튜닝에 안 쓴 26장. 표본이 작아 ±수 %p 출렁임.
> 이 수치는 **학습에 쓴 캡차와 같은 시스템 기준**이다. 사이트가 캡차 디자인을 바꾸면
> 재파인튜닝이 필요하다 (합성 pipeline + `best.pt` 가 폴백).

```
config.py        문자셋 / 이미지 크기 / 인코딩
synth.py         합성 캡차 생성기  (python synth.py -> samples.png 미리보기)
preprocess.py    이미지 -> 텐서 (학습/추론 공용)
model.py         LightCaptchaNet (CNN + 1D-conv-mix 헤드)
dataset.py       즉석 생성 Dataset + 텐서 augmentation + 고정 검증셋
train.py         합성 학습 루프  -> checkpoints/best.pt, last.pt
finetune.py      실제 캡차 소량 파인튜닝 -> checkpoints/finetuned.pt
eval_real.py     실제 캡차 폴더로 정확도 측정 (파일명=정답)
predict.py       추론 (CLI + 함수 API). finetuned.pt 있으면 그걸 우선 사용
export_onnx.py   체크포인트 -> captcha.onnx (단일 파일)
onnx_infer.py    onnxruntime 추론 (torch 불필요)
grab_web.py      웹페이지 캡차 캡처 후 인식만 (예시)
auto_fill.py     접속->캡처->인식->입력->클릭 전체 자동화 (--cdp/--browser 등)
Screenshot/      실제 캡차 (파일명 = 정답). finetune.py / eval_real.py 가 사용
```

## 다른 PC에서 세팅

저장소에 학습된 `checkpoints/best.pt` 와 `captcha.onnx` 가 포함돼 있어 **재학습 없이 바로 추론·자동화** 가능하다.

### 1) 공통

```powershell
git clone https://github.com/HSHeee/web-image-ML.git
cd web-image-ML
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

- Python 3.10~3.12 권장.
- 인식 스모크 테스트: `python predict.py 아무_캡차.png`

### 2) 웹 자동화까지 쓸 때

`playwright` 는 위 `requirements.txt` 로 이미 깔린다. **브라우저 실행 방식**만 고르면 된다.

| 방식 | 추가 설치 | 쓰는 인자 |
|---|---|---|
| 이미 로그인해 띄워둔 Chrome/Whale 에 붙기 (권장) | 없음 | `--cdp http://localhost:9222` |
| 설치된 Chrome / Edge 로 새로 띄우기 | 없음 | `--browser chrome` / `--browser msedge` |
| 네이버 웨일로 새로 띄우기 | 없음 | `--browser-path "C:\Program Files\Naver\Naver Whale\Application\whale.exe"` |
| Playwright 번들 Chromium | `playwright install chromium` | (기본값) |

### 3) 학습까지 다시 할 때

추가 설치 없음 (`torch` 로 충분). 폰트는 그 PC의 `C:\Windows\Fonts` 를 자동으로 쓴다.
다른 OS면 `--fonts-dir` 로 폰트 폴더를 지정.

## 1. 합성 데이터 확인

```bash
python synth.py        # samples.png 에 6x8 그리드 저장 — 실제 캡차와 비슷한지 눈으로 확인
```

`synth.py` 의 `CaptchaRenderer` 파라미터(배경색 범위, 회전 각도, 간섭선 빈도, 스페클 수,
폰트 목록)를 실제 타깃에 맞춰 조정할수록 정확도가 올라간다.

## 2. 학습

```bash
python train.py --steps 4000 --batch-size 128 --workers 8
```

- 데이터는 매 스텝 새로 생성되므로 epoch 개념이 없다. `--steps` 로 총량 조절.
- `checkpoints/best.pt` = 검증셋(고정 1000장) full-string 정확도 최고 시점.
- 이어서 학습: `python train.py --resume checkpoints/last.pt --steps 8000`
- GPU 있으면 자동 사용 (수 분이면 충분).

### CPU 속도 참고

이 프로젝트로 측정 시 CPU(16코어)에서 batch 128 기준 **약 40 img/s** 로 모델 연산이
병목이다 (워커 수를 늘려도 비슷). 대략:

| steps | 도달 정확도 | CPU 소요(대략) |
|------:|-----------:|---------------:|
|   800 | ~92 %      | ~20 분         |
|  4000 | ~98 %      | ~2 시간        |
|  6000 | ~99 %      | ~3 시간        |

- 빠르게 보려면 `--steps 800~2000`.
- 제대로 뽑으려면 **GPU 권장** (Colab 무료 T4 로 6000스텝 수 분).
- `--torch-threads` (기본 6) 로 연산 스레드와 DataLoader 워커의 코어 분배를 조정.

주요 지표
- **full-string acc**: 6글자 전부 맞은 비율 (실사용 기준)
- **per-char acc**: 글자 단위 정확도

## 3. 추론

```bash
python predict.py some_captcha.png
python predict.py --show-conf a.png b.png c.png
```

코드에서:

```python
from PIL import Image
from predict import load_model, predict_image

model = load_model()          # 기본: finetuned.pt 있으면 그것, 없으면 best.pt
text, conf, per_char = predict_image(model, Image.open("captcha.png"))
print(text, conf)
```

## 4. 웹페이지 자동 입력 (`auto_fill.py`, 서버 없이 단일 스크립트)

흐름: 접속 → (선행 클릭) → 캡차 요소 캡처 → 인식 → input 입력 → 버튼 클릭.
**본인 소유 / 자동화가 허용된 환경에서만 사용.**

전체 옵션은 `python auto_fill.py --help`. 핵심만:

| 인자 | 설명 |
|---|---|
| `--url` | 처음 여는 주소 |
| `--captcha "SELECTOR"` | 캡차 이미지 요소 CSS 선택자 (`--clip x,y,w,h` 로 좌표 캡처도 가능) |
| `--from-src` | 스크린샷 대신 `<img src>`(base64/data URI 등)를 직접 디코딩 |
| `--input "SELECTOR"` | 정답 입력란 |
| `--submit "SELECTOR"` | 제출 버튼 (생략 시 입력까지만) |
| `--pre-click "SEL"` | 캡차 전에 눌러야 할 것 (반복 지정, 준 순서대로) |
| `--date "2026년 10월 3일"` | 시작 전 날짜 클릭 (`abbr[aria-label=...]`). 생략=건너뜀 (on/off) |
| `--refresh "SEL" --min-conf 0.9 --max-attempts 5` | confidence 낮으면 새 캡차 받아 재시도 |
| `--confirm-submit` | 제출 직전 콘솔 Enter 확인 (권장) |
| `--cdp http://localhost:9222` | 이미 뜬 브라우저에 붙기 (로그인 유지) |
| `--browser chrome\|msedge` / `--browser-path "...whale.exe"` | 새로 띄울 브라우저 |
| `--headed` / `--hold` / `--save-shots` / `--timeout 120000` | 창 표시 / 종료 전 대기 / 캡처 저장 / 대기시간 |

예시 (PowerShell 한 줄, 선택자는 대상 사이트에 맞게 교체):

```powershell
python auto_fill.py --url "http://내부포털/start" --pre-click "text=시작하기" --captcha "img[alt='캡챠 이미지']" --from-src --input "input[placeholder*='문자를 입력']" --submit "text=입력완료" --confirm-submit --browser chrome --headed --save-shots --timeout 120000
```

이미 로그인해 띄워둔 브라우저에 붙이려면 그 브라우저를 먼저 이렇게 실행:

```powershell
& "C:\Program Files\Naver\Naver Whale\Application\whale.exe" --remote-debugging-port=9222
```

그리고 위 명령에서 `--browser chrome --headed` 를 `--cdp http://localhost:9222` 로 바꾼다.

**인식만** 필요하면: `python grab_web.py https://사이트 "img#captcha"`
또는 코드로:

```python
from predict import load_model, predict_bytes
model = load_model()
text, conf, _ = predict_bytes(model, png_bytes)
```

> 이 모델은 **6글자 영대문자 캡차 전용**이다. 일반 문서/이미지 텍스트는 `pytesseract`
> (Tesseract) 같은 범용 OCR 로 `predict_*` 부분만 교체하면 된다.

## 5. 경량 배포 (ONNX)

```bash
python export_onnx.py --ckpt checkpoints/finetuned.pt --out captcha.onnx
python onnx_infer.py some_captcha.png --model captcha.onnx
```

- dynamo exporter 사용(새 헤드 지원), 가중치까지 **단일 파일**로 저장. 약 2.6 MB.
- fp32 그대로 CPU 약 2 ms/이미지. int8 동적 양자화는 이 크기 모델에선 오히려 느려져 안 쓴다.

## 실제 캡차에 맞추는 절차 (이번에 효과 본 순서)

1. **실제 캡차를 `Screenshot/` 에 모은다** — 파일명 = 정답 6글자 (예: `ABCDEF.png`).
   `auto_fill.py --save-shots` 로 뽑은 뒤 사람이 라벨링. 100장 정도부터 유의미.
2. **`synth.py` 를 실물에 맞춘다** — `samples.png` 와 실물을 나란히 보며 배경색·폰트·
   글자 크기/간격·간섭선 위치·점 색을 조정. (이번엔 "선이 글자 관통", "점=글자색",
   "글자 작고 넓은 간격", "위치 변동" 이 핵심이었다.)
3. **합성으로 사전학습**: `python train.py --steps 3000 --real-dir Screenshot`
   → 매 검증 시 실제 정확도도 찍힘. `best.pt` 저장.
4. **실제로 파인튜닝**: `python finetune.py --steps 1000 --real-ratio 0.4`
   → 실제 일부는 홀드아웃(검증 전용). `finetuned.pt` 저장.
5. **측정**: `python eval_real.py --ckpt checkpoints/finetuned.pt --show`

여전히 부족하면: 폰트가 1종이면 `synth.py` 폰트 목록을 그거 하나로 고정 / `widths` 를
키우기 / 실제 샘플 더 모으기.
