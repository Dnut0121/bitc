# BITC

1초 캔들 기반으로 매수/매도 압력을 계산하고,  
LSTM으로 다음 1초 종가 움직임을 실험해 보는 개인 퀀트 연구용 프로젝트입니다.

- 개발자: [@Dnut0121](https://github.com/Dnut0121) (1인 개발)
- 개발 기간: 2025‑11‑17 ~ 진행 중
- 상태: 초기 개발 단계 (구조와 기능은 언제든 크게 변경될 수 있습니다)

> ⚠️ 투자 조언이 아닙니다.  
> 이 프로젝트를 사용한 모든 매매/손실에 대한 책임은 전적으로 사용자에게 있습니다.

---

## 기본 개발 환경

- Python 3.10+
- OS: Windows / Linux
- 의존성: `requirements.txt` 참고
  - Django 대시보드를 쓰고 싶다면 `Django`도 함께 설치됩니다.

### 로컬 설정 (Development Setup)

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

CUDA 환경이 있다면, PyTorch는 본인 GPU/드라이버에 맞춰 공식 문서를 참고해 별도로 설치하는 것을 권장합니다.

---

## 현재 상태

- 실험용 파이프라인과 간단한 LSTM 모델, 트레이드 시나리오 로직을 계속 수정/보완 중입니다.
- 앞으로 구조와 사용법이 자주 바뀔 수 있으니,  
  **이 README는 “진행 상황 메모용”** 정도로만 참고해 주세요.

안정된 버전이 나오면, 그때 전체 기능 설명 / 예제 / 사용 가이드를 정식으로 정리할 예정입니다.

### Django 대시보드 (모델 출력 확인용, 실험적)

LSTM이 계산한 `prob_up` 등 스냅샷을 간단히 웹으로 보고 싶다면 Django 개발 서버를 띄울 수 있습니다.

1. 의존성 설치  
   ```bash
   pip install -r requirements.txt
   ```

2. 데이터셋 준비 (기존 파이프라인 그대로 사용)  
   ```bash
   python run_pipeline.py --start 2025-01-01 --end 2025-01-02 --epochs 1 --tail-rows 5000
   ```
   이 과정에서 `dataset/binance_ohlcv.csv`가 생성/갱신됩니다.

3. Django 개발 서버 실행  
   ```bash
   cd web
   python manage.py runserver
   ```

4. 브라우저에서 접속  
   - `http://127.0.0.1:8000/`  
   - 최근 캔들 기준으로 계산된 압력/비율과 LSTM의 `prob_up`, 검증 성능 등을 확인할 수 있습니다.

환경변수로 기본 파라미터를 조정할 수 있습니다.

- `BITC_DATA_CSV` : 사용할 CSV 경로 (기본: `dataset/binance_ohlcv.csv`)
- `BITC_LOOKBACK` : LSTM lookback/window 길이 (기본: 60)
- `BITC_TAIL_ROWS`: CSV에서 뒤쪽 N행만 사용 (기본: 5000)
- `BITC_EPOCHS`, `BITC_BATCH_SIZE`, `BITC_LR`, `BITC_GRAD_CLIP` 등도 변경 가능
