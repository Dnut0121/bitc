# BITC

캔들 기반으로 매수/매도 압력을 계산하고,  
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
python -m venv .venv           # 가상환경 생성

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

CUDA 환경이 있다면, PyTorch는 본인 GPU/드라이버에 맞춰 공식 문서를 참고해 별도로 설치하는 것을 권장합니다.

---

## 데이터 준비

1) 바이낸스 일자별 ZIP 다운로드 (예: 1m 캔들)

```bash
python scripts/download_binance_klines.py \
  --symbol BTCUSDT \
  --interval 1m \
  --start 2025-01-01 \
  --end 2025-01-05 \
  --dest dataset/binance_raw
```

2) 단일 CSV로 병합

```bash
python scripts/prepare_dataset.py dataset/binance_raw dataset/binance_ohlcv.csv \
  --start 2025-01-01 --end 2025-01-05
```

`run_pipeline.py`는 `dataset/daily`에 일자별 CSV가 없으면 자동으로 `dataset/binance_ohlcv.csv`를 하루 단위로 쪼개서 만듭니다. 원본 폴더 구조(예: `dataset/binance_raw/2025-01-01/...`)를 그대로 쓰려면 `--use-raw-daily` 옵션을 줍니다.

---

## LSTM 파이프라인 (walk-forward)

`.env`로 기본값을 설정할 수 있습니다.

```bash
# 예시 .env
BITC_OUTPUT=dataset/binance_ohlcv.csv
BITC_SAVE_MODEL=models/btc_lstm.pt
BITC_WINDOW=60
BITC_EPOCHS=1
BITC_BATCH_SIZE=32
BITC_FEE_RATE=0.0006
BITC_SLIPPAGE_RATE=0.0003
```

실행 예시:

```bash
python run_pipeline.py \
  --source dataset/binance_ohlcv.csv \
  --daily-dir dataset/daily \
  --save-model models/btc_lstm.pt \
  --epochs 1 \
  --run-backtest \
  --eval-after \
  --eval-horizon 60
```

- `--single-day YYYY-MM-DD` : 특정 하루만 학습
- `--holdout-days N` : 마지막 N일은 학습 제외하고 평가/백테스트만 실행
- `--cost-aware-label` : 수수료/슬리피지 반영 라벨 사용
- `--wandb` : Weights & Biases 로깅 활성화 (`--wandb-project`, `--wandb-entity`와 함께 사용)

---

## 메타 모델 파이프라인 (triple-barrier)

기존 LSTM 체크포인트(`--base-model`)를 바탕으로 메타 모델을 학습합니다.

```bash
python run_meta_pipeline.py \
  --daily-dir dataset/daily \
  --base-model models/btc_lstm.pt \
  --save-meta-model models/btc_meta.pt \
  --tb-horizon 60 \
  --tb-up-pct 0.001 \
  --tb-down-pct 0.001
```

`--base-long-threshold` / `--base-short-threshold`로 LSTM 확률 컷오프를 조정할 수 있습니다.

---

## 하이브리드 파이프라인 (LSTM + XGBoost)

```bash
python run_hybrid_pipeline.py \
  --csv dataset/binance_ohlcv.csv \
  --tail-rows 400000 \
  --lookback 120 \
  --label-horizon 60 \
  --epochs 3 \
  --batch-size 64 \
  --xgb-rounds 300 \
  --save-dir models/hybrid_ensemble
```

주요 토글: `--cost-aware-label`, `--force-cpu`, `--resample 1s|none`, `--max-sequences`.

### Django 대시보드 (모델 출력 확인용, 실험적)

LSTM이 계산한 `prob_up` 등 스냅샷을 간단히 웹으로 보고 싶다면 Django 개발 서버를 띄울 수 있습니다.

1. 의존성 설치  
   ```bash
   pip install -r requirements.txt
   ```

2. 데이터셋 준비 (이미 CSV가 있다면 생략)
   ```bash
   python run_pipeline.py --source dataset/binance_ohlcv.csv --epochs 1 --batch-size 16 --window 60
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
