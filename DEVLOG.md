# DEVLOG

## 2025-11-17

### 개요

- 프로젝트 시작일. LSTM 실험 파이프라인을 처음 세팅하고, 기본 데이터 흐름과 간단한 트레이드 시뮬레이션까지 한 번에 돌아가도록 구성한 날.

### 작업 내용 요약

- **데이터 로딩 및 캐시 파이프라인 정리**
  - `src/data.py`에서 `load_ohlcv`가 CSV의 `timestamp` 형식(문자열/에포크)에 따라 자동으로 파싱되도록 개선.
  - tail‑rows만 읽는 `_read_last_rows`를 보완해서 큰 CSV도 일부만 안전하게 로드.
  - 캐시된 CSV가 이미 있을 경우, `scripts/prepare_dataset.py`에서 불필요한 재머지를 건너뛰고 재사용하도록 수정.

- **파이프라인 엔트리 및 경로 정리**
  - 루트의 `run_pipeline.py`를 메인 파이프라인 엔트리로 사용하도록 정리.
  - README와 코드 내에서 `scripts/run_pipeline.py` 등 잘못된 경로 언급을 `run_pipeline.py`로 통일.

- **LSTM 학습 및 CUDA 관련 버그 수정**
  - `src/model.py`의 LSTM 학습 루프를 정리하고 로그 출력(에폭/배치별 손실, 검증 성능) 추가.
  - CUDA 텐서를 바로 `numpy()`로 변환하던 부분을 `.cpu().numpy()`로 교체해 GPU 환경에서의 타입 에러 해결.

- **트레이드 추천/기대 수익 로직 추가**
  - `src/cli.py`에 `suggest_trades` 함수를 추가해:
    - 과거 데이터에서 `prob_up`이 높은 구간을 기준으로 가장 좋은 진입/청산 시나리오를 단순 백테스트.
    - 현재 캔들 기준으로 손익비, 수수료, 슬리피지를 반영한 단순 기대 수익을 계산해 콘솔에 출력.
  - CLI 옵션으로 `--prob-threshold`, `--horizon`, `--risk-reward`, `--fee-rate`, `--slippage-rate`, `--min-expected-pct` 등을 추가해, 전략 파라미터를 명령줄에서 튜닝할 수 있게 함.

- **CLI 스냅샷/출력 개선**
  - `print_snapshot`에서 buy/sell pressure, imbalance를 소수 여섯째 자리까지 출력하도록 포맷 조정.
  - 로깅 설정을 추가해 `bitc.cli` / `src.model` 로그를 시간/레벨과 함께 확인 가능하게 정리.


### 메모 / 다음 할 일 (초기 아이디어)

- LSTM 출력과 트레이드 로직을 이용한 제대로 된 백테스트 엔진(전체 PnL 커브, 드로우다운 등) 설계.
- 다른 심볼/시간 구간(BTCUSDT 외, 1m/5m 등)에 대한 일반화 가능성 검토.
- 노트북/대시보드 형태의 시각화/리포트 템플릿 추가.

