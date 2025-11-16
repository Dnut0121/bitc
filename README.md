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
