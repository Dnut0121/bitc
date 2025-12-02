from __future__ import annotations

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key-change-me")

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

ALLOWED_HOSTS: list[str] = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "dashboard",
]

MIDDLEWARE: list[str] = []

ROOT_URLCONF = "bitc_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [],
        },
    },
]

WSGI_APPLICATION = "bitc_web.wsgi.application"

DATABASES: dict[str, dict[str, str]] = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# BITC 전용 설정
# - 학습된 LSTM 결과를 담은 CSV/메타 JSON 경로
BITC_MODEL_RESULT_CSV = os.environ.get(
    "BITC_MODEL_RESULT_CSV", str((BASE_DIR.parent / "dataset" / "model_result.csv").resolve())
)
BITC_MODEL_META_JSON = os.environ.get(
    "BITC_MODEL_META_JSON", str((BASE_DIR.parent / "dataset" / "model_result_meta.json").resolve())
)

# LSTM 체크포인트 및 검증 데이터 디렉터리
BITC_LSTM_CHECKPOINT = os.environ.get("BITC_LSTM_CHECKPOINT", str((BASE_DIR.parent / "models" / "btc_lstm.pt").resolve()))
BITC_VALIDATE_DIR = os.environ.get(
    "BITC_VALIDATE_DIR",
    str((BASE_DIR.parent / "dataset" / "validate").resolve()),
)

