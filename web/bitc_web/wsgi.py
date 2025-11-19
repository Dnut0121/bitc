from __future__ import annotations

import os
from pathlib import Path

from django.core.wsgi import get_wsgi_application

BASE_DIR = Path(__file__).resolve().parent.parent

# 리포지토리 루트를 파이썬 경로에 추가해서 `import src` 가능
project_root = BASE_DIR.parent
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bitc_web.settings")

application = get_wsgi_application()

