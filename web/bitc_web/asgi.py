from __future__ import annotations

import os
from pathlib import Path

from django.core.asgi import get_asgi_application

BASE_DIR = Path(__file__).resolve().parent.parent

project_root = BASE_DIR.parent
import sys

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bitc_web.settings")

application = get_asgi_application()

