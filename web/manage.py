#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bitc_web.settings")

    # 프로젝트 루트(리포지토리 루트)를 파이썬 경로에 추가해서 `import src`가 가능하도록 한다.
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

