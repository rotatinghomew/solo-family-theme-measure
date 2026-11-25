#!/usr/bin/env python3
"""测试入口（隔离模式：python3 -I scripts/run_tests.py）。

先把 src 与 tests 加入 sys.path，再用 unittest 做发现，
保证在 -I 隔离模式、任意当前工作目录下都能导入 service_measure。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
