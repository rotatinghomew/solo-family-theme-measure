"""测试路径引导。

在隔离模式（python3 -I）下运行 `python3 -m unittest discover -s tests` 时，
工作目录不在 sys.path 上、PYTHONPATH 也不生效；本模块位于 tests 目录内，
可被发现机制直接导入，由它把仓库 src 加入 sys.path。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
