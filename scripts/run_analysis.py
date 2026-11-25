#!/usr/bin/env python3
"""分析入口脚本（隔离模式运行：python3 -I scripts/run_analysis.py ...）。

脚本自行把 src 加入 sys.path，不依赖当前工作目录或环境变量，
满足"用 -I 运行、解释器不从数据目录加载代码"的要求。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from service_measure.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    records = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "records.json"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(records, output)
    print(f"已写出 {output}")
    print(
        "材料决策期 =",
        result["editorial_decision"]["decision_period"],
        "；材料组合数 =",
        len(result["editorial_decision"]["combos"]),
    )


if __name__ == "__main__":
    main()
