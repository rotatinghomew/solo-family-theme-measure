"""命令行入口：python -m service_measure.pipeline_cli <records.json> <output.json>"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="家长留言主题量度与材料组合决策分析")
    parser.add_argument("records", type=Path, help="合成业务记录 records.json 路径")
    parser.add_argument("output", type=Path, help="分析结果输出 JSON 路径")
    args = parser.parse_args()
    result = run_pipeline(args.records, args.output)
    print(f"已写出 {args.output}")
    print(
        "材料决策期 =",
        result["editorial_decision"]["decision_period"],
        "；材料组合数 =",
        len(result["editorial_decision"]["combos"]),
    )


if __name__ == "__main__":
    main()
