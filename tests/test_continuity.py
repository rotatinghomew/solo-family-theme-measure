"""跨批次连续性测试：新回访批次进入同一项目，口径与决议可继续核对。

对应任务约定：接到新的回访批次，负责排班的人与编简报的人应能从同一项目
继续核对所见结论的适用口径（词典版本、撤回日期、同意状态）和决议差别。
"""

from __future__ import annotations

import _pathbootstrap  # noqa: F401  隔离模式下把 src 加入 sys.path
import unittest
from pathlib import Path

from service_measure.briefing import build_briefing
from service_measure.editorial import build_household_states, evaluate_combos
from service_measure.facts import build_session_facts
from service_measure.io import load_records
from service_measure.taxonomy import active_themes, snapshot

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "records.json"


class ContinuityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_records(RECORDS)
        self.facts = build_session_facts(self.records)

    def test_each_period_uses_one_explicit_taxonomy_snapshot(self):
        for period in range(1, 6):
            snap = snapshot(self.records, period)
            briefing = build_briefing(self.records, self.facts, period)
            self.assertEqual(briefing.taxonomy, snap.version)
            # 简报的主题空间与同期材料判定使用的主题空间完全一致。
            self.assertEqual(
                set(briefing.active_themes), set(active_themes(self.records, period))
            )

    def test_withdrawn_theme_excluded_from_brief_and_materials_but_kept_on_record(self):
        briefing5 = build_briefing(self.records, self.facts, 5)
        self.assertIn("地点", briefing5.withdrawn)
        self.assertNotIn("地点", briefing5.active_themes)
        self.assertNotIn("地点", briefing5.marginal_prevalence)
        # 历史复核行仍保留 地点 标签，已采用材料依据可查。
        history = [r for r in self.records.reviews if "地点" in r.labels]
        self.assertTrue(history)
        self.assertIn(
            "5-14", self.records.revision.approved_sessions
        )

    def test_recomputing_is_deterministic_across_runs(self):
        # 同一项目重复运行，简报与材料决议必须逐字段一致（可继续核对而非重新解释）。
        briefing_a = build_briefing(self.records, self.facts, 5)
        briefing_b = build_briefing(self.records, self.facts, 5)
        self.assertEqual(briefing_a, briefing_b)

        states_a = build_household_states(self.records, self.facts, 5)
        states_b = build_household_states(self.records, self.facts, 5)
        combos_a = evaluate_combos(self.records, states_a, 5)
        combos_b = evaluate_combos(self.records, states_b, 5)
        self.assertEqual(
            [(c.materials, c.decision) for c in combos_a],
            [(c.materials, c.decision) for c in combos_b],
        )

    def test_unissued_tasks_follow_new_taxonomy_when_theme_withdrawn(self):
        # 撤回日期之后：未发任务按新口径核定，即任何材料要求都不得依赖撤回主题。
        for material in self.records.editorial:
            self.assertNotIn(
                self.records.revision.withdrawn_theme,
                material["all"],
            )


if __name__ == "__main__":
    unittest.main()
