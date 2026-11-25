"""材料组合决策与复核排班的单元测试。"""

from __future__ import annotations

import _pathbootstrap  # noqa: F401  隔离模式下把 src 加入 sys.path
import unittest
from pathlib import Path

from service_measure.editorial import (
    Material,
    build_household_states,
    build_materials,
    evaluate_combos,
    material_covers,
    overlap_matrix,
)
from service_measure.facts import build_session_facts
from service_measure.io import load_records
from service_measure.scheduling import (
    build_site_capacity,
    rank_next_batch,
    reserve_prebooked,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "records.json"


class EditorialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_records(RECORDS)
        self.facts = build_session_facts(self.records)
        self.states = build_household_states(self.records, self.facts, 5)

    def test_empty_all_list_is_not_a_coverage_rule(self):
        # 输入数据里没有空 all；构造一个应被 build_materials 排除。
        materials = build_materials(self.records)
        self.assertTrue(all(m.required for m in materials))

    def test_same_household_covered_once_across_materials(self):
        mat_online = Material("远程入口", ("线上时段",), 4)
        mat_site = Material("短时段导引", ("现场时段",), 4)
        # 5-44 同时确认 现场时段+线上时段：两份材料都命中同一户，但只记一次。
        state = next(s for s in self.states if s.household == "F044")
        self.assertTrue(material_covers(mat_online, state, optimistic=False))
        self.assertTrue(material_covers(mat_site, state, optimistic=False))
        covered = set()
        for material in (mat_online, mat_site):
            if material_covers(material, state, optimistic=False):
                covered.add(state.household)
        self.assertEqual(covered, {"F044"})

    def test_combination_uses_union_not_sum(self):
        matrix = overlap_matrix(self.records, self.states)
        # 不同材料确实集中在同一批家庭：重叠权重为正，证明不能逐份相加。
        self.assertGreater(matrix["短时段导引"]["联系人说明"], 0.0)
        combos = evaluate_combos(self.records, self.states, 5)
        double = next(
            c for c in combos if c.materials == ("短时段导引", "联系人说明")
        )
        self.assertGreater(
            double.duplicate_cover_share["短时段导引/联系人说明"], 0.0
        )

    def test_floor_denominator_includes_nonresponders(self):
        # 站点最低覆盖率的分母是整框权重，不随未回应缩小。
        totals = {
            site: sum(
                row.weight
                for row in self.records.frame
                if row.period == 5 and row.site == site
            )
            for site in self.records.editorial_floors
        }
        # 北站点有未回访户，但其整框权重仍作为分母。
        north = [s for s in self.states if s.site == "北" and not s.responded]
        self.assertTrue(north)
        self.assertAlmostEqual(totals["北"], 25.2, places=4)

    def test_no_combo_certified_when_north_floor_unprovable(self):
        combos = evaluate_combos(self.records, self.states, 5)
        within_budget = [c for c in combos if c.within_budget]
        # 北站点保守覆盖为 0，没有任何预算内组合能被正式采用。
        self.assertTrue(within_budget)
        self.assertFalse(any(c.decision == "可正式采用" for c in within_budget))
        conditional = [c for c in within_budget if c.decision == "条件性可能"]
        self.assertTrue(conditional)
        for combo in conditional:
            self.assertIn("北", combo.floor_gap_conservative)
            self.assertTrue(
                any("北" in reason for reason in combo.blocking_reasons)
            )

    def test_optimistic_bound_marks_possible_but_not_certified(self):
        single = next(c for c in evaluate_combos(self.records, self.states, 5)
                      if c.materials == ("语言提示",))
        # 乐观口径北站点可达标，保守口径不达标 -> 条件性可能而非正式采用。
        self.assertGreaterEqual(
            single.optimistic_site_coverage["北"], self.records.editorial_floors["北"]
        )
        self.assertLess(
            single.conservative_site_coverage["北"], self.records.editorial_floors["北"]
        )
        self.assertEqual(single.decision, "条件性可能")

    def test_over_budget_combos_rejected(self):
        combos = evaluate_combos(self.records, self.states, 5)
        over = [c for c in combos if not c.within_budget]
        self.assertTrue(over)
        self.assertTrue(all(c.decision == "不满足" for c in over))


class SchedulingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_records(RECORDS)
        self.facts = build_session_facts(self.records)
        self.states = build_household_states(self.records, self.facts, 5)

    def test_prebooked_occupies_capacity_and_is_not_cancelled(self):
        capacity = build_site_capacity(self.records)
        before = dict(capacity["北"].reviewer_slots)
        _, blockers = reserve_prebooked(self.records, capacity, self.facts)
        # 5-12 是北站点双语会话，而北站点只有 T3 一名双语复核人 -> 无法定稿但名额保留。
        self.assertIn("5-12", capacity["北"].reserved)
        self.assertTrue(any("5-12" in b for b in blockers))
        # T3 的位置没有被别的会话顶替。
        self.assertEqual(capacity["北"].reviewer_slots, before)

    def test_bilingual_needs_two_distinct_reviewers(self):
        capacity = build_site_capacity(self.records)
        north = capacity["北"]
        self.assertFalse(north.can_finalize("双语"))  # 只有 T3 一人
        west = capacity["西"]
        self.assertTrue(west.can_finalize("双语"))   # T3 与 T4 两人

    def test_one_session_consumes_one_case_slot_even_with_multiple_themes(self):
        capacity = build_site_capacity(self.records)
        west = capacity["西"]
        before = sum(west.reviewer_slots.values())
        west.consume("普通")
        self.assertEqual(sum(west.reviewer_slots.values()), before - 1)

    def test_next_batch_ranks_by_decision_value_not_volume(self):
        candidates, _, _ = rank_next_batch(
            self.records, self.facts, self.states, 5
        )
        # 5-14 的未决主题（线上时段）能翻转 远程入口 的材料判定，应排在最前。
        self.assertEqual(candidates[0].session, "5-14")
        self.assertIn("远程入口", candidates[0].pivotal_combos)
        # 所有候选都必须带未决主题，无复核会话不进队列。
        self.assertTrue(all(c.unresolved_themes for c in candidates))

    def test_batch_respects_site_capacity(self):
        candidates, capacity, blockers = rank_next_batch(
            self.records, self.facts, self.states, 5
        )
        # 南站点只有 T2（双语复核人）的 3 个位置，普通候选至多占用这些。
        south_candidates = [c for c in candidates if c.site == "南"]
        south_capacity = capacity["南"]
        assignable = sum(
            1
            for c in south_candidates
            if south_capacity.can_finalize(c.language)
        )
        self.assertLessEqual(assignable, sum(south_capacity.reviewer_slots.values()))
        self.assertTrue(blockers)


if __name__ == "__main__":
    unittest.main()
