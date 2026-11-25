"""数据契约与会话事实构建的单元测试。"""

from __future__ import annotations

import _pathbootstrap  # noqa: F401  隔离模式下把 src 加入 sys.path
import unittest
from pathlib import Path

from service_measure.facts import ReviewState, build_session_facts
from service_measure.io import load_records
from service_measure.taxonomy import active_themes, carry_forward, snapshot

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "records.json"


class FactBuildingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_records(RECORDS)
        self.facts = build_session_facts(self.records)

    def test_multi_channel_session_is_one_household_one_session(self):
        # 站点 + 回访重复反馈同一件事，只能是一个会话、一户，不能当两户/两条困难。
        multi = [s for s, rows in self._channels().items() if len(rows) > 1]
        self.assertTrue(multi)
        for session in multi:
            item = self.facts[session]
            self.assertEqual(len(item.channels), 2)
            # 小句去重：两条转写内容相同时只保留一份主题信号。
            self.assertEqual(len(item.clauses), len(set(item.clauses)))

    def _channels(self):
        grouped: dict[str, set[str]] = {}
        for row in self.records.summaries:
            grouped.setdefault(row.session, set()).add(row.channel)
        return grouped

    def test_finalized_intersection_is_confirmed_fact(self):
        item = self.facts["1-2"]  # 两位复核人均定稿 地点+时段
        self.assertEqual(item.state, ReviewState.FINALIZED)
        self.assertEqual(set(item.confirmed), {"地点", "时段"})

    def test_pending_keeps_partial_agreement(self):
        # 1-14：T1 地点+对象，T2 对象 -> 对象为确认事实，地点未决。
        item = self.facts["1-14"]
        self.assertEqual(item.state, ReviewState.PENDING)
        self.assertEqual(set(item.confirmed), {"对象"})
        self.assertEqual(set(item.unresolved), {"地点"})

    def test_joint_review_partial_consensus(self):
        # 5-14：联合复核 agreement=语言，unresolved=线上时段。
        item = self.facts["5-14"]
        self.assertIn("语言", item.confirmed)
        self.assertIn("线上时段", item.unresolved)
        self.assertNotIn("线上时段", item.confirmed)

    def test_joint_review_unresolved_on_empty_labels(self):
        # 5-12：双人定稿空标签，但联合复核记录 对象 未决；定稿的空事实仍有效。
        item = self.facts["5-12"]
        self.assertEqual(set(item.confirmed), set())
        self.assertIn("对象", item.unresolved)

    def test_evidence_trail_entries_are_atomic(self):
        item = self.facts["5-14"]
        self.assertIn("review:5-14:待合议", item.evidence)
        self.assertIn("joint_review:5-14", item.evidence)
        # 证据条目不得被拆成单字符。
        self.assertTrue(all(len(entry) >= 3 for entry in item.evidence))

    def test_no_consent_session_blocked(self):
        item = self.facts["1-8"]  # consent=False
        self.assertFalse(item.usable)

    def test_nonresponse_is_not_confirmed_no_difficulty(self):
        # 未回访户在抽样框中 responded=False，且没有任何会话事实。
        non_responded = {
            row.household for row in self.records.frame if not row.responded
        }
        session_households = {item.household for item in self.facts.values()}
        # 未回访户根本没有会话，不能被当作"确证无障碍"。
        self.assertTrue(non_responded)
        for household in non_responded:
            self.assertNotIn(household, session_households)


class TaxonomyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_records(RECORDS)

    def test_period_versions(self):
        self.assertEqual(self.records.period_taxonomy(3), 1)
        self.assertEqual(self.records.period_taxonomy(4), 2)

    def test_withdrawal_excludes_theme_but_keeps_history(self):
        snap5 = snapshot(self.records, 5)
        self.assertIn("地点", snap5.withdrawn)
        self.assertNotIn("地点", active_themes(self.records, 5))
        # 撤回前时期不受影响。
        self.assertNotIn("地点", snapshot(self.records, 4).withdrawn)
        # 已采用材料的会话依据仍在记录中可查。
        approved = set(self.records.revision.approved_sessions)
        self.assertTrue(approved & {row.session for row in self.records.reviews})

    def test_one_to_many_bridge_never_auto_duplicates(self):
        children = carry_forward(self.records, "时段", 5)
        self.assertEqual(set(children or ()), {"线上时段", "现场时段"})
        # 一对多关系存在本身不产生任何标签：调用方必须拿到区分子标签的复核事实。
        one_to_one = carry_forward(self.records, "对象", 5)
        self.assertEqual(one_to_one, ("对象",))

    def test_unknown_old_theme_has_no_bridge(self):
        self.assertIsNone(carry_forward(self.records, "不存在的主题", 5))


if __name__ == "__main__":
    unittest.main()
