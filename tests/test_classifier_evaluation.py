"""判断方式与留出评估的单元测试。"""

from __future__ import annotations

import _pathbootstrap  # noqa: F401  隔离模式下把 src 加入 sys.path
import unittest
from pathlib import Path

from service_measure.classifier import (
    EvidenceClassifier,
    FullKeywordClassifier,
    LegacyKeywordClassifier,
    judge_new_material,
)
from service_measure.evaluation import (
    household_holdout_split,
    period_holdout_split,
    score_predictions,
)
from service_measure.facts import ReviewState, build_session_facts
from service_measure.io import load_records

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "records.json"


class ClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_records(RECORDS)
        self.facts = build_session_facts(self.records)

    def test_clause_learning_uses_intersection_not_union(self):
        # 用含多主题共现的定稿会话训练，小句只能学到它自己携带的主题。
        training = [
            item
            for item in self.facts.values()
            if item.period in (1, 2, 3)
            and item.state is ReviewState.FINALIZED
            and item.usable
            and item.household not in self.records.holdout_households
        ]
        clf = EvidenceClassifier(self.records).fit(training)
        table = clf.clause_themes[1]
        self.assertEqual(table["周末过来要换乘两次"], frozenset({"地点"}))
        self.assertEqual(table["回执有些词看不懂"], frozenset({"语言"}))
        self.assertEqual(table["放学后找不到空闲窗口"], frozenset({"时段"}))

    def test_multi_theme_household_keeps_both_themes(self):
        training = [
            item
            for item in self.facts.values()
            if item.period in (1, 2, 3)
            and item.state is ReviewState.FINALIZED
            and item.usable
        ]
        clf = EvidenceClassifier(self.records).fit(training)
        # 1-2 同时反馈 地点 + 时段，两项都应留下。
        pred = clf.predict(self.facts["1-2"])
        self.assertEqual(set(pred.themes), {"地点", "时段"})

    def test_cross_taxonomy_without_evidence_must_abstain(self):
        # 整期留出：训练只有 v1 事实，v2 新小句必须弃权，不能猜。
        split = period_holdout_split(self.records, self.facts)
        clf = EvidenceClassifier(self.records).fit(list(split.train))
        pred = clf.predict(self.facts["4-19"])  # 线上预约排不到合适时间
        self.assertEqual(set(pred.themes), set())
        self.assertTrue(pred.unseen_clauses)

    def test_legacy_hard_splits_one_old_label_into_two(self):
        import dataclasses

        split = period_holdout_split(self.records, self.facts)
        legacy = LegacyKeywordClassifier(self.records).fit(list(split.train))
        # 构造一个 v1 口径"时段"信号在 v2 时期被机械承接的场景。
        synthetic = dataclasses.replace(
            self.facts["4-19"], clauses=("放学后找不到空闲窗口",)
        )
        pred = legacy.predict(synthetic)
        self.assertEqual(set(pred.themes), {"线上时段", "现场时段"})

    def test_legacy_new_clause_disappears(self):
        split = period_holdout_split(self.records, self.facts)
        legacy = LegacyKeywordClassifier(self.records).fit(list(split.train))
        pred = legacy.predict(self.facts["4-19"])  # v2 新小句，旧系统不认识
        self.assertEqual(set(pred.themes), set())
        self.assertIn("线上预约排不到合适时间", pred.unseen_clauses)

    def test_no_consent_blocks_prediction(self):
        split = household_holdout_split(self.records, self.facts, taxonomy=2)
        clf = EvidenceClassifier(self.records).fit(list(split.train))
        blocked = judge_new_material(clf, self.facts["4-8"])  # consent=False
        self.assertEqual(set(blocked.themes), set())
        self.assertIn("blocked_no_consent", blocked.source)


class HoldoutEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = load_records(RECORDS)
        self.facts = build_session_facts(self.records)
        self.weights = {
            (row.period, row.household): row.weight for row in self.records.frame
        }

    def _score(self, split):
        clf = EvidenceClassifier(self.records).fit(list(split.train))
        predictions = {
            item.session: judge_new_material(clf, item) for item in split.test
        }
        return score_predictions("evidence", split, predictions, self.records, self.weights)

    def test_household_holdout_splits_are_disjoint(self):
        for taxonomy in (1, 2):
            split = household_holdout_split(self.records, self.facts, taxonomy)
            train_hh = {item.household for item in split.train}
            test_hh = {item.household for item in split.test}
            self.assertFalse(train_hh & test_hh)
            self.assertTrue(test_hh <= self.records.holdout_households)

    def test_period_holdout_excludes_whole_periods(self):
        split = period_holdout_split(self.records, self.facts)
        self.assertEqual(
            {item.period for item in split.train}, set(self.records.mature_periods)
        )
        self.assertEqual(
            {item.period for item in split.test}, set(self.records.holdout_periods)
        )

    def test_evidence_beats_baselines_on_in_taxonomy_holdout(self):
        split = household_holdout_split(self.records, self.facts, taxonomy=2)
        results = {}
        for name, builder in (
            ("evidence", EvidenceClassifier),
            ("legacy", LegacyKeywordClassifier),
            ("full", FullKeywordClassifier),
        ):
            clf = builder(self.records).fit(list(split.train))
            preds = {item.session: clf.predict(item) for item in split.test}
            results[name] = score_predictions(name, split, preds, self.records, self.weights)
        self.assertGreaterEqual(
            results["evidence"].exact_joint_match, results["full"].exact_joint_match
        )
        self.assertGreaterEqual(
            results["evidence"].mean_jaccard, results["legacy"].mean_jaccard
        )

    def test_cross_taxonomy_evidence_abstains_and_baseline_overreaches(self):
        split = period_holdout_split(self.records, self.facts)
        evidence = EvidenceClassifier(self.records).fit(list(split.train))
        legacy = LegacyKeywordClassifier(self.records).fit(list(split.train))
        evidence_preds = {item.session: evidence.predict(item) for item in split.test}
        legacy_preds = {item.session: legacy.predict(item) for item in split.test}
        evidence_score = score_predictions(
            "evidence", split, evidence_preds, self.records, self.weights
        )
        legacy_score = score_predictions(
            "legacy", split, legacy_preds, self.records, self.weights
        )
        # 拟采用方式：没有新词典复核事实时大量弃权，且绝不硬造一对多子标签。
        self.assertGreater(evidence_score.n_abstained_with_signal, 0)
        self.assertEqual(evidence_score.split_label_errors, ())

        # 基准在无证据时仍输出标签：对有真值的会话产生假阳性（过度自信），
        # 其假阳性总量严格高于诚实弃权的拟采用方式。
        def false_positives(score):
            return sum(s.false_positive for s in score.label_scores.values())

        self.assertGreater(
            false_positives(legacy_score), false_positives(evidence_score)
        )
        # 弃权方的占比误差来自"拒绝填报"，而非错误标注；基准误差来自错误标注。
        self.assertGreater(legacy_score.abstention_rate, 0.0)
        self.assertGreater(evidence_score.abstention_rate, legacy_score.abstention_rate)

    def test_marginal_columns_do_not_add_to_one(self):
        split = household_holdout_split(self.records, self.facts, taxonomy=1)
        score = self._score(split)
        # 多主题家庭存在 -> 边际占比之和不是 1，证明栏目不可相加。
        self.assertNotAlmostEqual(score.truth_sum_of_marginal_columns, 1.0)

    def test_hard_split_detector_flags_both_children_but_evidence_does_not(self):
        import dataclasses

        split = period_holdout_split(self.records, self.facts)
        legacy = LegacyKeywordClassifier(self.records).fit(list(split.train))
        # 把一个 v1"时段"信号放进 v2 会话：基准按 bridges 同时给两个子主题。
        target = dataclasses.replace(
            self.facts["4-19"], clauses=("放学后找不到空闲窗口",)
        )
        legacy_pred = legacy.predict(target)
        self.assertEqual(set(legacy_pred.themes), {"线上时段", "现场时段"})

        # 用只含该合成会话的最小测试集驱动评分，硬拆应被审计捕获。
        synth_split = dataclasses.replace(split, test=(target,))
        score = score_predictions(
            "legacy",
            synth_split,
            {target.session: legacy_pred},
            self.records,
            self.weights,
        )
        self.assertTrue(score.split_label_errors)
        self.assertEqual(score.split_label_errors[0]["bridge_parent"], "时段")

        # 同一输入下拟采用方式弃权，不产生任何硬拆。
        evidence = EvidenceClassifier(self.records).fit(list(split.train))
        ev_pred = evidence.predict(target)
        ev_score = score_predictions(
            "evidence", synth_split, {target.session: ev_pred},
            self.records, self.weights,
        )
        self.assertEqual(ev_score.split_label_errors, ())
        self.assertEqual(set(ev_pred.themes), set())


if __name__ == "__main__":
    unittest.main()
