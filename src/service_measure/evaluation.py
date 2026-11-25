"""留出资料上的判断方式比较。

两个分离检验，适用范围各自记录、互不混用：

1. 整户留出（household holdout）：
   approval.holdout_households 中的家庭全程不参与拟合；
   检验在同一词典版本内对陌生家庭的泛化。v1 用 1-3 期，v2 用 4-5 期。
2. 整期留出（period holdout）：
   holdout_periods（4-5 期）整期排除，训练只用 mature_periods（1-3 期）；
   检验跨词典版本的泛化。拟采用方式在没有新词典复核事实时必须弃权，
   旧标签的一对多关系不得拆成两份。

评估不只看逐标签正确率，还看：
- 联合主题集合的完全匹配与 Jaccard；
- 未决（弃权）比例是否与真实待合议比例相称；
- 加权主题占比估计的偏差（简报口径），并明确各主题占比不是可相加栏目。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .classifier import Prediction
from .facts import SessionFacts, ReviewState
from .io import Records


@dataclass(frozen=True)
class HoldoutSplit:
    name: str
    scope_note: str
    taxonomy: int
    train: tuple[SessionFacts, ...]
    test: tuple[SessionFacts, ...]


def household_holdout_split(
    records: Records, facts: dict[str, "SessionFacts"], taxonomy: int
) -> HoldoutSplit:
    periods = [p for p in range(1, 6) if records.period_taxonomy(p) == taxonomy]
    train: list[SessionFacts] = []
    test: list[SessionFacts] = []
    for item in facts.values():
        if item.period not in periods:
            continue
        if item.household in records.holdout_households:
            test.append(item)
        else:
            train.append(item)
    note = (
        f"整户留出：排除 {sorted(records.holdout_households)}；"
        f"仅用词典 v{taxonomy} 对应的第 {periods} 期；训练与测试家庭互不重叠。"
    )
    return HoldoutSplit(
        name=f"household_holdout_v{taxonomy}",
        scope_note=note,
        taxonomy=taxonomy,
        train=tuple(sorted(train, key=lambda x: x.session)),
        test=tuple(sorted(test, key=lambda x: x.session)),
    )


def period_holdout_split(
    records: Records, facts: dict[str, "SessionFacts"]
) -> HoldoutSplit:
    """整期留出：训练仅 mature_periods；测试为留出期的全部已同意会话。"""
    train = [
        item
        for item in facts.values()
        if item.period in records.mature_periods
        and item.household not in records.holdout_households
    ]
    test = [
        item
        for item in facts.values()
        if item.period in records.holdout_periods
    ]
    note = (
        f"整期留出：训练仅第 {list(records.mature_periods)} 期（并排除留出家庭），"
        f"测试为整期留出的第 {list(records.holdout_periods)} 期全部已同意会话；"
        "训练中不存在 v2 词典复核事实，跨版主题必须按 bridges 规则处理或弃权。"
    )
    return HoldoutSplit(
        name="period_holdout_cross_taxonomy",
        scope_note=note,
        taxonomy=2,
        train=tuple(sorted(train, key=lambda x: x.session)),
        test=tuple(sorted(test, key=lambda x: x.session)),
    )


@dataclass
class LabelScore:
    label: str
    true_positive: int
    false_positive: int
    false_negative: int
    support: int

    @property
    def precision(self) -> float | None:
        denom = self.true_positive + self.false_positive
        return None if denom == 0 else self.true_positive / denom

    @property
    def recall(self) -> float | None:
        denom = self.true_positive + self.false_negative
        return None if denom == 0 else self.true_positive / denom

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)


@dataclass
class ScoreReport:
    method: str
    split: str
    scope_note: str
    n_scored: int
    n_with_truth: int
    label_scores: dict[str, LabelScore]
    exact_joint_match: float
    mean_jaccard: float
    abstention_rate: float
    empirical_pending_rate: float
    abstained_sessions: tuple[str, ...]
    pending_sessions: tuple[str, ...]
    weighted_prevalence: dict[str, float]
    truth_weighted_prevalence: dict[str, float]
    prevalence_abs_error: dict[str, float]
    sum_of_marginal_columns: float
    truth_sum_of_marginal_columns: float
    joint_set_distribution: dict[tuple[str, ...], float] = field(default_factory=dict)
    split_label_errors: tuple[dict[str, object], ...] = ()
    n_abstained_with_signal: int = 0


def _ground_truth(item: SessionFacts) -> frozenset[str] | None:
    """有已确认事实才可评分；无复核会话没有真值，不参与正确率统计。"""
    if item.state is ReviewState.NOT_REVIEWED or not item.usable:
        return None
    return item.confirmed


def score_predictions(
    method: str,
    split: HoldoutSplit,
    predictions: dict[str, Prediction],
    records: Records,
    weights: dict[tuple[int, str], float],
) -> ScoreReport:
    tested = [item for item in split.test if _ground_truth(item) is not None]
    themes = records.themes(split.taxonomy)
    counters = {
        theme: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for theme in themes
    }
    exact_matches = 0
    jaccard_sum = 0.0
    abstained: list[str] = []
    abstained_with_signal = 0
    pending: list[str] = []
    split_errors: list[dict[str, object]] = []
    weighted_hits = {theme: 0.0 for theme in themes}
    weighted_pred = {theme: 0.0 for theme in themes}
    weight_total = 0.0
    joint_dist: dict[frozenset[str], float] = {}

    for item in tested:
        truth = _ground_truth(item)
        pred = predictions[item.session]
        weight = weights[(item.period, item.household)]
        weight_total += weight
        if item.state is ReviewState.PENDING:
            pending.append(item.session)
        # 只把"有困难信号但证据不足"的弃权计入弃权率；无主题会话弃权是正常输出。
        has_signal = any(
            clause != "暂未提出其他困难" for clause in item.clauses
        )
        if pred.is_abstention:
            abstained.append(item.session)
            if has_signal:
                abstained_with_signal += 1
        # 一份旧标签被硬拆成两个新主题：仅针对真正的一对多 bridges（子主题 >=2），
        # 且只对有完整真值（定稿会话）的判断做此审计。
        if item.state is ReviewState.FINALIZED:
            for parent, children in records.bridges().items():
                if len(children) < 2:
                    continue
                child_set = set(children)
                if child_set <= pred.themes and not (child_set <= truth):
                    split_errors.append(
                        {
                            "session": item.session,
                            "bridge_parent": parent,
                            "predicted": sorted(pred.themes),
                            "truth": sorted(truth),
                            "note": "旧标签被复制到一对多的两个子主题，违反不可拆分规则",
                        }
                    )

        if pred.themes == truth:
            exact_matches += 1
        union = pred.themes | truth
        jaccard_sum += (
            len(pred.themes & truth) / len(union) if union else 1.0
        )

        for theme in themes:
            in_pred = theme in pred.themes
            in_truth = theme in truth
            if in_truth:
                counters[theme]["support"] += 1
                weighted_hits[theme] += weight
            if in_pred and in_truth:
                counters[theme]["tp"] += 1
            elif in_pred and not in_truth:
                counters[theme]["fp"] += 1
            elif in_truth and not in_pred:
                counters[theme]["fn"] += 1
            if in_pred:
                weighted_pred[theme] += weight

        joint_dist[truth] = joint_dist.get(truth, 0.0) + weight

    label_scores = {
        theme: LabelScore(
            label=theme,
            true_positive=counters[theme]["tp"],
            false_positive=counters[theme]["fp"],
            false_negative=counters[theme]["fn"],
            support=counters[theme]["support"],
        )
        for theme in themes
    }
    truth_prev = {
        theme: (weighted_hits[theme] / weight_total if weight_total else 0.0)
        for theme in themes
    }
    pred_prev = {
        theme: (weighted_pred[theme] / weight_total if weight_total else 0.0)
        for theme in themes
    }
    prevalence_err = {
        theme: abs(pred_prev[theme] - truth_prev[theme]) for theme in themes
    }
    joint_distribution = {
        tuple(sorted(theme_set)): (w / weight_total if weight_total else 0.0)
        for theme_set, w in joint_dist.items()
    }
    return ScoreReport(
        method=method,
        split=split.name,
        scope_note=split.scope_note,
        n_scored=len(tested),
        n_with_truth=len(tested),
        label_scores=label_scores,
        exact_joint_match=exact_matches / len(tested) if tested else 0.0,
        mean_jaccard=jaccard_sum / len(tested) if tested else 0.0,
        abstention_rate=abstained_with_signal / len(tested) if tested else 0.0,
        empirical_pending_rate=len(pending) / len(tested) if tested else 0.0,
        abstained_sessions=tuple(sorted(abstained)),
        pending_sessions=tuple(sorted(pending)),
        weighted_prevalence=pred_prev,
        truth_weighted_prevalence=truth_prev,
        prevalence_abs_error=prevalence_err,
        sum_of_marginal_columns=sum(pred_prev.values()),
        truth_sum_of_marginal_columns=sum(truth_prev.values()),
        joint_set_distribution=joint_distribution,
        split_label_errors=tuple(split_errors),
        n_abstained_with_signal=abstained_with_signal,
    )


def macro_f1(report: ScoreReport) -> float:
    values = [score.f1 for score in report.label_scores.values() if score.f1 is not None]
    return sum(values) / len(values) if values else 0.0


def micro_f1(report: ScoreReport) -> float | None:
    tp = sum(s.true_positive for s in report.label_scores.values())
    fp = sum(s.false_positive for s in report.label_scores.values())
    fn = sum(s.false_negative for s in report.label_scores.values())
    if tp + fp == 0 or tp + fn == 0:
        return None
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) if p + r else 0.0
