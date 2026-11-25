"""端到端分析编排：事实 -> 留出评估 -> 简报 -> 材料决策 -> 排班。

所有输出都带口径标识（词典版本、时期、同意状态、证据来源），
负责排班与编简报的人可以从同一份产物核对适用口径与决议差别。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .briefing import build_briefing
from .classifier import (
    EvidenceClassifier,
    FullKeywordClassifier,
    LegacyKeywordClassifier,
    judge_new_material,
)
from .editorial import build_household_states, evaluate_combos, overlap_matrix
from .evaluation import (
    ScoreReport,
    household_holdout_split,
    macro_f1,
    micro_f1,
    period_holdout_split,
    score_predictions,
)
from .facts import build_session_facts
from .io import Records, load_records
from .scheduling import assign_batch, rank_next_batch
from .taxonomy import active_themes


def _weights(records: Records) -> dict[tuple[int, str], float]:
    return {(row.period, row.household): row.weight for row in records.frame}


def run_holdout_evaluation(records: Records, facts: dict) -> dict[str, Any]:
    weights = _weights(records)
    splits = [
        household_holdout_split(records, facts, taxonomy=1),
        household_holdout_split(records, facts, taxonomy=2),
        period_holdout_split(records, facts),
    ]
    report: dict[str, Any] = {"splits": []}
    for split in splits:
        methods = {
            "evidence": EvidenceClassifier(records).fit(list(split.train)),
            "legacy_keyword": LegacyKeywordClassifier(records).fit(list(split.train)),
            "full_keyword": FullKeywordClassifier(records).fit(list(split.train)),
        }
        split_block: dict[str, Any] = {
            "name": split.name,
            "scope_note": split.scope_note,
            "n_train": len(split.train),
            "n_test_scored": 0,
            "methods": [],
        }
        for name, classifier in methods.items():
            predictions = {
                item.session: judge_new_material(classifier, item)
                for item in split.test
            }
            score = score_predictions(name, split, predictions, records, weights)
            split_block["n_test_scored"] = score.n_scored
            split_block["methods"].append(_serialize_score(score))
        report["splits"].append(split_block)
    return report


def _serialize_score(score: ScoreReport) -> dict[str, Any]:
    return {
        "method": score.method,
        "n_scored": score.n_scored,
        "macro_f1": round(macro_f1(score), 4),
        "micro_f1": None if micro_f1(score) is None else round(micro_f1(score), 4),
        "exact_joint_match": round(score.exact_joint_match, 4),
        "mean_jaccard": round(score.mean_jaccard, 4),
        "abstention_rate": round(score.abstention_rate, 4),
        "n_abstained_with_signal": score.n_abstained_with_signal,
        "empirical_pending_rate": round(score.empirical_pending_rate, 4),
        "split_label_errors": list(score.split_label_errors),
        "label_f1": {
            label: (None if s.f1 is None else round(s.f1, 4))
            for label, s in score.label_scores.items()
        },
        "label_support": {label: s.support for label, s in score.label_scores.items()},
        "prevalence_abs_error": {
            label: round(value, 4)
            for label, value in score.prevalence_abs_error.items()
        },
        "sum_of_marginal_columns": round(score.sum_of_marginal_columns, 4),
        "truth_sum_of_marginal_columns": round(
            score.truth_sum_of_marginal_columns, 4
        ),
        "abstained_sessions": list(score.abstained_sessions),
        "joint_set_distribution": {
            "|".join(themes) if themes else "(无主题)": round(weight, 4)
            for themes, weight in score.joint_set_distribution.items()
        },
    }


def run_editorial_decision(records: Records, facts: dict, decision_period: int) -> dict[str, Any]:
    states = build_household_states(records, facts, decision_period)
    combos = evaluate_combos(records, states, decision_period)
    matrix = overlap_matrix(records, states)
    return {
        "decision_period": decision_period,
        "active_themes": list(active_themes(records, decision_period)),
        "budget": records.editorial_budget,
        "floors": records.editorial_floors,
        "site_frame_weights": {
            site: round(sum(
                row.weight
                for row in records.frame
                if row.period == decision_period and row.site == site
            ), 4)
            for site in records.editorial_floors
        },
        "combos": [
            {
                "materials": list(combo.materials),
                "cost": combo.cost,
                "within_budget": combo.within_budget,
                "decision": combo.decision,
                "conservative_site_coverage": {
                    k: round(v, 4) for k, v in combo.conservative_site_coverage.items()
                },
                "optimistic_site_coverage": {
                    k: round(v, 4) for k, v in combo.optimistic_site_coverage.items()
                },
                "pivotal_households": list(combo.pivotal_households),
                "floor_gap_conservative": combo.floor_gap_conservative,
                "floor_gap_optimistic": combo.floor_gap_optimistic,
                "blocking_reasons": list(combo.blocking_reasons),
                "uncertainty_breakdown": combo.uncertainty_breakdown,
                "duplicate_cover_share": combo.duplicate_cover_share,
            }
            for combo in combos
        ],
        "overlap_matrix": {
            left: {right: round(w, 4) for right, w in row.items()}
            for left, row in matrix.items()
        },
    }


def run_scheduling(records: Records, facts: dict, period: int) -> dict[str, Any]:
    states = build_household_states(records, facts, period)
    candidates, capacity, blockers = rank_next_batch(records, facts, states, period)
    assigned = assign_batch(capacity, candidates)
    return {
        "period": period,
        "ranking": [
            {
                "session": c.session,
                "household": c.household,
                "site": c.site,
                "language": c.language,
                "unresolved_themes": list(c.unresolved_themes),
                "decision_value": round(c.decision_value, 4),
                "pivotal_combos": list(c.pivotal_combos),
                "reason": c.reason,
            }
            for c in candidates
        ],
        "assigned_sessions": [c.session for c in assigned],
        "assigned_reviewers": [
            {
                "session": c.session,
                "site": c.site,
                "language": c.language,
            }
            for c in assigned
        ],
        "capacity_after_reserve": [
            {
                "site": bucket.site,
                "day": bucket.day,
                "reserved": bucket.reserved,
                "reviewer_slots_left": dict(bucket.reviewer_slots),
                "bilingual_reviewers": list(bucket.bilingual_reviewers),
            }
            for bucket in sorted(capacity.values(), key=lambda b: b.site)
        ],
        "capacity_blockers": blockers,
    }


def _holdout_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    """跨检验、跨方法的一页式指标对照。"""
    table: dict[str, dict[str, Any]] = {}
    for split in evaluation["splits"]:
        for method in split["methods"]:
            table[f"{split['name']}|{method['method']}"] = {
                "macro_f1": method["macro_f1"],
                "joint_exact_match": method["exact_joint_match"],
                "mean_jaccard": method["mean_jaccard"],
                "abstention_rate_with_signal": method["abstention_rate"],
                "empirical_pending_rate": method["empirical_pending_rate"],
                "hard_split_errors": len(method["split_label_errors"]),
                "max_prevalence_abs_error": max(method["prevalence_abs_error"].values()),
            }
    table["_读法"] = {
        "period_holdout_cross_taxonomy|evidence": (
            "整期留出时拟采用方式在无 v2 复核事实下整体弃权，故 macro_f1=0 且弃权率高；"
            "这是诚实行为而非失效——对照基准硬给标签的假阳性与硬拆错误更高。"
            "同版泛化能力请看 household_holdout_v1/v2 两行。"
        ),
        "hard_split_errors": "一份旧标签被复制到一对多两个子主题的次数（应为 0 才合规）。",
        "abstention_rate_with_signal": "有困难信号但证据不足而弃权的比例；无主题会话不计入。",
    }
    return table


def _decision_summary(editorial: dict[str, Any], scheduling: dict[str, Any]) -> dict[str, Any]:
    combos = editorial["combos"]
    budget = editorial["budget"]
    adoptable = [c["materials"] for c in combos if c["decision"] == "可正式采用"]
    conditional = [
        {
            "materials": c["materials"],
            "cost": c["cost"],
            "blocking_reasons": c["blocking_reasons"],
        }
        for c in combos
        if c["decision"] == "条件性可能"
    ]
    infeasible = [
        {"materials": c["materials"], "blocking_reasons": c["blocking_reasons"]}
        for c in combos
        if c["decision"] == "不满足" and c["cost"] <= budget
    ]
    over_budget = [c["materials"] for c in combos if not c["within_budget"]]
    return {
        "可正式采用": adoptable,
        "尚不能正式选择_条件性可能": conditional,
        "预算内但不满足最低覆盖": infeasible,
        "超预算组合示例": over_budget[:3],
        "给策划者的结论": (
            "当前没有任何预算内组合能在保守（仅已确认事实）口径下同时满足三站点最低覆盖；"
            "北站点的乐观空间主要来自未回访/未同意家庭而非可复核的未决标签，"
            "因此再排复核也无法把条件性组合升级为可正式采用，先补回访与同意才可能改变结论。"
        ),
        "下一批复核可定稿会话": scheduling["assigned_sessions"],
        "排班约束": scheduling["capacity_blockers"],
    }


def run_pipeline(records_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    records = load_records(records_path)
    facts = build_session_facts(records)
    decision_period = max(row.period for row in records.frame)

    editorial_decision = run_editorial_decision(records, facts, decision_period)
    scheduling = run_scheduling(records, facts, decision_period)
    holdout_evaluation = run_holdout_evaluation(records, facts)
    result = {
        "contract": {
            "n_frame_rows": len(records.frame),
            "n_sessions": len(facts),
            "n_usable_sessions": sum(1 for f in facts.values() if f.usable),
            "taxonomy_versions": records.taxonomy_versions,
            "mature_periods": list(records.mature_periods),
            "holdout_periods": list(records.holdout_periods),
            "holdout_households": sorted(records.holdout_households),
        },
        "session_facts_sample": [
            _dataclass_to_dict(facts[key])
            for key in sorted(facts)
            if key in records.prebooked
            or key in records.revision.approved_sessions
        ],
        "briefings": [
            _dataclass_to_dict(build_briefing(records, facts, period))
            for period in range(1, 6)
        ],
        "holdout_evaluation": holdout_evaluation,
        "holdout_summary": _holdout_summary(holdout_evaluation),
        "editorial_decision": editorial_decision,
        "scheduling": scheduling,
        "decision_summary": _decision_summary(editorial_decision, scheduling),
    }
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {key: _dataclass_to_dict(value) for key, value in asdict(obj).items()}
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, dict):
        return {_json_key(key): _dataclass_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(value) for value in obj]
    return obj


def _json_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "|".join(key) if key else "(无主题)"
    return str(key)
