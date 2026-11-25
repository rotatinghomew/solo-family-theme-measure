"""简报口径：加权多主题分布与不可相加栏目说明。

错误复现与修正：
- 不同主题的边际占比来自存在多主题家庭的联合分布，逐栏相加会重复计数
  （边际占比之和 > 1）。简报以"联合主题集合"为主表，边际占比只作参考列；
- 词典换版后某主题"消失"时，同时展示 bridges 去向与未决比例，
  说明它可能只是被拆到两栏或处于未决，而非困难不存在；
- 未回访（responded=False）与"确证没有障碍"分开报告；
  未同意摘要不进入任何占比。
"""

from __future__ import annotations

from dataclasses import dataclass

from .facts import SessionFacts, ReviewState
from .io import Records
from .taxonomy import active_themes, snapshot


@dataclass(frozen=True)
class PeriodBriefing:
    period: int
    taxonomy: int
    active_themes: tuple[str, ...]
    withdrawn: tuple[str, ...]
    frame_weight_total: float
    responded_weight: float
    nonresponse_weight: float
    usable_weight: float
    excluded_no_consent_weight: float
    joint_distribution: dict[tuple[str, ...], float]
    marginal_prevalence: dict[str, float]
    marginal_sum: float
    pending_weight: float
    pending_rate: float
    notes: tuple[str, ...]


def build_briefing(
    records: Records,
    facts: dict[str, SessionFacts],
    period: int,
) -> PeriodBriefing:
    snap = snapshot(records, period)
    themes = active_themes(records, period)

    frame_rows = [row for row in records.frame if row.period == period]
    frame_total = sum(row.weight for row in frame_rows)
    responded_w = sum(row.weight for row in frame_rows if row.responded)
    nonresponse_w = frame_total - responded_w

    period_sessions = [item for item in facts.values() if item.period == period]
    usable = [item for item in period_sessions if item.usable]
    no_consent_w = sum(
        _household_weight(records, period, item.household)
        for item in period_sessions
        if not item.usable
    )
    usable_weight = sum(
        _household_weight(records, period, item.household) for item in usable
    )

    # 联合主题集合（确认事实口径；未决主题不冒充确认）。
    joint_weight: dict[frozenset[str], float] = {}
    marginal = {theme: 0.0 for theme in themes}
    pending_weight = 0.0
    for item in usable:
        weight = _household_weight(records, period, item.household)
        confirmed = frozenset(item.confirmed & set(themes))
        joint_weight[confirmed] = joint_weight.get(confirmed, 0.0) + weight
        for theme in confirmed:
            marginal[theme] += weight
        if item.state is ReviewState.PENDING:
            pending_weight += weight

    joint_distribution = {
        tuple(sorted(theme_set)): (w / usable_weight if usable_weight else 0.0)
        for theme_set, w in sorted(joint_weight.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
    }
    marginal_prevalence = {
        theme: (marginal[theme] / usable_weight if usable_weight else 0.0)
        for theme in themes
    }
    notes: list[str] = []
    notes.append(
        "各主题边际占比来自联合分布，存在多主题家庭，不可逐栏相加；"
        f"本期边际占比之和={sum(marginal_prevalence.values()):.3f}"
        "（等于 1 仅当每户至多一个主题，偏离 1 即存在联合/重复计数效应）。"
    )
    if snap.withdrawn:
        notes.append(
            f"主题 {'、'.join(snap.withdrawn)} 自第 {snap.period} 期起撤回，不进入本期口径；"
            "已采用材料的历史依据仍可查，未发任务按新口径核定。"
        )
    if period >= records.revision.active_from:
        children = snap.bridges.get("时段", ())
        if children:
            notes.append(
                "旧主题'时段'为一对多关系（"
                + "、".join(children)
                + "）：无法区分子标签的旧标签保持未决，未向两栏各记一份。"
            )
    notes.append(
        f"未回访加权占框={nonresponse_w / frame_total:.3f}，"
        "未回访不等于确证无障碍；未同意摘要未参与任何占比。"
    )

    return PeriodBriefing(
        period=period,
        taxonomy=snap.version,
        active_themes=themes,
        withdrawn=snap.withdrawn,
        frame_weight_total=frame_total,
        responded_weight=responded_w,
        nonresponse_weight=nonresponse_w,
        usable_weight=usable_weight,
        excluded_no_consent_weight=no_consent_w,
        joint_distribution=joint_distribution,
        marginal_prevalence=marginal_prevalence,
        marginal_sum=sum(marginal_prevalence.values()),
        pending_weight=pending_weight,
        pending_rate=(pending_weight / usable_weight if usable_weight else 0.0),
        notes=tuple(notes),
    )


def _household_weight(records: Records, period: int, household: str) -> float:
    for row in records.frame:
        if row.period == period and row.household == household:
            return row.weight
    raise KeyError((period, household))
