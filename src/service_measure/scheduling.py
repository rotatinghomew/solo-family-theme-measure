"""复核排班与下一批复核的决策导向排序。

运行口径：
- capacity 是每名复核人每个时段（site/day）可承担的"会话数"；
- 同一家庭一次复核可同时解开多项议题，但只占用一次该时段的可用案例容量；
- 双语会话必须由两名双语复核人分别确认，同一个人不能占两个复核位置；
- 普通会话可由任何复核人（含双语复核人）承担；双语会话只能由双语复核人承担；
- prebooked 会话继续占用原站点时段名额，约定好的复核不为追赶其他主题被取消；
- 下一批复核按"能否改变当前材料决策"排序（决策价值），
  不按摘要条数或单标签置信度排队。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .editorial import HouseholdState, build_materials, material_covers
from .facts import SessionFacts
from .io import Records


@dataclass(frozen=True)
class ShiftSlot:
    reviewer: str
    dialect: str
    site: str
    day: int
    capacity: int


@dataclass
class SiteCapacity:
    site: str
    day: int
    # 每名复核人剩余的会话容量；普通/双语会话都从这里按人占用。
    reviewer_slots: dict[str, int]
    bilingual_reviewers: tuple[str, ...]
    reserved: list[str] = field(default_factory=list)

    def available_reviewers(self, language: str) -> list[str]:
        names = [name for name, left in self.reviewer_slots.items() if left > 0]
        if language == "双语":
            return [name for name in names if name in self.bilingual_reviewers]
        return names

    def can_finalize(self, language: str) -> bool:
        if language == "双语":
            # 需要两名不同的双语复核人各出一个位置。
            return len(self.available_reviewers("双语")) >= 2
        return bool(self.available_reviewers("普通"))

    def consume(self, language: str) -> tuple[str, ...]:
        chosen = self.available_reviewers(language)
        if language == "双语":
            if len(chosen) < 2:
                raise RuntimeError("双语会话需要两名不同的双语复核人")
            chosen = chosen[:2]
        else:
            if not chosen:
                raise RuntimeError("普通会话容量不足")
            chosen = chosen[:1]
        for name in chosen:
            self.reviewer_slots[name] -= 1
        return tuple(chosen)


def build_slots(records: Records) -> list[ShiftSlot]:
    return [
        ShiftSlot(
            reviewer=row["reviewer"],
            dialect=row["dialect"],
            site=row["site"],
            day=row["day"],
            capacity=int(row["capacity"]),
        )
        for row in records.shift
    ]


def build_site_capacity(records: Records) -> dict[str, SiteCapacity]:
    """每个站点按其实际排班日聚合容量；同一复核人跨站点的行各自独立。"""
    grouped: dict[str, dict] = {}
    for slot in build_slots(records):
        bucket = grouped.setdefault(
            slot.site,
            {"day": slot.day, "slots": {}, "bilingual": set()},
        )
        bucket["slots"][slot.reviewer] = (
            bucket["slots"].get(slot.reviewer, 0) + slot.capacity
        )
        if slot.dialect == "双语":
            bucket["bilingual"].add(slot.reviewer)
    return {
        site: SiteCapacity(
            site=site,
            day=bucket["day"],
            reviewer_slots=dict(bucket["slots"]),
            bilingual_reviewers=tuple(sorted(bucket["bilingual"])),
        )
        for site, bucket in grouped.items()
    }


def reserve_prebooked(
    records: Records,
    capacity: dict[str, SiteCapacity],
    facts: dict[str, SessionFacts],
) -> tuple[dict[str, SiteCapacity], list[str]]:
    """prebooked 会话继续占用其站点时段名额，约定复核不被其他主题挤掉。

    若约定的双语会话在该站点凑不齐两名双语复核人，名额仍保留、不上线新任务，
    并把 staffing 缺口作为可证约束返回，而不是静默放行。
    """
    blockers: list[str] = []
    for session in records.prebooked:
        item = facts.get(session)
        if item is None:
            continue
        bucket = capacity.get(item.site)
        if bucket is None:
            blockers.append(f"{session}：站点 {item.site} 无排班，约定复核无法落地")
            continue
        bucket.reserved.append(session)
        if bucket.can_finalize(item.language):
            bucket.consume(item.language)
        else:
            blockers.append(
                f"{session}：已约定的{item.language}复核在 {item.site} 站点 day{bucket.day} "
                "凑不齐两名双语复核人；名额保留，不安排新会话顶替"
            )
    return capacity, blockers


@dataclass(frozen=True)
class ReviewCandidate:
    session: str
    household: str
    site: str
    language: str
    unresolved_themes: tuple[str, ...]
    decision_value: float
    pivotal_combos: tuple[str, ...]
    reason: str


def decision_value(
    item: SessionFacts,
    states_by_hh: dict[str, HouseholdState],
    materials: list,
    floors: dict[str, float],
    totals: dict[str, float],
) -> ReviewCandidate:
    """一个未决会话的复核能改变多少材料-站点达标判断，按权重计价值。"""
    state = states_by_hh.get(item.household)
    pivotal: list[str] = []
    value = 0.0
    if state is None:
        return ReviewCandidate(
            item.session, item.household, item.site, item.language,
            tuple(sorted(item.unresolved)), 0.0, (), "无抽样框状态",
        )

    for material in materials:
        required = set(material.required)
        missing = required - state.confirmed
        # 只有当未决主题恰好是该材料还缺的主题时，复核结论才可能改变覆盖。
        if missing and missing <= set(item.unresolved):
            if material_covers(material, state, optimistic=True):
                weight_share = state.weight / totals.get(state.site, 1.0)
                if state.site in floors:
                    pivotal.append(material.material)
                    value += weight_share
    reason = (
        "未决主题可翻转 " + ("、".join(pivotal) if pivotal else "无在选材料")
        if pivotal
        else "未决主题不影响任何在选材料的覆盖判定"
    )
    return ReviewCandidate(
        session=item.session,
        household=item.household,
        site=item.site,
        language=item.language,
        unresolved_themes=tuple(sorted(item.unresolved)),
        decision_value=value,
        pivotal_combos=tuple(sorted(pivotal)),
        reason=reason,
    )


def rank_next_batch(
    records: Records,
    facts: dict[str, SessionFacts],
    states: list[HouseholdState],
    period: int,
) -> tuple[list[ReviewCandidate], dict[str, SiteCapacity], list[str]]:
    """返回按决策价值排序的候选、预留后的容量、以及受容量限制无法定稿的说明。"""
    materials = build_materials(records)
    totals: dict[str, float] = {}
    for row in records.frame:
        if row.period == period:
            totals[row.site] = totals.get(row.site, 0.0) + row.weight
    states_by_hh = {state.household: state for state in states}

    candidates = [
        decision_value(item, states_by_hh, materials, records.editorial_floors, totals)
        for item in facts.values()
        if item.period == period and item.usable and item.unresolved
    ]
    candidates.sort(key=lambda c: (-c.decision_value, c.site, c.session))

    capacity = build_site_capacity(records)
    capacity, reserve_blockers = reserve_prebooked(records, capacity, facts)
    reserved_sessions = set(records.prebooked)

    blockers: list[str] = list(reserve_blockers)
    for candidate in candidates:
        if candidate.session in reserved_sessions:
            # 约定复核的约束已在 reserve_blockers 中说明，不重复报警。
            continue
        bucket = capacity.get(candidate.site)
        if bucket is None:
            blockers.append(f"{candidate.session}：站点 {candidate.site} 无排班")
            continue
        if not bucket.can_finalize(candidate.language):
            blockers.append(
                f"{candidate.session}：{candidate.site} 站点 day{bucket.day} "
                + (
                    "缺两名双语复核人（同一人不能占两个复核位置）"
                    if candidate.language == "双语"
                    else "会话容量不足"
                )
            )
    return candidates, capacity, blockers


def assign_batch(
    capacity: dict[str, SiteCapacity],
    candidates: list[ReviewCandidate],
) -> list[ReviewCandidate]:
    """在容量约束内按决策价值依次定稿；双语会话一次只占一户容量。"""
    assigned: list[ReviewCandidate] = []
    for candidate in candidates:
        bucket = capacity.get(candidate.site)
        if bucket is None or not bucket.can_finalize(candidate.language):
            continue
        bucket.consume(candidate.language)
        assigned.append(candidate)
    return assigned
