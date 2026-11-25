"""从双人复核与联合复核结论构建可追溯的会话事实。

口径（对应资料字典与任务约定）：
- 只有符合当前词典、且两名复核人共同定稿的标签才是"已确认事实"；
- `待合议` 不等同两人从未看过：分歧标签计入 unresolved，交集标签仍是已确认事实；
- joint_review 记部分共识：agreement 进入已确认事实，unresolved 与复核分歧合并；
  已确认事实不被后续未决标签覆盖；
- 未经同意（consent=False）的摘要不进入任何分析事实，仅在数据层可查；
- 同一 session 的多渠道转写归并为同一户、同一次会话，绝不重复计数；
- 每条事实都带证据链（复核行/联合复核条目/摘要 ref），便于追溯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .io import Records, ReviewRow, SummaryRow


class ReviewState(str, Enum):
    NOT_REVIEWED = "无复核"
    FINALIZED = "定稿"
    PENDING = "待合议"


@dataclass(frozen=True)
class SessionFacts:
    session: str
    period: int
    household: str
    site: str
    language: str
    consent: bool
    channels: tuple[str, ...]
    clauses: tuple[str, ...]
    taxonomy: int
    state: ReviewState
    confirmed: frozenset[str]
    unresolved: frozenset[str]
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        """未经同意的摘要一律不用于分析。"""
        return self.consent

    @property
    def decided(self) -> bool:
        """是否存在任何未决标签。"""
        return not self.unresolved


def _session_indexes(records: Records):
    summaries_by_session: dict[str, list[SummaryRow]] = {}
    for row in records.summaries:
        summaries_by_session.setdefault(row.session, []).append(row)

    reviews_by_session: dict[str, list[ReviewRow]] = {}
    for row in records.reviews:
        reviews_by_session.setdefault(row.session, []).append(row)

    joint_by_session = {item.session: item for item in records.joint_review}
    return summaries_by_session, reviews_by_session, joint_by_session


def build_session_facts(records: Records) -> dict[str, SessionFacts]:
    summaries_by_session, reviews_by_session, joint_by_session = _session_indexes(records)
    facts: dict[str, SessionFacts] = {}

    for session, rows in summaries_by_session.items():
        first = rows[0]
        # 同一 session 的渠道转写必须指向同一户/期/站点/语言/同意状态。
        for row in rows[1:]:
            if (row.period, row.household, row.site, row.language, row.consent) != (
                first.period,
                first.household,
                first.site,
                first.language,
                first.consent,
            ):
                raise ValueError(f"会话 {session} 的多渠道转写主键不一致")

        consent = all(row.consent for row in rows)
        channels = tuple(sorted({row.channel for row in rows}))
        # 小句去重但保留多主题：同一户在站点、回访重复说同一件事只算一个主题。
        clauses: list[str] = []
        for row in rows:
            for clause in row.clauses:
                if clause not in clauses:
                    clauses.append(clause)

        review_rows = reviews_by_session.get(session, [])
        evidence = tuple(sorted({row.ref for row in rows}))
        confirmed: set[str] = set()
        unresolved: set[str] = set()
        state = ReviewState.NOT_REVIEWED
        taxonomy = records.period_taxonomy(first.period)

        if review_rows:
            if len({row.reviewer for row in review_rows}) != len(review_rows):
                raise ValueError(f"会话 {session} 同一复核人出现多次")
            taxonomy = review_rows[0].taxonomy
            statuses = {row.status for row in review_rows}
            allowed = set(records.themes(taxonomy))
            label_sets = [set(row.labels) for row in review_rows]

            if statuses == {"定稿"}:
                state = ReviewState.FINALIZED
                # 共同定稿：两人标签集合的交集才是确认事实。
                confirmed = set.intersection(*label_sets) if label_sets else set()
                evidence = evidence + (f"review:{session}:定稿",)
            else:
                # 待合议：交集仍为已确认事实，差集为未决。
                state = ReviewState.PENDING
                confirmed = set.intersection(*label_sets) if label_sets else set()
                unresolved = set.union(*label_sets) - confirmed if label_sets else set()
                evidence = evidence + (f"review:{session}:待合议",)

            for row in review_rows:
                if set(row.labels) - allowed:
                    raise ValueError(f"会话 {session} 含词典 v{taxonomy} 之外的标签")

        joint = joint_by_session.get(session)
        if joint is not None:
            evidence += (f"joint_review:{session}",)
            # 部分共识继续有效；未决标签并入，且不能覆盖已确认事实。
            confirmed |= set(joint.agreement)
            unresolved = (unresolved | set(joint.unresolved)) - set(joint.agreement)
            if state == ReviewState.NOT_REVIEWED:
                state = ReviewState.PENDING

        facts[session] = SessionFacts(
            session=session,
            period=first.period,
            household=first.household,
            site=first.site,
            language=first.language,
            consent=consent,
            channels=channels,
            clauses=tuple(clauses),
            taxonomy=taxonomy,
            state=state,
            confirmed=frozenset(confirmed),
            unresolved=frozenset(unresolved),
            evidence=evidence,
        )

    return facts


def frame_index(records: Records) -> dict[tuple[int, str], object]:
    return {(row.period, row.household): row for row in records.frame}


def sessions_of_period(facts: dict[str, SessionFacts], period: int) -> list[SessionFacts]:
    return sorted(
        (item for item in facts.values() if item.period == period),
        key=lambda item: item.session,
    )


def usable_facts(facts: dict[str, SessionFacts]) -> list[SessionFacts]:
    """已同意、可进入分析的会话事实。"""
    return [item for item in facts.values() if item.usable]
