"""读取合成业务记录并校验输入契约。

校验规则对应 README《资料字典》：
- 每条摘要必须能在抽样框中找到同一 (period, household)；
- 每条复核的标签必须属于该复核行所用词典版本；
- 同一 session 可有多条（多渠道）转写；
- 留出整期与成熟（可拟合）时期不得相交。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameRow:
    period: int
    household: str
    site: str
    weight: float
    responded: bool


@dataclass(frozen=True)
class SummaryRow:
    ref: str
    session: str
    period: int
    household: str
    site: str
    channel: str
    consent: bool
    language: str
    summary: str

    @property
    def clauses(self) -> tuple[str, ...]:
        """同一条摘要里的多个小句；多主题家庭在此一次保留多项。"""
        return tuple(part.strip() for part in self.summary.split("；") if part.strip())


@dataclass(frozen=True)
class ReviewRow:
    session: str
    period: int
    taxonomy: int
    reviewer: str
    labels: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class JointReview:
    session: str
    active_from: int
    agreement: tuple[str, ...]
    unresolved: tuple[str, ...]


@dataclass(frozen=True)
class Revision:
    active_from: int
    withdrawn_theme: str
    approved_sessions: tuple[str, ...]


@dataclass(frozen=True)
class Records:
    frame: tuple[FrameRow, ...]
    summaries: tuple[SummaryRow, ...]
    reviews: tuple[ReviewRow, ...]
    definitions: dict[str, Any]
    mature_periods: tuple[int, ...]
    holdout_periods: tuple[int, ...]
    holdout_households: frozenset[str]
    prebooked: tuple[str, ...]
    revision: Revision
    editorial_budget: int
    editorial: tuple[dict[str, Any], ...]
    editorial_floors: dict[str, float]
    joint_review: tuple[JointReview, ...]
    shift: tuple[dict[str, Any], ...]

    @property
    def taxonomy_versions(self) -> tuple[int, ...]:
        return tuple(int(k) for k in self.definitions if k.isdigit())

    def themes(self, taxonomy: int) -> tuple[str, ...]:
        return tuple(self.definitions[str(taxonomy)])

    def bridges(self) -> dict[str, list[str]]:
        return self.definitions["bridges"]

    def period_taxonomy(self, period: int) -> int:
        """数据中时期 1-3 用词典 v1，时期 4-5 用词典 v2。"""
        return 1 if period <= 3 else 2


def load_records(path: str | Path) -> Records:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = _build(payload)
    validate_contract(records)
    return records


def _build(payload: dict[str, Any]) -> Records:
    frame = tuple(
        FrameRow(
            period=item["period"],
            household=item["household"],
            site=item["site"],
            weight=float(item["weight"]),
            responded=bool(item["responded"]),
        )
        for item in payload["frame"]
    )
    summaries = tuple(
        SummaryRow(
            ref=item["ref"],
            session=item["session"],
            period=item["period"],
            household=item["household"],
            site=item["site"],
            channel=item["channel"],
            consent=bool(item["consent"]),
            language=item["language"],
            summary=item["summary"],
        )
        for item in payload["summaries"]
    )
    reviews = tuple(
        ReviewRow(
            session=item["session"],
            period=item["period"],
            taxonomy=int(item["taxonomy"]),
            reviewer=item["reviewer"],
            labels=tuple(item["labels"]),
            status=item["status"],
        )
        for item in payload["reviews"]
    )
    joint = tuple(
        JointReview(
            session=item["session"],
            active_from=item["active_from"],
            agreement=tuple(item["agreement"]),
            unresolved=tuple(item["unresolved"]),
        )
        for item in payload["joint_review"]
    )
    approval = payload["approval"]
    revision = Revision(
        active_from=approval["revision"]["active_from"],
        withdrawn_theme=approval["revision"]["withdrawn_theme"],
        approved_sessions=tuple(approval["revision"]["approved_sessions"]),
    )
    return Records(
        frame=frame,
        summaries=summaries,
        reviews=reviews,
        definitions=payload["definitions"],
        mature_periods=tuple(approval["mature_periods"]),
        holdout_periods=tuple(approval["holdout_periods"]),
        holdout_households=frozenset(approval["holdout_households"]),
        prebooked=tuple(approval["prebooked"]),
        revision=revision,
        editorial_budget=int(payload["editorial_budget"]),
        editorial=tuple(payload["editorial"]),
        editorial_floors=dict(payload["editorial_floors"]),
        joint_review=joint,
        shift=tuple(payload["shift"]),
    )


def validate_contract(records: Records) -> None:
    """与 tests/test_records.py 同一套输入契约，并补充业务一致性检查。"""

    frame_keys = {(row.period, row.household) for row in records.frame}
    for item in records.summaries:
        if (item.period, item.household) not in frame_keys:
            raise ValueError(f"摘要 {item.ref} 在抽样框中找不到对应家庭")

    sessions = {item.session for item in records.summaries}
    theme_sets = {
        version: set(records.definitions[str(version)])
        for version in records.taxonomy_versions
    }
    for item in records.reviews:
        if item.session not in sessions:
            raise ValueError(f"复核 {item.session} 找不到对应会话")
        unknown = set(item.labels) - theme_sets[item.taxonomy]
        if unknown:
            raise ValueError(f"复核 {item.session} 使用了词典 v{item.taxonomy} 之外的标签 {sorted(unknown)}")

    if len(sessions) >= len(records.summaries):
        # 资料字典明确同一 session 可由多渠道转写，数据里应确实存在这种会话。
        raise ValueError("预期存在同一 session 的多渠道转写")

    if set(records.mature_periods) & set(records.holdout_periods):
        raise ValueError("留出整期与成熟时期不得相交")

    # 联合复核结论里的标签必须属于其生效词典。
    for item in records.joint_review:
        version = records.period_taxonomy(item.active_from)
        allowed = theme_sets[version]
        bad = (set(item.agreement) | set(item.unresolved)) - allowed
        if bad:
            raise ValueError(f"联合复核 {item.session} 含词典 v{version} 之外的标签 {sorted(bad)}")

    # 抽样框中同一 (period, household) 只能出现一次。
    if len(frame_keys) != len(records.frame):
        raise ValueError("抽样框存在重复的 (period, household)")
