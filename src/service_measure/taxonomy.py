"""词典的时期版本管理。

关键口径：
- 每个时期只有一个生效词典版本（数据中 1-3 期为 v1，4-5 期为 v2）；
- 修订（revision）在 active_from 当期生效：撤回主题不再用于该口径的简报与材料判定，
  但 revision.approved_sessions 里已采用材料的依据仍可查（不追溯抹除）；
- bridges 给出旧主题 -> 新主题的跨版关系：
  * 一对一可以在有复核事实支持时沿用；
  * 一对多（如 时段 -> 线上时段/现场时段）表示"不能确定子标签时不可复制父标签"，
    即严禁把一份旧标签拆成两份、各记一次。
"""

from __future__ import annotations

from dataclasses import dataclass

from .io import Records


@dataclass(frozen=True)
class TaxonomySnapshot:
    """某一时期口径下实际可用的词典快照。"""

    period: int
    version: int
    themes: tuple[str, ...]
    withdrawn: tuple[str, ...]
    bridges: dict[str, tuple[str, ...]]

    def contains(self, theme: str) -> bool:
        return theme in self.themes and theme not in self.withdrawn


def snapshot(records: Records, period: int) -> TaxonomySnapshot:
    version = records.period_taxonomy(period)
    withdrawn: list[str] = []
    revision = records.revision
    if period >= revision.active_from and revision.withdrawn_theme in records.themes(version):
        withdrawn.append(revision.withdrawn_theme)
    bridges = {
        parent: tuple(children)
        for parent, children in records.bridges().items()
    }
    return TaxonomySnapshot(
        period=period,
        version=version,
        themes=records.themes(version),
        withdrawn=tuple(withdrawn),
        bridges=bridges,
    )


def active_themes(records: Records, period: int) -> tuple[str, ...]:
    snap = snapshot(records, period)
    return tuple(theme for theme in snap.themes if theme not in snap.withdrawn)


def carry_forward(records: Records, old_theme: str, target_period: int) -> tuple[str, ...] | None:
    """旧主题在目标时期可追溯的新主题集合。

    返回 None 表示没有跨版关系，不能直接引用旧标签；
    返回单元素元组表示一对一沿用；
    返回多元素元组表示一对多——调用方必须拿到能区分子标签的已复核事实，
    否则只能整体存疑（未决），不得向两个子主题各记一份。
    """

    snap = snapshot(records, target_period)
    children = snap.bridges.get(old_theme)
    if not children:
        return None
    active = tuple(theme for theme in children if theme not in snap.withdrawn)
    return active
