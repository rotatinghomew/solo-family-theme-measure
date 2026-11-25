"""材料组合的覆盖核算与可证决策。

核算口径（对应任务与资料字典）：
- 材料 `all` 是同一家庭需要"同时"具有的全部主题；空列表不作为覆盖规则（直接排除）；
- 同一家庭被多份材料覆盖只记一次：组合覆盖按家庭取并集，不逐份相加；
- 站点最低覆盖率（editorial_floors）的分母是该站点抽样框的全部加权人群，
  不因未回应、未同意或资料撤回而缩小；
- 未决标签、未同意摘要、未回访家庭都不是"确证没有障碍"：
  只给保守（仅确认事实）与乐观（全部未知按可满足处理）两个可证边界，
  会颠倒结论的解释只能作为敏感性展示，不能当作正式选择。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .facts import SessionFacts
from .io import Records
from .taxonomy import active_themes


@dataclass(frozen=True)
class HouseholdState:
    period: int
    household: str
    site: str
    weight: float
    responded: bool
    consent: bool
    confirmed: frozenset[str]
    unresolved: frozenset[str]

    @property
    def unknown_themes(self) -> frozenset[str]:
        """该家庭尚未被证实、也未被否证的主题空间。"""
        return self.unresolved


@dataclass(frozen=True)
class Material:
    material: str
    required: tuple[str, ...]
    cost: int


@dataclass(frozen=True)
class ComboEvaluation:
    materials: tuple[str, ...]
    cost: int
    within_budget: bool
    conservative_site_coverage: dict[str, float]
    optimistic_site_coverage: dict[str, float]
    floors: dict[str, float]
    decision: str  # 可正式采用 / 条件性可能 / 不满足
    pivotal_households: tuple[str, ...]
    floor_gap_conservative: dict[str, float]
    floor_gap_optimistic: dict[str, float]
    blocking_reasons: tuple[str, ...]
    uncertainty_breakdown: dict[str, dict[str, float]]
    duplicate_cover_share: dict[str, float] = field(default_factory=dict)


def build_materials(records: Records) -> list[Material]:
    materials: list[Material] = []
    for item in records.editorial:
        required = tuple(item["all"])
        if not required:
            # 空列表不作为覆盖规则。
            continue
        materials.append(
            Material(material=item["material"], required=required, cost=int(item["cost"]))
        )
    return materials


def build_household_states(
    records: Records,
    facts: dict[str, SessionFacts],
    period: int,
) -> list[HouseholdState]:
    """汇总决策期内每一户的确认/未决主题；一户一行，杜绝跨渠道重复计数。"""
    frame_rows = [row for row in records.frame if row.period == period]
    sessions_by_hh: dict[str, list[SessionFacts]] = {}
    for item in facts.values():
        if item.period == period:
            sessions_by_hh.setdefault(item.household, []).append(item)

    states: list[HouseholdState] = []
    allowed = set(active_themes(records, period))
    for row in frame_rows:
        sessions = sessions_by_hh.get(row.household, [])
        usable = [s for s in sessions if s.usable]
        consent = any(s.consent for s in sessions)
        confirmed: set[str] = set()
        unresolved: set[str] = set()
        for session in usable:
            confirmed |= session.confirmed & allowed
            unresolved |= session.unresolved & allowed
        unresolved -= confirmed
        states.append(
            HouseholdState(
                period=period,
                household=row.household,
                site=row.site,
                weight=row.weight,
                responded=row.responded,
                consent=consent,
                confirmed=frozenset(confirmed),
                unresolved=frozenset(unresolved),
            )
        )
    return states


def material_covers(
    material: Material, state: HouseholdState, optimistic: bool
) -> bool:
    required = set(material.required)
    known = state.confirmed
    if required <= known:
        return True
    if not optimistic:
        return False
    # 乐观边界：未决主题、未同意/未回访导致的整体未知都按可满足处理。
    unknown_space = set(state.unknown_themes)
    if not state.responded or not state.consent:
        # 未回访与未同意：没有任何可反驳的事实，整户进入"可能"空间。
        return True
    return required <= (known | unknown_space)


def combo_covered(
    materials: tuple[Material, ...],
    states: list[HouseholdState],
    optimistic: bool,
) -> set[str]:
    covered: set[str] = set()
    for state in states:
        if any(material_covers(material, state, optimistic) for material in materials):
            covered.add(state.household)
    return covered


def site_totals(records: Records, period: int) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in records.frame:
        if row.period == period:
            totals[row.site] = totals.get(row.site, 0.0) + row.weight
    return totals


def evaluate_combos(
    records: Records, states: list[HouseholdState], period: int
) -> list[ComboEvaluation]:
    materials = build_materials(records)
    totals = site_totals(records, period)
    floors = records.editorial_floors
    budget = records.editorial_budget
    results: list[ComboEvaluation] = []

    # 单份材料各自覆盖的家庭，用于展示"逐份相加"会重复记多少。
    solo_cover = {
        material.material: combo_covered((material,), states, optimistic=False)
        for material in materials
    }

    for r in range(1, len(materials) + 1):
        for combo in itertools.combinations(materials, r):
            cost = sum(material.cost for material in combo)
            within = cost <= budget
            conservative = combo_covered(combo, states, optimistic=False)
            optimistic = combo_covered(combo, states, optimistic=True)

            conservative_cov = {
                site: sum(
                    state.weight
                    for state in states
                    if state.site == site and state.household in conservative
                )
                / totals[site]
                for site in totals
            }
            optimistic_cov = {
                site: sum(
                    state.weight
                    for state in states
                    if state.site == site and state.household in optimistic
                )
                / totals[site]
                for site in totals
            }

            passes_low = all(conservative_cov[site] >= floors[site] for site in floors)
            passes_high = all(optimistic_cov[site] >= floors[site] for site in floors)
            if passes_low and within:
                decision = "可正式采用"
            elif passes_high and within:
                decision = "条件性可能"
            else:
                decision = "不满足"

            gap_low = {
                site: round(floors[site] - conservative_cov[site], 4)
                for site in floors
                if conservative_cov[site] < floors[site]
            }
            gap_high = {
                site: round(floors[site] - optimistic_cov[site], 4)
                for site in floors
                if optimistic_cov[site] < floors[site]
            }
            blocking: list[str] = []
            if not within:
                blocking.append(f"费用 {cost} 超过预算 {budget}")
            for site, gap in gap_high.items():
                blocking.append(
                    f"{site} 站点即使按最乐观解释仍差 {gap:.3f} 才到最低覆盖 {floors[site]}"
                )
            for site, gap in gap_low.items():
                if site not in gap_high:
                    blocking.append(
                        f"{site} 站点保守口径差 {gap:.3f}，乐观口径可达标——结论取决于未决/缺失回应，不可正式确认"
                    )

            # 把乐观-保守之间的"可能覆盖"按不确定性来源拆开，便于策划者判断该补什么。
            breakdown: dict[str, dict[str, float]] = {}
            for site in totals:
                buckets = {"未回访": 0.0, "未同意": 0.0, "未决标签": 0.0}
                for state in states:
                    if state.site != site or state.household not in (optimistic - conservative):
                        continue
                    if not state.responded:
                        buckets["未回访"] += state.weight
                    elif not state.consent:
                        buckets["未同意"] += state.weight
                    else:
                        buckets["未决标签"] += state.weight
                breakdown[site] = {key: round(w / totals[site], 4) for key, w in buckets.items()}

            pivotal = tuple(
                sorted(
                    state.household
                    for state in states
                    if state.household in optimistic - conservative
                    and _could_flip(state, combo, floors, states, totals)
                )
            )

            # 重复覆盖：同一户被多份材料记上的权重占组合覆盖的比例。
            hit_counts: dict[str, int] = {}
            for material in combo:
                for household in solo_cover[material.material]:
                    hit_counts[household] = hit_counts.get(household, 0) + 1
            dup_weight = sum(
                state.weight
                for state in states
                if hit_counts.get(state.household, 0) >= 2
            )
            base_weight = sum(
                state.weight for state in states if state.household in conservative
            )
            duplicate_share = (
                {"/".join(m.material for m in combo): dup_weight / base_weight}
                if base_weight
                else {}
            )

            results.append(
                ComboEvaluation(
                    materials=tuple(material.material for material in combo),
                    cost=cost,
                    within_budget=within,
                    conservative_site_coverage=conservative_cov,
                    optimistic_site_coverage=optimistic_cov,
                    floors=dict(floors),
                    decision=decision,
                    pivotal_households=pivotal,
                    floor_gap_conservative=gap_low,
                    floor_gap_optimistic=gap_high,
                    blocking_reasons=tuple(blocking),
                    uncertainty_breakdown=breakdown,
                    duplicate_cover_share=duplicate_share,
                )
            )

    results.sort(key=lambda item: (item.cost, item.materials))
    return results


def _could_flip(
    state: HouseholdState,
    combo: tuple[Material, ...],
    floors: dict[str, float],
    states: list[HouseholdState],
    totals: dict[str, float],
) -> bool:
    """该户的口径解释是否足以改变其所在站点的达标判断。"""
    site = state.site
    if site not in floors:
        return False
    base = sum(
        other.weight
        for other in states
        if other.site == site
        and other.household
        in combo_covered(combo, states, optimistic=False)
    )
    low, high = base / totals[site], (base + state.weight) / totals[site]
    return low < floors[site] <= high


def overlap_matrix(records: Records, states: list[HouseholdState]) -> dict[str, dict[str, float]]:
    """两两材料在确认事实口径下的覆盖家庭重叠（加权），证明不可逐份相加。"""
    materials = build_materials(records)
    cover = {
        material.material: combo_covered((material,), states, optimistic=False)
        for material in materials
    }
    weight = {state.household: state.weight for state in states}
    matrix: dict[str, dict[str, float]] = {}
    for left in materials:
        matrix[left.material] = {}
        for right in materials:
            overlap = cover[left.material] & cover[right.material]
            matrix[left.material][right.material] = sum(
                weight[hh] for hh in overlap
            )
    return matrix
