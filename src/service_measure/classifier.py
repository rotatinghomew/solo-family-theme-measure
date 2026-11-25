"""可追溯的多主题判断器与简单基准。

三种判断方式共用同一套输出协议，便于在留出资料上公平比较：

- 拟采用方式 `EvidenceClassifier`：
  只从"已同意 + 双人共同定稿"的会话学习 小句 -> 主题集合 的映射；
  对学不到证据的小句弃权（abstain），不靠猜测补标签；
  旧词典的一对多关系（时段 -> 线上/现场时段）在没有可区分的新词典复核事实时
  整体弃权，绝不把一份旧标签拆成两份。

- 简单基准 A `LegacyKeywordClassifier`：
  只用 v1 词典期学过的固定关键词，碰到新词典期靠 bridges 硬套，
  复现"改了词典后一类困难看似消失/被硬拆两栏"的错误。

- 简单基准 B `FullKeywordClassifier`：
  把全部关键词写成一张不带时期版本的固定词典、且从不弃权，
  复现"逐标签看似正确、但联合主题与未决比例不可信"的错误。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .facts import SessionFacts, ReviewState
from .io import Records
from .taxonomy import snapshot


@dataclass(frozen=True)
class Prediction:
    session: str
    taxonomy: int
    themes: frozenset[str]          # 该判断方式最终给出的主题
    abstained: frozenset[str]       # 有信号但证据不足、保持未决的主题
    unseen_clauses: tuple[str, ...] # 训练中没有已复核事实支撑的小句
    source: str                     # 判断方式标识，便于追溯

    @property
    def is_abstention(self) -> bool:
        return bool(self.abstained) or bool(self.unseen_clauses)


@dataclass
class EvidenceClassifier:
    """从双人定稿事实学习小句级证据的多主题判断器。"""

    records: Records
    name: str = "evidence"
    clause_themes: dict[int, dict[str, frozenset[str]]] = field(default_factory=dict)
    trained_sessions: tuple[str, ...] = ()

    def fit(self, training_sessions: list[SessionFacts]) -> "EvidenceClassifier":
        # 小句自身携带的主题 = 所有含该小句的定稿会话确认主题的交集。
        # 用并集会把同会话里其他小句的主题错误挂到该小句上（多主题共现污染）。
        learned: dict[int, dict[str, set[str]]] = {}
        seen: dict[int, dict[str, bool]] = {}
        used: list[str] = []
        for item in training_sessions:
            if not item.usable or item.state is not ReviewState.FINALIZED:
                # 只有共同定稿的已同意会话才能作为训练事实。
                continue
            bucket = learned.setdefault(item.taxonomy, {})
            marker = seen.setdefault(item.taxonomy, {})
            confirmed = set(item.confirmed)
            for clause in item.clauses:
                if not marker.get(clause):
                    bucket[clause] = set(confirmed)
                    marker[clause] = True
                else:
                    bucket[clause] &= confirmed
            used.append(item.session)
        self.clause_themes = {
            version: {clause: frozenset(themes) for clause, themes in bucket.items()}
            for version, bucket in learned.items()
        }
        self.trained_sessions = tuple(sorted(used))
        return self

    def predict(self, item: SessionFacts) -> Prediction:
        snap = snapshot(self.records, item.period)
        table = self.clause_themes.get(item.taxonomy, {})
        themes: set[str] = set()
        unseen: list[str] = []
        for clause in item.clauses:
            if clause in table:
                mapped = {theme for theme in table[clause] if theme not in snap.withdrawn}
                themes.update(mapped)
            else:
                unseen.append(clause)
        # 该时期已撤回的主题不能进入判断；没有证据支持的主题保持未决而非强行给标签。
        for withdrawn_theme in snap.withdrawn:
            themes.discard(withdrawn_theme)
        return Prediction(
            session=item.session,
            taxonomy=item.taxonomy,
            themes=frozenset(themes),
            abstained=frozenset(),
            unseen_clauses=tuple(unseen),
            source=self.name,
        )


@dataclass
class LegacyKeywordClassifier:
    """基准 A：冻结在 v1 词典口径的固定关键词系统。

    它只"认识"v1 时期出现过的小句，跨版时严格按 bridges 机械处理：
    - 一对一（对象/地点/语言）直接沿用；
    - 一对多（时段 -> 线上时段/现场时段）不看新词典复核事实，
      把同一份旧标签复制到两个子主题（复现硬拆两栏、各记一次的错误）；
    - v2 才出现的新小句在旧系统里没有关键词，对应困难在简报中"消失"。
    """

    records: Records
    name: str = "legacy_keyword"
    keywords: dict[str, str] = field(default_factory=dict)

    # v1 口径下小句的固定语义（来自 v1 期已复核事实的人工关键词表）。
    V1_KEYWORD_TABLE: tuple[tuple[str, str], ...] = (
        ("不知道哪位负责联络", "对象"),
        ("周末过来要换乘两次", "地点"),
        ("回执有些词看不懂", "语言"),
        ("放学后找不到空闲窗口", "时段"),
        ("暂未提出其他困难", ""),
    )

    def fit(self, training_sessions: list[SessionFacts]) -> "LegacyKeywordClassifier":
        # 冻结系统：关键词表不随训练资料更新，fit 仅保留接口一致性。
        table = {clause: theme for clause, theme in self.V1_KEYWORD_TABLE if theme}
        if training_sessions:
            # 用 v1 训练资料校验固定表与已复核事实一致（发现漂移即报错，不静默改表）。
            for item in training_sessions:
                if item.taxonomy != 1 or not item.usable or item.state is not ReviewState.FINALIZED:
                    continue
                for clause in item.clauses:
                    expected = table.get(clause)
                    if expected is not None and expected not in item.confirmed:
                        raise ValueError(
                            f"固定 v1 关键词表与复核事实冲突：{clause} -> {expected}"
                        )
        self.keywords = table
        return self

    def predict(self, item: SessionFacts) -> Prediction:
        snap = snapshot(self.records, item.period)
        themes: set[str] = set()
        unseen: list[str] = []
        for clause in item.clauses:
            old_theme = self.keywords.get(clause)
            if old_theme is None:
                # 旧系统不认识的新小句：没有任何关键词命中，困难"消失"为未捕获。
                unseen.append(clause)
                continue
            if item.taxonomy == 1:
                if old_theme not in snap.withdrawn:
                    themes.add(old_theme)
                continue
            children = snap.bridges.get(old_theme)
            if not children:
                # 新词典里找不到承接关系：困难在简报里"消失"。
                continue
            # 基准的错误所在：不区分一对多、不看新词典复核事实，直接复制到每个子主题。
            for child in children:
                if child not in snap.withdrawn:
                    themes.add(child)
        return Prediction(
            session=item.session,
            taxonomy=item.taxonomy,
            themes=frozenset(themes),
            abstained=frozenset(),
            unseen_clauses=tuple(unseen),
            source=self.name,
        )


@dataclass
class FullKeywordClassifier:
    """基准 B：不带时期版本的全量关键词表，任何会话都不给未决。"""

    records: Records
    name: str = "full_keyword"
    keywords: dict[str, str] = field(default_factory=dict)

    def fit(self, training_sessions: list[SessionFacts]) -> "FullKeywordClassifier":
        table: dict[str, str] = {}
        for item in training_sessions:
            if not item.usable or item.state is not ReviewState.FINALIZED:
                continue
            for clause in item.clauses:
                for theme in item.confirmed:
                    table.setdefault(clause, theme)
        self.keywords = table
        return self

    def predict(self, item: SessionFacts) -> Prediction:
        snap = snapshot(self.records, item.period)
        themes: set[str] = set()
        unseen: list[str] = []
        for clause in item.clauses:
            theme = self.keywords.get(clause)
            if theme is None:
                unseen.append(clause)
                continue
            if theme in snap.themes and theme not in snap.withdrawn:
                themes.add(theme)
        return Prediction(
            session=item.session,
            taxonomy=item.taxonomy,
            themes=frozenset(themes),
            abstained=frozenset(),
            unseen_clauses=tuple(unseen),
            source=self.name,
        )


def judge_new_material(
    classifier: EvidenceClassifier, item: SessionFacts
) -> Prediction:
    """新资料的判断入口：未经同意的会话不产出任何主题判断。"""
    if not item.usable:
        return Prediction(
            session=item.session,
            taxonomy=item.taxonomy,
            themes=frozenset(),
            abstained=frozenset(),
            unseen_clauses=(),
            source=f"{classifier.name}:blocked_no_consent",
        )
    return classifier.predict(item)
