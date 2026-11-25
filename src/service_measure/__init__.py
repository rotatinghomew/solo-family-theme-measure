"""家长留言主题量度、联合复核与材料组合决策。

本包只处理合成业务记录，全部判断均可追溯到：
- 抽样框（frame）的时期/站点/权重/是否回访；
- 经同意（consent）的本地匿名会话摘要；
- 双人复核记录与联合复核结论；
- 词典的时期版本（taxonomy 版本与 bridges 跨版关系）。
"""

__all__ = [
    "io",
    "taxonomy",
    "facts",
    "classifier",
    "evaluation",
    "editorial",
    "scheduling",
]
