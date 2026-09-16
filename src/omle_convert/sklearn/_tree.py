"""sklearn.tree → OMLE: DecisionTree{Classifier,Regressor}."""
from __future__ import annotations

from omle.ir.bodies import Tree

from .._common import TaskType
from ._trees import parse_dt_tree, parse_dt_tree_multiclass

_TREE_CLASSES: frozenset[str] = frozenset({
    "DecisionTreeClassifier", "DecisionTreeRegressor",
})


def _decision_tree(estimator, task: TaskType, n_classes: int, builder=None) -> Tree:
    """Convert a sklearn DecisionTree to a single OMLE Tree."""
    tree_ = estimator.tree_
    if task == TaskType.REGRESSION:
        t = parse_dt_tree(tree_, lambda v: float(v[0]), builder)
    elif task == TaskType.BINARY:
        t = parse_dt_tree_multiclass(tree_, 2, builder)
    else:
        t = parse_dt_tree_multiclass(tree_, n_classes, builder)
    t.task_type = task
    return t
