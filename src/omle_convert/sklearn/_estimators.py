"""Estimator dispatcher: routes sklearn estimators to per-package converters."""
from __future__ import annotations

import re
from typing import Optional

import omle
from omle.ir.bodies import TreeEnsemble
from omle.ir.tensor import Tensor
from omle.ir.types import Scalar, TensorValue

from .._common import TaskType, make_node_outputs
from ._anomaly_detection import (
    _ANOMALY_DETECTION_CLASSES,
    _convert_anomaly_detection,
)
from ._builder import Builder, _node_inputs
from ._cluster import _CLUSTERING_CLASSES, _convert_clustering
from ._ensemble import (
    _ADABOOST_CLASSES,
    _BAGGING_CLASSES,
    _FOREST_CLASSES,
    _GB_CLASSES,
    _HISTGB_CLASSES,
    _STACKING_CLASSES,
    _VOTING_CLASSES,
    _convert_adaboost,
    _convert_bagging,
    _convert_stacking,
    _convert_voting,
    _forest,
    _gradient_boosting,
    _hist_gradient_boosting,
)
from ._linear_model import _LINEAR_ESTIMATOR_CLASSES, _convert_linear
from ._multiclass import _MULTICLASS_CLASSES, _convert_multiclass
from ._multioutput import _MULTIOUTPUT_CLASSES, _convert_multioutput
from ._naive_bayes import _NAIVE_BAYES_CLASSES, _convert_naive_bayes
from ._neighbors import _NEIGHBORS_CLASSES, _convert_knn
from ._neural_network import _NN_CLASSES, _convert_mlp
from ._svm import _SVM_CLASSES, _convert_svm
from ._tree import _TREE_CLASSES, _decision_tree
from ._xgboost_lgbm import (
    _LGB_CLASSES,
    _XGB_CLASSES,
    _convert_lgb,
    _convert_xgb,
)

_ISOTONIC_CLASSES: frozenset[str] = frozenset({"IsotonicRegression"})

# Only XGBoost converts inputs to float32 internally.
# sklearn trees (DT, RF, GB, HGB) store thresholds and leaf values as float64.
_SKLEARN_FLOAT32_CLASSES: frozenset[str] = _XGB_CLASSES

def cls_snake(estimator) -> str:
    """Return the estimator class name in snake_case, e.g. RandomForestClassifier → random_forest_classifier."""
    name = type(estimator).__name__
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s).lower()


# Re-export class sets for use by __init__.py
__all__ = [
    "convert_estimator_to_node",
    "cls_snake",
    "_LINEAR_ESTIMATOR_CLASSES", "_CLUSTERING_CLASSES", "_NAIVE_BAYES_CLASSES",
    "_VOTING_CLASSES", "_STACKING_CLASSES", "_MULTICLASS_CLASSES", "_MULTIOUTPUT_CLASSES",
    "_SKLEARN_FLOAT32_CLASSES", "_ANOMALY_DETECTION_CLASSES",
]


def convert_estimator_to_node(
    estimator,
    input_name,
    task: TaskType,
    n_classes: int,
    builder: Builder,
    *,
    name_prefix: str = "",
) -> list[str]:
    """Dispatch estimator to per-package converter; return output names."""
    cls_name = type(estimator).__name__
    n_nodes_before = len(builder.nodes)
    _tree_params: Optional[tuple] = None  # (trees, tree_group, aggregation, post_transform, base_score)

    if cls_name in _TREE_CLASSES:
        tree = _decision_tree(estimator, task, n_classes, builder)
        builder.add_node(omle.Node(
            name="tree",
            domain="omle.ml",
            op="Tree",
            inputs=_node_inputs(input_name),
            outputs=make_node_outputs(task, n_classes),
            tree=tree,
        ))
    elif cls_name in _FOREST_CLASSES:
        _tree_params = _forest(estimator, task, n_classes, builder)
    elif cls_name in _GB_CLASSES:
        _tree_params = _gradient_boosting(estimator, task, n_classes, builder)
    elif cls_name in _HISTGB_CLASSES:
        _tree_params = _hist_gradient_boosting(estimator, task, n_classes, builder)
    elif cls_name in _SVM_CLASSES:
        _convert_svm(estimator, input_name, task, n_classes, builder)
    elif cls_name in _NN_CLASSES:
        _convert_mlp(estimator, input_name, task, n_classes, builder)
    elif cls_name in _NEIGHBORS_CLASSES:
        _convert_knn(estimator, input_name, task, n_classes, builder)
    elif cls_name in _LINEAR_ESTIMATOR_CLASSES:
        _convert_linear(estimator, input_name, task, n_classes, builder)
    elif cls_name in _CLUSTERING_CLASSES:
        _convert_clustering(estimator, input_name, builder)
    elif cls_name in _NAIVE_BAYES_CLASSES:
        _convert_naive_bayes(estimator, input_name, task, n_classes, builder)
    elif cls_name in _VOTING_CLASSES:
        _convert_voting(estimator, input_name, task, n_classes, builder)
    elif cls_name in _STACKING_CLASSES:
        _convert_stacking(estimator, input_name, task, n_classes, builder)
    elif cls_name in _MULTICLASS_CLASSES:
        _convert_multiclass(estimator, input_name, task, n_classes, builder)
    elif cls_name in _MULTIOUTPUT_CLASSES:
        _convert_multioutput(estimator, input_name, task, n_classes, builder)
    elif cls_name in _BAGGING_CLASSES:
        _convert_bagging(estimator, input_name, task, n_classes, builder)
    elif cls_name in _ADABOOST_CLASSES:
        _convert_adaboost(estimator, input_name, task, n_classes, builder)
    elif cls_name in _XGB_CLASSES:
        _convert_xgb(estimator, input_name, task, n_classes, builder)
    elif cls_name in _LGB_CLASSES:
        _convert_lgb(estimator, input_name, task, n_classes, builder)
    elif cls_name in _ANOMALY_DETECTION_CLASSES:
        _convert_anomaly_detection(estimator, input_name, builder)
    elif cls_name in _ISOTONIC_CLASSES:
        _convert_isotonic_regression(estimator, input_name, builder)
    else:
        _all = (
            _TREE_CLASSES | _FOREST_CLASSES | _GB_CLASSES | _HISTGB_CLASSES |
            _SVM_CLASSES | _NN_CLASSES | _NEIGHBORS_CLASSES |
            _LINEAR_ESTIMATOR_CLASSES | _CLUSTERING_CLASSES |
            _NAIVE_BAYES_CLASSES | _VOTING_CLASSES | _STACKING_CLASSES |
            _MULTICLASS_CLASSES | _MULTIOUTPUT_CLASSES | _BAGGING_CLASSES | _ADABOOST_CLASSES |
            _XGB_CLASSES | _LGB_CLASSES | _ANOMALY_DETECTION_CLASSES | _ISOTONIC_CLASSES
        )
        raise NotImplementedError(
            f"Unsupported estimator type: {cls_name}. "
            f"Supported: {sorted(_all)}."
        )

    if _tree_params is not None:
        trees, tree_group, aggregation, post_transform, base_score = _tree_params
        builder.add_node(omle.Node(
            name="tree_ensemble",
            domain="omle.ml",
            op="TreeEnsemble",
            inputs=_node_inputs(input_name),
            outputs=make_node_outputs(task, n_classes),
            tree_ensemble=TreeEnsemble(
                task_type=task,
                trees=trees,
                aggregation=aggregation,
                post_transform=post_transform,
                # A single value; multiclass gradient boosting folds its
                # per-class intercepts into bias trees instead (see _ensemble).
                base_scores=(TensorValue.of_tensor(Tensor(float64_data=[base_score]))
                             if base_score is not None else None),
                tree_group=tree_group,
            ),
        ))

    new_nodes = builder.nodes[n_nodes_before:]
    if new_nodes:
        new_nodes[-1].name = cls_snake(estimator)

    if name_prefix:
        for node in new_nodes:
            node.name = f"{name_prefix}{node.name}"
            node.outputs = [
                omle.NodeOutput(name=f"{name_prefix}{o.name}", role=o.role, type=o.type)
                for o in node.outputs
            ]

    return [o.name for o in builder.nodes[-1].outputs]


def _convert_isotonic_regression(estimator, input_name: str, builder: Builder) -> None:
    """Convert sklearn IsotonicRegression to a NormContinuous node.

    IsotonicRegression's fitted form is a piecewise-linear function defined by
    breakpoints (X_thresholds_) and fitted values (y_thresholds_), identical
    in structure to Spark IsotonicRegressionModel's boundaries/predictions.
    """
    import numpy as np

    boundaries  = np.asarray(estimator.X_thresholds_, dtype=np.float64)
    predictions = np.asarray(estimator.y_thresholds_, dtype=np.float64)
    offsets     = np.array([0, len(boundaries)], dtype=np.int64)

    builder.add_node(omle.Node(
        name=builder.unique_node_name("isotonic_regression"),
        domain="omle.feature",
        op="NormContinuous",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION)],
        attributes=[
            builder.tensor_attr("orig_points", "ir_boundaries", boundaries),
            builder.tensor_attr("norm_points", "ir_predictions", predictions),
            builder.int_tensor_attr("point_offsets", "ir_offsets", offsets),
            omle.Attribute(name="outlier_treatment", s="as_extreme_values"),
        ],
    ))
