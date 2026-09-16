"""sklearn.ensemble → OMLE: RF/ET, GBM, HistGBM, Voting, Stacking."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import omle
from omle.ir.enums import PostTransform, TreeAggregation

from .._common import TaskType
from ._builder import Builder, _node_inputs, out_type
from ._trees import (
    cls_prob,
    make_bias_tree,
    parse_dt_tree,
    parse_dt_tree_multiclass,
    parse_histgb_nodes,
)

_FOREST_CLASSES: frozenset[str] = frozenset({
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier",   "ExtraTreesRegressor",
})
_GB_CLASSES: frozenset[str] = frozenset({
    "GradientBoostingClassifier", "GradientBoostingRegressor",
})
_HISTGB_CLASSES: frozenset[str] = frozenset({
    "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
})
_VOTING_CLASSES: frozenset[str] = frozenset({
    "VotingClassifier", "VotingRegressor",
})
_STACKING_CLASSES: frozenset[str] = frozenset({
    "StackingClassifier", "StackingRegressor",
})
_BAGGING_CLASSES: frozenset[str] = frozenset({
    "BaggingClassifier", "BaggingRegressor",
})
_ADABOOST_CLASSES: frozenset[str] = frozenset({
    "AdaBoostClassifier",
    "AdaBoostRegressor",
})


def _forest(estimator, task: TaskType, n_classes: int, builder: Builder):
    base_estimators = estimator.estimators_

    if task == TaskType.REGRESSION:
        trees = [parse_dt_tree(e.tree_, lambda v: float(v[0]), builder) for e in base_estimators]
        return trees, [], TreeAggregation.AVERAGE, PostTransform.POST_TRANSFORM_UNSPECIFIED, None

    if task == TaskType.BINARY:
        trees = [parse_dt_tree_multiclass(e.tree_, 2, builder) for e in base_estimators]
        return trees, [], TreeAggregation.AVERAGE, PostTransform.POST_TRANSFORM_UNSPECIFIED, None

    trees, tree_group = [], []
    for e in base_estimators:
        for k in range(n_classes):
            trees.append(parse_dt_tree(e.tree_, lambda v, k=k: cls_prob(v, k), builder))
            tree_group.append(k)
    return trees, tree_group, TreeAggregation.SOFT_VOTE, PostTransform.POST_TRANSFORM_UNSPECIFIED, None


def _gradient_boosting(estimator, task: TaskType, n_classes: int, builder: Builder):
    lr = estimator.learning_rate
    leaf_fn = lambda v: float(v[0]) * lr  # noqa: E731

    if task == TaskType.REGRESSION:
        trees = [parse_dt_tree(e[0].tree_, leaf_fn, builder) for e in estimator.estimators_]
        return trees, [], TreeAggregation.SUM, PostTransform.POST_TRANSFORM_UNSPECIFIED, _gb_base_score_regression(estimator)

    if task == TaskType.BINARY:
        trees = [parse_dt_tree(e[0].tree_, leaf_fn, builder) for e in estimator.estimators_]
        return trees, [], TreeAggregation.SUM, PostTransform.SIGMOID_BINARY, _gb_base_score_binary(estimator)

    trees, tree_group = [], []
    base_scores = _gb_base_scores_multiclass(estimator, n_classes)
    for k, bs in enumerate(base_scores):
        trees.append(make_bias_tree(bs, builder))
        tree_group.append(k)
    for iter_ests in estimator.estimators_:
        for k, est in enumerate(iter_ests):
            trees.append(parse_dt_tree(est.tree_, leaf_fn, builder))
            tree_group.append(k)
    return trees, tree_group, TreeAggregation.SUM, PostTransform.SOFTMAX, None


def _gb_base_scores_multiclass(estimator, n_classes: int) -> list[float]:
    init_ = getattr(estimator, "init_", None)
    if init_ is None:
        return [0.0] * n_classes
    try:
        priors = np.asarray(init_.class_prior_).ravel()
        priors = np.clip(priors, 1e-9, 1 - 1e-9)
        return [math.log(p) for p in priors]
    except Exception:
        return [0.0] * n_classes


def _gb_base_score_regression(estimator) -> Optional[float]:
    init_ = getattr(estimator, "init_", None)
    if init_ is None:
        return None
    for attr in ("constant_", "quantiles_"):
        arr = getattr(init_, attr, None)
        if arr is not None:
            try:
                return float(np.asarray(arr).ravel()[0])
            except Exception:
                pass
    return None


def _gb_base_score_binary(estimator) -> Optional[float]:
    init_ = getattr(estimator, "init_", None)
    if init_ is None:
        return None
    try:
        p = float(np.asarray(init_.class_prior_).ravel()[1])
        p = max(min(p, 1 - 1e-9), 1e-9)
        return math.log(p / (1 - p))
    except Exception:
        return None


def _hist_gradient_boosting(estimator, task: TaskType, n_classes: int, builder: Builder):
    baseline = np.asarray(estimator._baseline_prediction).ravel()

    if task == TaskType.REGRESSION:
        trees = [parse_histgb_nodes(ip[0].nodes, builder) for ip in estimator._predictors]
        return trees, [], TreeAggregation.SUM, PostTransform.POST_TRANSFORM_UNSPECIFIED, float(baseline[0]) if len(baseline) == 1 else None

    if task == TaskType.BINARY:
        trees = [parse_histgb_nodes(ip[0].nodes, builder) for ip in estimator._predictors]
        return trees, [], TreeAggregation.SUM, PostTransform.SIGMOID_BINARY, float(baseline[0]) if len(baseline) == 1 else None

    trees, tree_group = [], []
    for k, bs in enumerate(baseline):
        trees.append(make_bias_tree(float(bs), builder))
        tree_group.append(k)
    for iter_preds in estimator._predictors:
        for k, pred in enumerate(iter_preds):
            trees.append(parse_histgb_nodes(pred.nodes, builder))
            tree_group.append(k)
    return trees, tree_group, TreeAggregation.SUM, PostTransform.SOFTMAX, None


# ── DAG helpers for Voting/Stacking ──────────────────────────────────────────

def _agg_node(op: str, input_names: list[str], outputs: list[omle.NodeOutput],
              attributes: list[omle.Attribute], builder: Builder) -> None:
    builder.add_node(omle.Node(
        name=outputs[0].name,
        domain="omle.core",
        op=op,
        inputs=[omle.NodeInput(name=n) for n in input_names],
        outputs=outputs,
        attributes=attributes,
    ))


def _convert_child_to_dag(
    estimator, input_name, task: TaskType, n_classes: int,
    builder: Builder, name_prefix: str,
) -> list[str]:
    # Lazy imports to avoid circular dependency
    from ._estimators import convert_estimator_to_node
    from ._transformers import convert_transformer

    current = input_name
    final_est = estimator
    try:
        from sklearn.pipeline import Pipeline as _Pipeline
        _is_pipeline = isinstance(estimator, _Pipeline)
    except ImportError:
        _is_pipeline = False
    if _is_pipeline:
        for idx, (step_name, transformer) in enumerate(estimator.steps[:-1]):
            current = convert_transformer(
                transformer, current,
                f"{name_prefix}pre{idx}_{step_name}", builder,
            )
        final_est = estimator.steps[-1][1]

    child_classes: Optional[list] = getattr(final_est, "classes_", None)
    child_n_classes = len(child_classes) if child_classes is not None else n_classes
    return convert_estimator_to_node(
        final_est, current, task, child_n_classes, builder, name_prefix=name_prefix,
    )


def _convert_voting(
    estimator, input_name, task: TaskType, n_classes: int, builder: Builder,
) -> None:
    voting  = getattr(estimator, "voting", "hard")
    raw_w   = getattr(estimator, "weights", None)
    child_pred_names: list[str] = []
    child_prob_names: list[str] = []
    active_weights:   list[float] = []
    # estimators_ is a flat list of fitted estimators; names come from estimators parameter
    named_pairs = list(zip([n for n, _ in estimator.estimators], estimator.estimators_, strict=True))
    for i, (est_name, est) in enumerate(named_pairs):
        if est == "drop" or est is None:
            continue
        prefix = f"vote{i}_{est_name}_"
        out = _convert_child_to_dag(est, input_name, task, n_classes, builder, prefix)
        child_pred_names.append(out[0])
        if len(out) > 1:
            child_prob_names.append(out[1])
        if raw_w is not None and raw_w[i] is not None:
            active_weights.append(float(raw_w[i]))
    weights_attr = ([omle.Attribute(name="weights", float64s=active_weights)] if active_weights else [])
    if task == TaskType.REGRESSION:
        op = "WeightedAverage" if active_weights else "Average"
        _agg_node(op, child_pred_names,
                  [omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                                      type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
                  weights_attr, builder)
    elif voting == "soft":
        prob_shape = [-1, 2] if task == TaskType.BINARY else [-1, n_classes]
        _agg_node("SoftVote", child_prob_names,
                  [omle.NodeOutput(name="y_prob", role=omle.OutputRole.PROBABILITY,
                                      type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=prob_shape)),
                   omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                                      type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1]))],
                  weights_attr, builder)
    else:
        op = "WeightedMajorityVote" if active_weights else "MajorityVote"
        _agg_node(op, child_pred_names,
                  [omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                                      type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1]))],
                  weights_attr, builder)


def _convert_bagging(
    estimator, input_name, task: TaskType, n_classes: int, builder: Builder,
) -> None:
    n_features = estimator.n_features_in_
    all_indices = list(range(n_features))
    child_pred_names: list[str] = []
    child_prob_names: list[str] = []

    for i, (est, feat_indices) in enumerate(
        zip(estimator.estimators_, estimator.estimators_features_, strict=True)
    ):
        prefix = f"bag{i}_"
        feat_list = feat_indices.tolist()
        if feat_list != all_indices:
            slot_name = builder.unique_name(f"{prefix}slots")
            builder.add_node(omle.Node(
                name=slot_name,
                domain="omle.core",
                op="TakeSlots",
                inputs=_node_inputs(input_name),
                outputs=[omle.NodeOutput(name=slot_name,
                                            type=out_type(len(feat_list), omle.DataType.FLOAT32))],
                attributes=[omle.Attribute(name="indices", ints=feat_list)],
            ))
            child_input = slot_name
        else:
            child_input = input_name
        child_classes = getattr(est, "classes_", None)
        child_n = len(child_classes) if child_classes is not None else n_classes
        out = _convert_child_to_dag(est, child_input, task, child_n, builder, prefix)
        child_pred_names.append(out[0])
        if len(out) > 1:
            child_prob_names.append(out[1])

    if task == TaskType.REGRESSION:
        _agg_node("Average", child_pred_names,
                  [omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                                      type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
                  [], builder)
    else:
        prob_shape = [-1, 2] if task == TaskType.BINARY else [-1, n_classes]
        _agg_node("SoftVote", child_prob_names,
                  [omle.NodeOutput(name="y_prob", role=omle.OutputRole.PROBABILITY,
                                      type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=prob_shape)),
                   omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                                      type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1]))],
                  [], builder)


def _convert_stacking(
    estimator, input_name, task: TaskType, n_classes: int, builder: Builder,
) -> None:
    from ._estimators import convert_estimator_to_node
    passthrough = getattr(estimator, "passthrough", False)
    raw_stack_methods = getattr(estimator, "stack_method_", None)
    base_outputs: list[str] = []
    # estimators_ is a flat list of fitted estimators; names come from estimators parameter
    est_names = [n for n, _ in estimator.estimators]
    named_pairs = list(zip(est_names, estimator.estimators_, strict=True))
    for i, (est_name, est) in enumerate(named_pairs):
        if est == "drop" or est is None:
            continue
        prefix = f"base{i}_{est_name}_"
        out = _convert_child_to_dag(est, input_name, task, n_classes, builder, prefix)
        # stack_method_ may be a list (new sklearn) or dict keyed by name (old sklearn)
        if isinstance(raw_stack_methods, dict):
            method = raw_stack_methods.get(est_name, "predict_proba")
        elif isinstance(raw_stack_methods, list) and i < len(raw_stack_methods):
            method = raw_stack_methods[i]
        else:
            method = "predict_proba"
        if method == "predict_proba" and len(out) > 1:
            y_prob = out[1]
            child_classes = getattr(est, "classes_", None)
            try:
                from sklearn.pipeline import Pipeline as _Pipeline
                if isinstance(est, _Pipeline):
                    child_classes = getattr(est.steps[-1][1], "classes_", child_classes)
            except ImportError:
                pass
            if child_classes is not None and len(child_classes) == 2:
                # sklearn takes only column 1 (positive class) for binary base estimators
                col1_name = builder.unique_name(f"{prefix}prob_col1")
                builder.add_node(omle.Node(
                    name=col1_name,
                    domain="omle.core",
                    op="TakeSlots",
                    inputs=[omle.NodeInput(name=y_prob)],
                    outputs=[omle.NodeOutput(name=col1_name, type=out_type(1))],
                    attributes=[omle.Attribute(name="indices", ints=[1])],
                ))
                base_outputs.append(col1_name)
            else:
                base_outputs.append(y_prob)
        else:
            base_outputs.append(out[0])
    if passthrough:
        if isinstance(input_name, list):
            for n in reversed(input_name):
                base_outputs.insert(0, n)
        else:
            base_outputs.insert(0, input_name)
    meta_input = base_outputs[0] if len(base_outputs) == 1 else base_outputs
    convert_estimator_to_node(estimator.final_estimator_, meta_input, task, n_classes, builder)


def _convert_adaboost(
    estimator, input_name, task: TaskType, n_classes: int, builder: Builder,
) -> None:
    weights = estimator.estimator_weights_.tolist()
    weights_attr = [omle.Attribute(name="weights", float64s=weights)]

    child_pred_names: list[str] = []
    child_prob_names: list[str] = []

    for i, est in enumerate(estimator.estimators_):
        prefix = f"ada{i}_"
        out = _convert_child_to_dag(est, input_name, task, n_classes, builder, prefix)
        child_pred_names.append(out[0])
        if len(out) > 1:
            child_prob_names.append(out[1])

    if task == TaskType.REGRESSION:
        # Weighted median over child predictions
        _agg_node(
            "WeightedMedian",
            child_pred_names,
            [omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                                type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
            weights_attr,
            builder,
        )
    elif child_pred_names:
        # sklearn 1.6+ always uses SAMME: weighted hard votes + softmax(decision/(K-1)).
        # SAMMEVote takes y_pred from each estimator and applies the SAMME formula.
        samme_attrs = weights_attr + [omle.Attribute(name="n_classes", i=n_classes)]
        prob_shape = [-1, 2] if task == TaskType.BINARY else [-1, n_classes]
        _agg_node(
            "SAMMEVote",
            child_pred_names,
            [
                omle.NodeOutput(name="y_prob", role=omle.OutputRole.PROBABILITY,
                                   type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=prob_shape)),
                omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                                   type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1])),
            ],
            samme_attrs,
            builder,
        )
