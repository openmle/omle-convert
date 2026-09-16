"""sklearn.multioutput → OMLE: MultiOutputClassifier/Regressor, ClassifierChain, RegressorChain."""
from __future__ import annotations

import omle

from .._common import TaskType
from ._builder import Builder, _node_inputs, out_type

_MULTIOUTPUT_CLASSES: frozenset[str] = frozenset({
    "MultiOutputClassifier",
    "MultiOutputRegressor",
    "ClassifierChain",
    "RegressorChain",
})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _concat_preds(
    input_names: list[str],
    out_name: str,
    builder: Builder,
    output_type: "omle.TensorType | None" = None,
) -> None:
    builder.add_node(omle.Node(
        name=out_name,
        domain="omle.core",
        op="Concat",
        inputs=[omle.NodeInput(name=n) for n in input_names],
        outputs=[omle.NodeOutput(
            name=out_name,
            role=omle.OutputRole.PREDICTION,
            type=output_type,
        )],
        attributes=[],
    ))


# ── MultiOutputRegressor ──────────────────────────────────────────────────────

def _convert_multioutput_regressor(estimator, input_name, builder: Builder) -> None:
    """One regressor per output; y_pred outputs concatenated into a single tensor."""
    from ._estimators import (
        convert_estimator_to_node,  # lazy: avoids circular import
    )

    pred_names: list[str] = []
    for k, est in enumerate(estimator.estimators_):
        prefix = f"out{k}_"
        out = convert_estimator_to_node(
            est, input_name, TaskType.REGRESSION, 1, builder, name_prefix=prefix,
        )
        pred_names.append(out[0])

    out_name = builder.unique_name("y_pred")
    _concat_preds(pred_names, out_name, builder,
                  out_type(len(pred_names), dtype=omle.DataType.FLOAT64))


# ── MultiOutputClassifier ─────────────────────────────────────────────────────

def _convert_multioutput_classifier(estimator, input_name, builder: Builder) -> None:
    """One classifier per output; y_pred outputs concatenated into a single tensor.

    When all child estimators are binary classifiers, the positive-class
    probability (col 1 of each y_prob) is also exported as y_prob (n, n_outputs).
    """
    from ._estimators import (
        convert_estimator_to_node,  # lazy: avoids circular import
    )

    pred_names: list[str] = []
    prob_col_names: list[str] = []

    for k, est in enumerate(estimator.estimators_):
        prefix = f"out{k}_"
        child_classes = getattr(est, "classes_", None)
        child_n = len(child_classes) if child_classes is not None else 2
        child_task = TaskType.MULTICLASS if child_n > 2 else TaskType.BINARY
        out = convert_estimator_to_node(
            est, input_name, child_task, child_n, builder, name_prefix=prefix,
        )
        pred_names.append(out[0])

        if child_task == TaskType.BINARY and len(out) > 1:
            pos_name = builder.unique_name(f"out{k}_pos_prob")
            builder.add_node(omle.Node(
                name=pos_name,
                domain="omle.core",
                op="TakeSlots",
                inputs=[omle.NodeInput(name=out[1])],
                outputs=[omle.NodeOutput(name=pos_name, type=out_type(1))],
                attributes=[omle.Attribute(name="indices", ints=[1])],
            ))
            prob_col_names.append(pos_name)

    n_outputs = len(pred_names)
    out_pred_name = builder.unique_name("y_pred")
    _concat_preds(pred_names, out_pred_name, builder,
                  out_type(n_outputs, dtype=omle.DataType.INT64))

    # Export y_prob only when every output is binary (all prob columns collected).
    if len(prob_col_names) == n_outputs:
        out_prob_name = builder.unique_name("y_prob")
        builder.add_node(omle.Node(
            name=out_prob_name,
            domain="omle.core",
            op="Concat",
            inputs=[omle.NodeInput(name=n) for n in prob_col_names],
            outputs=[omle.NodeOutput(
                name="y_prob",
                role=omle.OutputRole.PROBABILITY,
                type=out_type(n_outputs),
            )],
            attributes=[],
        ))


# ── ClassifierChain / RegressorChain ─────────────────────────────────────────

def _convert_chain(estimator, input_name, builder: Builder) -> None:
    """Convert ClassifierChain / RegressorChain to a sequential DAG.

    sklearn 1.6+ uses binary label predictions (chain_method_='predict') as
    chain features for both predict and predict_proba.  The probability output
    at each link is computed independently on the same (binary-chained) input.
    So both y_pred and y_prob share the same chain features.

    For ClassifierChain: chain feature = binary y_pred (0/1); y_prob is
    assembled from each link's positive-class probability (TakeSlots col 1).
    For RegressorChain: chain feature = predicted value; no y_prob.
    """
    from ._estimators import (
        convert_estimator_to_node,  # lazy: avoids circular import
    )

    is_regression = "Regressor" in type(estimator).__name__
    estimators_ = estimator.estimators_
    n_outputs = len(estimators_)
    _order = getattr(estimator, "order_", None)
    order_: list[int] = list(_order) if _order is not None else list(range(n_outputs))

    current_features = input_name
    chain_preds: list[tuple[int, str]] = []   # (target_idx, y_pred_name)
    chain_probs: list[tuple[int, str]] = []   # (target_idx, pos_prob_col_name) — classifiers only
    n_input_cols: int = getattr(estimator, "n_features_in_", 0)
    current_n_cols: int = n_input_cols

    for step_idx in range(n_outputs):
        est = estimators_[step_idx]
        target_idx = order_[step_idx]
        prefix = f"chain{target_idx}_"

        if is_regression:
            child_task, child_n = TaskType.REGRESSION, 1
        else:
            child_classes = getattr(est, "classes_", None)
            child_n = len(child_classes) if child_classes is not None else 2
            child_task = TaskType.MULTICLASS if child_n > 2 else TaskType.BINARY

        out = convert_estimator_to_node(
            est, current_features, child_task, child_n, builder, name_prefix=prefix,
        )
        y_step = out[0]  # y_pred (binary label) — used as chain feature
        chain_preds.append((target_idx, y_step))

        # For classifiers, also extract positive-class prob (col 1 of y_prob)
        # for the final y_prob output.
        if not is_regression and len(out) > 1:
            pos_name = builder.unique_name(f"chain{target_idx}_pos_prob")
            builder.add_node(omle.Node(
                name=pos_name,
                domain="omle.core",
                op="TakeSlots",
                inputs=[omle.NodeInput(name=out[1])],
                outputs=[omle.NodeOutput(name=pos_name, type=out_type(1))],
                attributes=[omle.Attribute(name="indices", ints=[1])],
            ))
            chain_probs.append((target_idx, pos_name))

        if step_idx < n_outputs - 1:
            # Chain feature = binary y_pred (0/1), matching sklearn's chain_method_='predict'.
            current_n_cols += 1
            next_feat = builder.unique_name(f"chain_feat_{step_idx + 1}")
            feat_type = out_type(current_n_cols) if current_n_cols > 0 else None
            builder.add_node(omle.Node(
                name=next_feat,
                domain="omle.core",
                op="Concat",
                inputs=_node_inputs(current_features) + [omle.NodeInput(name=y_step)],
                outputs=[omle.NodeOutput(name=next_feat, type=feat_type)],
                attributes=[],
            ))
            current_features = next_feat

    # y_pred: re-order to original target column order and concatenate.
    chain_preds.sort(key=lambda x: x[0])
    ordered_preds = [n for _, n in chain_preds]
    pred_dtype = omle.DataType.FLOAT64 if is_regression else omle.DataType.INT64
    out_pred_name = builder.unique_name("y_pred")
    _concat_preds(ordered_preds, out_pred_name, builder, out_type(n_outputs, dtype=pred_dtype))

    # y_prob: concatenate per-link positive-class prob columns (classifiers only).
    if chain_probs:
        chain_probs.sort(key=lambda x: x[0])
        ordered_probs = [n for _, n in chain_probs]
        out_prob_name = builder.unique_name("y_prob")
        builder.add_node(omle.Node(
            name=out_prob_name,
            domain="omle.core",
            op="Concat",
            inputs=[omle.NodeInput(name=n) for n in ordered_probs],
            outputs=[omle.NodeOutput(
                name="y_prob",
                role=omle.OutputRole.PROBABILITY,
                type=out_type(n_outputs),
            )],
            attributes=[],
        ))


# ── Public dispatcher ─────────────────────────────────────────────────────────

def _convert_multioutput(
    estimator, input_name, task: TaskType, n_classes: int, builder: Builder,
) -> None:
    cls_name = type(estimator).__name__
    if cls_name == "MultiOutputRegressor":
        _convert_multioutput_regressor(estimator, input_name, builder)
    elif cls_name == "MultiOutputClassifier":
        _convert_multioutput_classifier(estimator, input_name, builder)
    elif cls_name in ("ClassifierChain", "RegressorChain"):
        _convert_chain(estimator, input_name, builder)
    else:
        raise NotImplementedError(f"Unsupported multioutput estimator: {cls_name}")
