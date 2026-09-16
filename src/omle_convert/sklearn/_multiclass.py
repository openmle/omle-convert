"""sklearn.multiclass → OMLE: OneVsRestClassifier."""
from __future__ import annotations

import omle

from .._common import TaskType
from ._builder import Builder, out_type

_MULTICLASS_CLASSES: frozenset[str] = frozenset({
    "OneVsRestClassifier",
    "OneVsOneClassifier",
})


def _convert_ovr(
    estimator, input_name, task: TaskType, n_classes: int, builder: Builder,
) -> None:
    """Convert OneVsRestClassifier to an OMLE DAG.

    For multiclass: each binary child estimator's positive-class probability
    (column 1 of its (n,2) y_prob) is extracted via ``TakeSlots`` and
    concatenated to form a [batch, n_classes] raw probability tensor, which is
    then normalised and argmax'd by ``SoftVote``.

    For binary OVR (1 estimator): the child's full (n,2) y_prob is passed
    directly to ``SoftVote``, which normalises and argmax's to produce the
    final (n,2) y_prob and y_pred matching sklearn's output.
    """
    from ._estimators import (
        convert_estimator_to_node,  # lazy: avoids circular import
    )

    is_binary_ovr = len(estimator.estimators_) == 1
    prob_col_names: list[str] = []
    for k, est in enumerate(estimator.estimators_):
        prefix = f"ovr{k}_"
        child_classes = getattr(est, "classes_", None)
        child_n = len(child_classes) if child_classes is not None else 2
        out = convert_estimator_to_node(
            est, input_name, TaskType.BINARY, child_n, builder, name_prefix=prefix,
        )
        # out[0] = y_pred [batch], out[1] = y_prob [batch, 2]
        if is_binary_ovr:
            # Binary OVR (1 estimator): pass the full (n,2) y_prob directly to
            # SoftVote so it emits a (n,2) normalised output (matches sklearn).
            prob_col_names.append(out[1])
        else:
            # Multiclass OVR: extract positive-class probability (column 1)
            # from each binary estimator to form a (n,1) column per class.
            pos_name = builder.unique_name(f"ovr{k}_prob_pos")
            builder.add_node(omle.Node(
                name=pos_name,
                domain="omle.core",
                op="TakeSlots",
                inputs=[omle.NodeInput(name=out[1])],
                outputs=[omle.NodeOutput(name=pos_name, type=out_type(1))],
                attributes=[omle.Attribute(name="indices", ints=[1])],
            ))
            prob_col_names.append(pos_name)

    # SoftVote row-normalizes the per-class probabilities and produces y_pred
    # via argmax. For multiclass OVR, multiple (n,1) column inputs are stacked
    # into (n, n_classes) by the column-stacking mode in op_soft_vote, so no
    # explicit Concat node is needed. normalize_rows=true matches sklearn's L1
    # normalization: Y /= Y.sum(axis=1, keepdims=True). Binary OVR skips this
    # because the single child already emits a fully normalized (n,2) distribution.
    softvote_attrs = ([] if is_binary_ovr else
                      [omle.Attribute(name="normalize_rows", b=True)])
    prob_shape = [-1, 2] if is_binary_ovr else [-1, n_classes]
    builder.add_node(omle.Node(
        name="ovr_vote",
        domain="omle.core",
        op="SoftVote",
        inputs=[omle.NodeInput(name=n) for n in prob_col_names],
        outputs=[
            omle.NodeOutput(name="y_prob", role=omle.OutputRole.PROBABILITY,
                               type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=prob_shape)),
            omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION,
                               type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1])),
        ],
        attributes=softvote_attrs,
    ))


def _convert_multiclass(
    estimator, input_name, task: TaskType, n_classes: int, builder: Builder,
) -> None:
    cls_name = type(estimator).__name__
    if cls_name == "OneVsRestClassifier":
        _convert_ovr(estimator, input_name, task, n_classes, builder)
    else:
        raise NotImplementedError(
            f"{cls_name} is not supported. "
            "Only OneVsRestClassifier is supported from sklearn.multiclass; "
            "OneVsOneClassifier and OutputCodeClassifier require indexed pairwise "
            "voting operations not yet available in the OMLE op set."
        )
