"""sklearn anomaly detection → OMLE.

IsolationForest, LocalOutlierFactor, EllipticEnvelope, OneClassSVM and
SGDOneClassSVM.
"""
from __future__ import annotations

import numpy as np

import omle
from omle.ir.bodies import (
    AnomalyDetection,
    Tree,
)
from omle.ir.bodies import (
    EllipticEnvelope as IREllipticEnvelope,
)
from omle.ir.bodies import (
    IsolationForest as IRIsolationForest,
)
from omle.ir.bodies import (
    KernelSVM as IRKernelSVM,
)
from omle.ir.bodies import (
    LinearOneClassSVM as IRLinearOneClassSVM,
)
from omle.ir.bodies import (
    LocalOutlierFactor as IRLocalOutlierFactor,
)
from omle.ir.bodies import (
    OneClassSVM as IROneClassSVM,
)
from omle.ir.enums import (
    DetectionMode,
    ScorePolarity,
    SVMKernelType,
    TreeNodeKind,
    TreeSplitOp,
)
from omle.ir.enums import TaskType as IRTaskType
from omle.ir.types import Scalar, TensorValue

from ._builder import Builder, _node_inputs
from ._trees import _TREE_LEAF

_ANOMALY_DETECTION_CLASSES: frozenset[str] = frozenset({
    "IsolationForest", "LocalOutlierFactor", "EllipticEnvelope",
    "OneClassSVM", "SGDOneClassSVM",
})

_SVM_KERNELS = {
    "linear": SVMKernelType.LINEAR,
    "poly": SVMKernelType.POLY,
    "rbf": SVMKernelType.RBF,
    "sigmoid": SVMKernelType.SIGMOID,
}


def _avg_path_length(n: int) -> float:
    """Expected path length of an unsuccessful BST search (sklearn 1.6+ convention).

    This is the c(n) normalization factor used by sklearn's IsolationForest:
    - n <= 1  → 0.0
    - n == 2  → 1.0
    - n > 2   → 2*(ln(n-1) + γ) - 2*(n-1)/n
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + np.euler_gamma) - 2.0 * (n - 1) / n


def _parse_iforest_tree(tree_, builder=None) -> Tree:
    """Convert an IsolationForest ExtraTreeRegressor.tree_ to an OMLE Tree.

    Each leaf's value is precomputed as ``depth + c(n_node_samples)`` so the
    runtime only needs to average across trees and apply the 2^(-mean/c) formula.
    sklearn split rule: ``x[feature] <= threshold`` → left child.
    """
    # BFS traversal tracking depth alongside node id
    bfs_order: list[int] = []
    depth_of: dict[int, int] = {0: 0}
    queue = [0]
    while queue:
        nid = queue.pop(0)
        bfs_order.append(nid)
        if tree_.children_left[nid] != _TREE_LEAF:
            left  = int(tree_.children_left[nid])
            right = int(tree_.children_right[nid])
            depth_of[left]  = depth_of[nid] + 1
            depth_of[right] = depth_of[nid] + 1
            queue.append(left)
            queue.append(right)

    id_map = {orig: seq for seq, orig in enumerate(bfs_order)}

    node_kind:       list[TreeNodeKind] = []
    split_feature:   list[int]          = []
    split_threshold: list[float]        = []
    split_op:        list[TreeSplitOp]  = []
    children_index:  list[int]          = []
    children_offset: list[int]          = []
    children_count:  list[int]          = []
    default_child:   list[int]          = []
    leaf_value:      list[float]        = []

    missing_left = getattr(tree_, "missing_go_to_left", None)
    child_ptr = 0

    for orig_id in bfs_order:
        children_offset.append(child_ptr)

        if tree_.children_left[orig_id] == _TREE_LEAF:
            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            children_count.append(0)
            default_child.append(0)
            n_samples = int(tree_.n_node_samples[orig_id])
            leaf_value.append(float(depth_of[orig_id]) + _avg_path_length(n_samples))
        else:
            node_kind.append(TreeNodeKind.BRANCH)
            split_feature.append(int(tree_.feature[orig_id]))
            split_threshold.append(float(tree_.threshold[orig_id]))
            split_op.append(TreeSplitOp.LESS_OR_EQUAL)

            left_seq  = id_map[int(tree_.children_left[orig_id])]
            right_seq = id_map[int(tree_.children_right[orig_id])]
            children_index.extend([left_seq, right_seq])
            children_count.append(2)
            child_ptr += 2

            goes_left = bool(missing_left[orig_id]) if missing_left is not None else True
            default_child.append(left_seq if goes_left else right_seq)
            leaf_value.append(0.0)

    if builder is not None:
        st_tv = builder.body_tensor_value("split_threshold", np.array(split_threshold))
        lv_tv = builder.body_tensor_value("leaf_value", np.array(leaf_value))
    else:
        from omle.ir.tensor import Tensor
        st_tv = TensorValue.of_tensor(Tensor(float64_data=split_threshold))
        lv_tv = TensorValue.of_tensor(Tensor(float64_data=leaf_value))

    return Tree(
        task_type=IRTaskType.ANOMALY_DETECTION,
        num_nodes=len(bfs_order),
        node_kind=node_kind,
        split_feature=split_feature,
        split_threshold=st_tv,
        split_op=split_op,
        children_index=children_index,
        children_offset=children_offset,
        children_count=children_count,
        default_child=default_child,
        leaf_value=lv_tv,
    )


def _convert_isolation_forest(estimator, input_name, builder: Builder) -> None:
    trees = [_parse_iforest_tree(est.tree_, builder) for est in estimator.estimators_]
    max_samples = int(estimator.max_samples_)
    offset = Scalar(double_value=float(estimator.offset_))

    builder.add_node(omle.Node(
        name="anomaly_detection",
        domain="omle.ml",
        op="AnomalyDetection",
        inputs=_node_inputs(input_name),
        outputs=[omle.NodeOutput(name="anomaly_score", role=omle.OutputRole.SCORE,
                                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
        anomaly_detection=AnomalyDetection(
            task_type=IRTaskType.ANOMALY_DETECTION,
            mode=DetectionMode.OUTLIER_DETECTION,
            raw_score_polarity=ScorePolarity.LOWER_MORE_ABNORMAL,
            threshold=offset,
            isolation_forest=IRIsolationForest(
                trees=trees,
                max_samples=max_samples,
                offset=offset,
            ),
        ),
    ))


def _convert_local_outlier_factor(estimator, input_name, builder: Builder) -> None:
    if not getattr(estimator, "novelty", False):
        raise NotImplementedError(
            "LocalOutlierFactor with novelty=False is transductive and cannot be "
            "applied to new data. Fit with novelty=True to convert."
        )

    fit_X = np.asarray(estimator._fit_X, dtype=np.float64)

    metric = getattr(estimator, "metric", "minkowski")
    raw_params = getattr(estimator, "effective_metric_params_", {}) or {}
    metric_params = {str(k): str(v) for k, v in raw_params.items()}
    n_neighbors = int(getattr(estimator, "n_neighbors_", estimator.n_neighbors))
    offset = Scalar(double_value=float(estimator.offset_))

    builder.add_node(omle.Node(
        name="anomaly_detection",
        domain="omle.ml",
        op="AnomalyDetection",
        inputs=_node_inputs(input_name),
        outputs=[omle.NodeOutput(name="anomaly_score", role=omle.OutputRole.SCORE,
                                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
        anomaly_detection=AnomalyDetection(
            task_type=IRTaskType.ANOMALY_DETECTION,
            mode=DetectionMode.NOVELTY_DETECTION,
            raw_score_polarity=ScorePolarity.LOWER_MORE_ABNORMAL,
            threshold=offset,
            local_outlier_factor=IRLocalOutlierFactor(
                reference_samples=builder.body_tensor_value("lof_reference_samples", fit_X),
                n_neighbors=n_neighbors,
                metric=metric,
                metric_params=metric_params,
                offset=offset,
            ),
        ),
    ))


def _convert_elliptic_envelope(estimator, input_name, builder: Builder) -> None:
    location  = np.asarray(estimator.location_,   dtype=np.float64)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    precision  = np.asarray(estimator.precision_,  dtype=np.float64)
    offset     = Scalar(double_value=float(estimator.offset_))

    builder.add_node(omle.Node(
        name="anomaly_detection",
        domain="omle.ml",
        op="AnomalyDetection",
        inputs=_node_inputs(input_name),
        outputs=[omle.NodeOutput(name="anomaly_score", role=omle.OutputRole.SCORE,
                                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
        anomaly_detection=AnomalyDetection(
            task_type=IRTaskType.ANOMALY_DETECTION,
            mode=DetectionMode.NOVELTY_DETECTION,
            raw_score_polarity=ScorePolarity.LOWER_MORE_ABNORMAL,
            threshold=offset,
            elliptic_envelope=IREllipticEnvelope(
                location=builder.body_tensor_value("elliptic_location", location),
                covariance=builder.body_tensor_value("elliptic_covariance", covariance),
                precision=builder.body_tensor_value("elliptic_precision", precision),
                offset=offset,
            ),
        ),
    ))


def _convert_one_class_svm(estimator, input_name, builder: Builder) -> None:
    """sklearn.svm.OneClassSVM → AnomalyDetection.one_class_svm.

    The body is lowered so that evaluating it directly yields sklearn's
    ``score_samples``, matching every other anomaly body. sklearn defines

        decision_function(X) = dual_coef_ @ K(SV, X) + intercept_
        score_samples(X)     = decision_function(X) + offset_

    and sets ``offset_ = -intercept_``, so ``score_samples`` is exactly the
    kernel sum with no intercept term. The intercept is therefore left unset
    rather than written as a value the body would have to cancel again; it
    stays recoverable as ``-offset``, and ``decision_function`` as
    ``score - offset``.
    """
    kernel = str(getattr(estimator, "kernel", "rbf"))
    if kernel not in _SVM_KERNELS:
        raise NotImplementedError(
            f"OneClassSVM with kernel={kernel!r} is not supported; "
            f"supported kernels are {sorted(_SVM_KERNELS)}."
        )

    support_vectors = np.asarray(estimator.support_vectors_, dtype=np.float64)
    dual_coef = np.asarray(estimator.dual_coef_, dtype=np.float64).ravel()
    offset = Scalar(double_value=float(np.ravel(estimator.offset_)[0]))

    kernel_svm = IRKernelSVM(
        kernel_type=_SVM_KERNELS[kernel],
        support_vectors=builder.body_tensor_value(
            "ocsvm_support_vectors", support_vectors),
        dual_coefficients=builder.body_tensor_value(
            "ocsvm_dual_coefficients", dual_coef),
        gamma=Scalar(double_value=float(estimator._gamma)),
        degree=int(getattr(estimator, "degree", 3)),
        coef0=Scalar(double_value=float(getattr(estimator, "coef0", 0.0))),
        n_support=[int(n) for n in np.ravel(estimator.n_support_)]
        if hasattr(estimator, "n_support_") else [],
    )

    builder.add_node(omle.Node(
        name="anomaly_detection",
        domain="omle.ml",
        op="AnomalyDetection",
        inputs=_node_inputs(input_name),
        outputs=[omle.NodeOutput(name="anomaly_score", role=omle.OutputRole.SCORE,
                                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
        anomaly_detection=AnomalyDetection(
            task_type=IRTaskType.ANOMALY_DETECTION,
            mode=DetectionMode.NOVELTY_DETECTION,
            raw_score_polarity=ScorePolarity.LOWER_MORE_ABNORMAL,
            threshold=offset,
            one_class_svm=IROneClassSVM(
                kernel_svm=kernel_svm,
                offset=offset,
            ),
        ),
    ))


def _convert_sgd_one_class_svm(estimator, input_name, builder: Builder) -> None:
    """sklearn.linear_model.SGDOneClassSVM → AnomalyDetection.linear_one_class_svm.

    sklearn computes ``score_samples(X) = X @ coef_`` and
    ``decision_function(X) = score_samples(X) - offset_``, so the linear body
    carries the coefficients with no intercept and the offset separately.
    """
    coef = np.asarray(estimator.coef_, dtype=np.float64).ravel()
    offset = Scalar(double_value=float(np.ravel(estimator.offset_)[0]))

    builder.add_node(omle.Node(
        name="anomaly_detection",
        domain="omle.ml",
        op="AnomalyDetection",
        inputs=_node_inputs(input_name),
        outputs=[omle.NodeOutput(name="anomaly_score", role=omle.OutputRole.SCORE,
                                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]))],
        anomaly_detection=AnomalyDetection(
            task_type=IRTaskType.ANOMALY_DETECTION,
            mode=DetectionMode.NOVELTY_DETECTION,
            raw_score_polarity=ScorePolarity.LOWER_MORE_ABNORMAL,
            threshold=offset,
            linear_one_class_svm=IRLinearOneClassSVM(
                coefficients=builder.body_tensor_value("sgd_ocsvm_coefficients", coef),
                offset=offset,
            ),
        ),
    ))


def _convert_anomaly_detection(estimator, input_name, builder: Builder) -> None:
    cls_name = type(estimator).__name__
    if cls_name == "IsolationForest":
        _convert_isolation_forest(estimator, input_name, builder)
    elif cls_name == "LocalOutlierFactor":
        _convert_local_outlier_factor(estimator, input_name, builder)
    elif cls_name == "EllipticEnvelope":
        _convert_elliptic_envelope(estimator, input_name, builder)
    elif cls_name == "OneClassSVM":
        _convert_one_class_svm(estimator, input_name, builder)
    elif cls_name == "SGDOneClassSVM":
        _convert_sgd_one_class_svm(estimator, input_name, builder)
    else:
        raise NotImplementedError(f"Unsupported anomaly detection estimator: {cls_name}")
