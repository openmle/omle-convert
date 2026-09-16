"""sklearn.neighbors → OMLE: KNeighbors and RadiusNeighbors classifiers/regressors."""
from __future__ import annotations

import numpy as np

import omle

from .._common import TaskType, make_node_outputs
from ._builder import Builder, _node_inputs

_NEIGHBORS_CLASSES: frozenset[str] = frozenset({
    "KNeighborsClassifier", "KNeighborsRegressor",
    "RadiusNeighborsClassifier", "RadiusNeighborsRegressor",
})

_RADIUS_CLASSES: frozenset[str] = frozenset({
    "RadiusNeighborsClassifier", "RadiusNeighborsRegressor",
})

_METRIC_MAP: dict[str, str] = {
    "euclidean": "euclidean",
    "l2":        "euclidean",
    "manhattan": "manhattan",
    "l1":        "manhattan",
    "minkowski": "minkowski",
    "cosine":    "cosine",
}


def _convert_knn(estimator, input_name, task: TaskType, n_classes: int, builder: Builder) -> None:
    """Convert a KNeighbors/RadiusNeighbors classifier or regressor to a KNN node."""
    try:
        import scipy.sparse as sp
        fit_is_sparse = sp.issparse(estimator._fit_X)
    except ImportError:
        fit_is_sparse = False

    if fit_is_sparse:
        feat_name = builder.add_sparse_tensor("knn_X", estimator._fit_X)
    else:
        feat_name = builder.add_tensor("knn_X",
                                       np.asarray(estimator._fit_X, dtype=np.float64))

    targets = np.asarray(estimator._y, dtype=np.float64).reshape(-1, 1)
    tgt_name = builder.add_tensor("knn_y", targets)

    metric_raw = getattr(estimator, "metric", "euclidean")
    metric = _METRIC_MAP.get(metric_raw, metric_raw)
    if metric not in ("euclidean", "manhattan", "minkowski", "cosine"):
        metric = "euclidean"
    weights = getattr(estimator, "weights", "uniform")
    if weights not in ("uniform", "distance"):
        weights = "uniform"
    task_str = "regression" if task == TaskType.REGRESSION else "classification"
    is_radius = type(estimator).__name__ in _RADIUS_CLASSES

    attrs = [
        omle.Attribute(name="train_features", tensor_ref=omle.TensorRef(id=feat_name)),
        omle.Attribute(name="train_targets",  tensor_ref=omle.TensorRef(id=tgt_name)),
        omle.Attribute(name="task",           s=task_str),
        omle.Attribute(name="metric",         s=metric),
        omle.Attribute(name="weights",        s=weights),
    ]

    if task in (TaskType.BINARY, TaskType.MULTICLASS):
        n_classes = len(estimator.classes_)
        attrs.append(omle.Attribute(name="n_classes", i=n_classes))

    if is_radius:
        attrs.append(omle.Attribute(name="neighbor_mode", s="radius"))
        attrs.append(omle.Attribute(name="radius", f64=float(estimator.radius)))
        outlier_label = getattr(estimator, "outlier_label", None)
        if outlier_label is not None and task == TaskType.BINARY or \
                outlier_label is not None and task == TaskType.MULTICLASS:
            attrs.append(builder.tensor_attr(
                "outlier_label", "knn_outlier_label",
                np.asarray([float(outlier_label)], dtype=np.float64),
            ))
    else:
        attrs.append(omle.Attribute(name="n_neighbors", i=int(estimator.n_neighbors)))

    if metric == "minkowski":
        p = float(getattr(estimator, "p", 2.0))
        attrs.append(omle.Attribute(name="metric_param_names", strings=["p"]))
        attrs.append(builder.tensor_attr(
            "metric_param_values", "knn_metric_p",
            np.asarray([p], dtype=np.float64),
        ))

    builder.add_node(omle.Node(
        name="knn",
        domain="omle.ml",
        op="KNN",
        inputs=_node_inputs(input_name),
        outputs=make_node_outputs(task, n_classes),
        attributes=attrs,
    ))
