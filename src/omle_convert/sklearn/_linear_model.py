"""sklearn.linear_model → OMLE: LinearRegression, Ridge, Lasso, ElasticNet, etc."""
from __future__ import annotations

from typing import Optional

import numpy as np

import omle
from omle.ir.bodies import Linear
from omle.ir.enums import PostTransform

from .._common import TaskType, make_node_outputs
from ._builder import Builder, _node_inputs

_LINEAR_ESTIMATOR_CLASSES: frozenset[str] = frozenset({
    # Regressors
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "Lars", "LassoLars",
    "MultiTaskLasso", "MultiTaskElasticNet",
    "BayesianRidge", "ARDRegression", "HuberRegressor", "TheilSenRegressor",
    "QuantileRegressor", "OrthogonalMatchingPursuit",
    "SGDRegressor", "PassiveAggressiveRegressor",
    # Classifiers
    "LogisticRegression", "LogisticRegressionCV",
    "RidgeClassifier", "RidgeClassifierCV",
    "SGDClassifier", "PassiveAggressiveClassifier", "Perceptron",
})

_BAYESIAN_LINEAR_CLASSES: frozenset[str] = frozenset({"BayesianRidge", "ARDRegression"})


def _convert_linear(estimator, input_name, task: TaskType, n_classes: int, builder: Builder) -> None:
    """Convert a sklearn linear estimator to a Linear node.

    Handles both regression (1-D ``coef_``) and classification (2-D ``coef_``
    with shape ``(n_classes, n_features)``).
    """
    coef = np.asarray(estimator.coef_, dtype=np.float64)
    # Keep sklearn's native layout:
    # - regression/binary: 1-D (n_features,) → shape [n_features] → runtime n_outputs=1
    # - multiclass: 2-D (n_classes, n_features) → shape [n_classes, n_features] → runtime n_outputs=n_classes

    intercept_tensor: Optional[omle.TensorValue] = None
    intercept_raw = getattr(estimator, "intercept_", None)
    if intercept_raw is not None:
        intercept = np.asarray(intercept_raw, dtype=np.float64).ravel()
        intercept_tensor = builder.body_tensor_value("linear_intercept", intercept)

    weight_covariances_ref: Optional[omle.TensorValue] = None
    noise_precision_tensor: Optional[omle.TensorValue] = None
    cls_name = type(estimator).__name__
    if cls_name in _BAYESIAN_LINEAR_CLASSES:
        sigma = getattr(estimator, "sigma_", None)
        if sigma is not None:
            arr = np.asarray(sigma, dtype=np.float64)
            weight_covariances_ref = builder.body_tensor_value("linear_weight_covariances", arr)
        alpha = getattr(estimator, "alpha_", None)
        if alpha is not None:
            noise_precision_tensor = builder.body_tensor_value("linear_noise_precision",
                                                                np.array([float(alpha)]))

    if task == TaskType.REGRESSION:
        post_transform = PostTransform.POST_TRANSFORM_UNSPECIFIED
    elif task == TaskType.BINARY:
        post_transform = PostTransform.SIGMOID
    else:
        post_transform = PostTransform.SOFTMAX

    builder.add_node(omle.Node(
        name="linear",
        domain="omle.ml",
        op="Linear",
        inputs=_node_inputs(input_name),
        outputs=make_node_outputs(task, n_classes),
        linear=Linear(
            task_type=task,
            coefficients=builder.body_tensor_value("linear_coef", coef),
            intercept=intercept_tensor,
            weight_covariances=weight_covariances_ref,
            noise_precision=noise_precision_tensor,
            post_transform=post_transform,
        ),
    ))
