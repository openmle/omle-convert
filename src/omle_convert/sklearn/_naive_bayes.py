"""sklearn.naive_bayes → OMLE: GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB."""
from __future__ import annotations

from typing import Optional

import numpy as np

import omle
from omle.ir.bodies import (
    BernoulliNaiveBayes,
    CategoricalNaiveBayes,
    GaussianNaiveBayes,
    MultinomialNaiveBayes,
    NaiveBayes,
)
from omle.ir.types import Scalar

from .._common import TaskType, make_node_outputs
from ._builder import Builder, _node_inputs

_NAIVE_BAYES_CLASSES: frozenset[str] = frozenset({
    "GaussianNB", "MultinomialNB", "BernoulliNB", "CategoricalNB", "ComplementNB",
})


def _convert_naive_bayes(estimator, input_name, task: TaskType, n_classes: int, builder: Builder) -> None:
    """Convert a sklearn NaiveBayes estimator to a NaiveBayes node."""
    cls_name = type(estimator).__name__

    if hasattr(estimator, "class_log_prior_"):
        class_log_prior = np.asarray(estimator.class_log_prior_, dtype=np.float64)
    else:
        # GaussianNB stores class_prior_ (not log-scaled)
        class_log_prior = np.log(np.asarray(estimator.class_prior_, dtype=np.float64))

    if cls_name == "GaussianNB":
        var_eps: Optional[float] = float(estimator.var_smoothing) \
            if hasattr(estimator, "var_smoothing") else None
        impl = NaiveBayes(
            task_type=task,
            class_log_priors=builder.body_tensor_value("nb_class_log_priors", class_log_prior),
            gaussian=GaussianNaiveBayes(
                means=builder.body_tensor_value("nb_means",
                                                np.asarray(estimator.theta_, dtype=np.float64)),
                variances=builder.body_tensor_value("nb_variances",
                                                    np.asarray(estimator.var_, dtype=np.float64)),
                variance_epsilon=Scalar(double_value=var_eps) if var_eps is not None else None,
            ),
        )

    elif cls_name in ("MultinomialNB", "ComplementNB"):
        # ComplementNB._joint_log_likelihood does not add class_log_prior_; store zeros.
        if cls_name == "ComplementNB":
            n_classes = len(estimator.classes_)
            class_log_prior = np.zeros(n_classes, dtype=np.float64)
        impl = NaiveBayes(
            task_type=task,
            class_log_priors=builder.body_tensor_value("nb_class_log_priors", class_log_prior),
            multinomial=MultinomialNaiveBayes(
                feature_log_prob=builder.body_tensor_value(
                    "nb_feature_log_prob",
                    np.asarray(estimator.feature_log_prob_, dtype=np.float64),
                ),
            ),
        )

    elif cls_name == "BernoulliNB":
        binarize: Optional[float] = float(estimator.binarize) \
            if estimator.binarize is not None else None
        impl = NaiveBayes(
            task_type=task,
            class_log_priors=builder.body_tensor_value("nb_class_log_priors", class_log_prior),
            bernoulli=BernoulliNaiveBayes(
                feature_log_prob=builder.body_tensor_value(
                    "nb_feature_log_prob",
                    np.asarray(estimator.feature_log_prob_, dtype=np.float64),
                ),
                binarize_threshold=Scalar(double_value=binarize) if binarize is not None else None,
            ),
        )

    else:  # CategoricalNB
        # feature_log_prob_ is a list of (n_classes, n_categories_i) arrays
        arrays = [np.asarray(a, dtype=np.float64) for a in estimator.feature_log_prob_]
        category_count = [int(a.shape[1]) for a in arrays]
        category_offset: list[int] = [0]
        for c in category_count[:-1]:
            category_offset.append(category_offset[-1] + c)
        impl = NaiveBayes(
            task_type=task,
            class_log_priors=builder.body_tensor_value("nb_class_log_priors", class_log_prior),
            categorical=CategoricalNaiveBayes(
                category_log_prob=builder.body_tensor_value(
                    "nb_category_log_prob",
                    np.concatenate(arrays, axis=1),
                ),
                category_offset=category_offset,
                category_count=category_count,
            ),
        )

    builder.add_node(omle.Node(
        name="naive_bayes",
        domain="omle.ml",
        op="NaiveBayes",
        inputs=_node_inputs(input_name),
        outputs=make_node_outputs(task, n_classes),
        naive_bayes=impl,
    ))
