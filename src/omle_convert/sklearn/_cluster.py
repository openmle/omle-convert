"""sklearn.cluster + sklearn.mixture → OMLE: KMeans, MiniBatchKMeans, GaussianMixture."""
from __future__ import annotations

import numpy as np

import omle
from omle.ir.bodies import (
    Clustering,
    GaussianMixtureClustering,
    PrototypeClustering,
)
from omle.ir.enums import CovarianceType, DistanceMeasure
from omle.ir.types import Scalar

from .._common import TaskType
from ._builder import Builder, _node_inputs

_CLUSTERING_CLASSES: frozenset[str] = frozenset({
    "KMeans", "MiniBatchKMeans", "GaussianMixture",
})


def _convert_clustering(estimator, input_name, builder: Builder) -> None:
    """Convert a sklearn clustering estimator to a Clustering node."""
    cls_name = type(estimator).__name__

    if cls_name == "GaussianMixture":
        _convert_gaussian_mixture(estimator, input_name, builder)
        return

    # KMeans / MiniBatchKMeans
    centers = np.asarray(estimator.cluster_centers_, dtype=np.float64)
    n_clusters = centers.shape[0]
    cluster_labels = [Scalar.int(i) for i in range(n_clusters)]

    builder.add_node(omle.Node(
        name="clustering",
        domain="omle.ml",
        op="Clustering",
        inputs=_node_inputs(input_name),
        outputs=[omle.NodeOutput(name="cluster_id", role=omle.OutputRole.ENTITY_ID,
                                    type=omle.TensorType(dtype=omle.DataType.INT32, shape=[-1]))],
        clustering=Clustering(
            task_type=TaskType.CLUSTERING,
            prototype=PrototypeClustering(
                centers=builder.body_tensor_value("cluster_centers", centers),
                distance_measure=DistanceMeasure.EUCLIDEAN,
                cluster_labels=cluster_labels,
            )
        ),
    ))


def _convert_gaussian_mixture(estimator, input_name, builder: Builder) -> None:
    """Convert a GaussianMixture to a Clustering node with GaussianMixtureClustering."""
    weights = np.asarray(estimator.weights_, dtype=np.float64)
    means   = np.asarray(estimator.means_,   dtype=np.float64)
    cov_type_str = getattr(estimator, "covariance_type", "full")
    cov_raw = np.asarray(estimator.covariances_, dtype=np.float64)

    n_components, n_features = means.shape

    # For full/tied covariance, store precisions_cholesky_ (upper triangular L where P=L^T@L).
    # Runtime uses y = L@(x-mu), mahal = ||y||^2, log_det_cov = -2*sum(log(diag(L))).
    prec_chol_raw = np.asarray(estimator.precisions_cholesky_, dtype=np.float64)
    if cov_type_str == "tied":
        # (n_features, n_features) → tile to (n_components, n_features, n_features)
        covariances = np.tile(prec_chol_raw[np.newaxis], (n_components, 1, 1))
        cov_enum = CovarianceType.FULL
    elif cov_type_str == "diag":
        covariances = cov_raw  # (n_components, n_features)
        cov_enum = CovarianceType.DIAGONAL
    elif cov_type_str == "spherical":
        # shape (n_components,) → expand to (n_components, n_features)
        covariances = np.tile(cov_raw[:, np.newaxis], (1, n_features))
        cov_enum = CovarianceType.SPHERICAL
    else:  # full
        covariances = prec_chol_raw  # (n_components, n_features, n_features)
        cov_enum = CovarianceType.FULL

    component_labels = [Scalar.int(i) for i in range(n_components)]

    builder.add_node(omle.Node(
        name="clustering",
        domain="omle.ml",
        op="Clustering",
        inputs=_node_inputs(input_name),
        outputs=[omle.NodeOutput(name="cluster_id", role=omle.OutputRole.ENTITY_ID,
                                    type=omle.TensorType(dtype=omle.DataType.INT32, shape=[-1]))],
        clustering=Clustering(
            task_type=TaskType.CLUSTERING,
            gaussian_mixture=GaussianMixtureClustering(
                weights=builder.body_tensor_value("gmm_weights", weights),
                means=builder.body_tensor_value("gmm_means", means),
                covariances=builder.body_tensor_value("gmm_covariances", covariances),
                covariance_type=cov_enum,
                component_labels=component_labels,
            )
        ),
    ))
