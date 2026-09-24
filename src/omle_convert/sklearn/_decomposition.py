"""sklearn.decomposition → OMLE feature operators."""
from __future__ import annotations

import numpy as np

import omle
from omle.ir.bodies import Linear
from omle.ir.enums import DataType as IRDataType
from omle.ir.enums import TaskType
from omle.ir.tensor import Tensor as IRTensor
from omle.ir.types import TensorRef, TensorValue
from omle.ir.types import TensorType as IRTensorType

from ._builder import Builder, out_type


def _attr(name, **kw):
    return omle.Attribute(name=name, **kw)


def _feature_node(op, input_name, output_name, attributes, builder, output_type=None):
    inputs = ([omle.NodeInput(name=n) for n in input_name]
              if isinstance(input_name, list) else [omle.NodeInput(name=input_name)])
    builder.add_node(omle.Node(
        name=output_name, domain="omle.feature", op=op,
        inputs=inputs,
        outputs=[omle.NodeOutput(name=output_name, type=output_type)],
        attributes=attributes,
    ))
    return output_name


def _pca(t, input_name: str, prefix: str, builder: Builder) -> str:
    """PCA as a Linear node: y = (X - mean_) @ components_.T."""
    out = builder.unique_name(f"{prefix}_pca")
    coef_name = builder.add_tensor(f"{prefix}_components", np.asarray(t.components_))
    intercept_tensor = None
    if t.mean_ is not None:
        intercept_arr = np.asarray(-(t.mean_ @ t.components_.T), dtype=np.float64).ravel()
        intercept_tensor = TensorValue.of_tensor(IRTensor(
            float64_data=list(intercept_arr),
            type=IRTensorType(dtype=IRDataType.FLOAT64, shape=[len(intercept_arr)]),
        ))
    builder.add_node(omle.Node(
        name=out,
        domain="omle.ml",
        op="Linear",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name=out, type=out_type(t.n_components_))],
        linear=Linear(
            task_type=TaskType.TASK_TYPE_UNSPECIFIED,
            coefficients=TensorValue.of_ref(TensorRef(id=coef_name)),
            intercept=intercept_tensor,
        ),
    ))
    return out


def _truncated_svd(t, input_name: str, prefix: str, builder: Builder) -> str:
    out = builder.unique_name(f"{prefix}_truncated_svd")
    return _feature_node("TruncatedSVD", input_name, out, [
        builder.tensor_attr("components", f"{prefix}_components", np.asarray(t.components_)),
    ], builder, output_type=out_type(t.components_.shape[0]))


def _fast_ica(t, input_name: str, prefix: str, builder: Builder) -> str:
    attrs = []
    if getattr(t, "whiten", False):
        mean = getattr(t, "mean_", None)
        if mean is not None:
            attrs.append(builder.tensor_attr("mean", f"{prefix}_mean", np.asarray(mean)))
    # No "whitening" attribute: sklearn's transform is (X - mean_) @ components_.T,
    # and components_ is already unmixing_ @ whitening_. Emitting whitening_ as well
    # made the runtime apply it a second time, on top of components that contained
    # it -- the FastICA node then produced a different projection than sklearn for
    # every whitened model. The runtime still honours a "whitening" attribute for
    # producers that store raw unmixing components instead.
    attrs.append(builder.tensor_attr("components", f"{prefix}_components", np.asarray(t.components_)))
    out = builder.unique_name(f"{prefix}_fast_ica")
    return _feature_node("FastICA", input_name, out, attrs, builder,
                         output_type=out_type(t.components_.shape[0]))


def _factor_analysis(t, input_name: str, prefix: str, builder: Builder) -> str:
    attrs = [
        builder.tensor_attr("mean",       f"{prefix}_mean",       np.asarray(t.mean_)),
        builder.tensor_attr("components", f"{prefix}_components", np.asarray(t.components_)),
    ]
    noise_var = getattr(t, "noise_variance_", None)
    if noise_var is not None:
        attrs.append(builder.tensor_attr("noise_variance", f"{prefix}_noise_variance", np.asarray(noise_var)))
    out = builder.unique_name(f"{prefix}_factor_analysis")
    return _feature_node("FactorAnalysis", input_name, out, attrs, builder,
                         output_type=out_type(t.components_.shape[0]))


def _kernel_pca(t, input_name: str, prefix: str, builder: Builder) -> str:
    _ev = getattr(t, "eigenvectors_", None)
    eigenvectors = np.asarray(_ev if _ev is not None else t.alphas_)
    _el = getattr(t, "eigenvalues_", None)
    eigenvalues  = np.asarray(_el if _el is not None else t.lambdas_)
    dual_coef = eigenvectors / np.sqrt(eigenvalues)
    kernel = str(t.kernel)
    attrs = [
        builder.tensor_attr("fit_samples",      f"{prefix}_fit_samples",      np.asarray(t.X_fit_)),
        builder.tensor_attr("dual_components",   f"{prefix}_dual_components",  dual_coef),
        _attr("kernel", s=kernel),
    ]
    gamma = getattr(t, "gamma_", None)
    if gamma is None:
        gamma = t.gamma
    if isinstance(gamma, float):
        attrs.append(_attr("gamma", f64=gamma))
    if kernel in ("poly", "sigmoid"):
        attrs.append(_attr("coef0", f64=float(t.coef0)))
    if kernel == "poly":
        attrs.append(_attr("degree", i=int(t.degree)))
    centerer = getattr(t, "_centerer", None)
    if centerer is not None:
        row_mean = getattr(centerer, "K_fit_rows_", None)
        all_mean = getattr(centerer, "K_fit_all_", None)
        if row_mean is not None:
            attrs.append(builder.tensor_attr("kernel_center_row_mean", f"{prefix}_row_mean", np.asarray(row_mean)))
        if all_mean is not None:
            attrs.append(builder.tensor_attr("kernel_center_all_mean", f"{prefix}_all_mean", np.asarray([all_mean])))
    out = builder.unique_name(f"{prefix}_kernel_pca")
    return _feature_node("KernelPCA", input_name, out, attrs, builder,
                         output_type=out_type(dual_coef.shape[1]))


def _sparse_pca(t, input_name: str, prefix: str, builder: Builder) -> str:
    attrs = []
    mean = getattr(t, "mean_", None)
    if mean is not None:
        attrs.append(builder.tensor_attr("mean", f"{prefix}_mean", np.asarray(mean)))
    attrs.append(builder.tensor_attr("components", f"{prefix}_components", np.asarray(t.components_)))
    out = builder.unique_name(f"{prefix}_sparse_pca")
    return _feature_node("SparsePCA", input_name, out, attrs, builder,
                         output_type=out_type(t.components_.shape[0]))


def _nmf(t, input_name: str, prefix: str, builder: Builder) -> str:
    out = builder.unique_name(f"{prefix}_nmf")
    return _feature_node("NMF", input_name, out, [
        builder.tensor_attr("components", f"{prefix}_components", np.asarray(t.components_)),
    ], builder, output_type=out_type(t.components_.shape[0]))


def _lda(t, input_name: str, prefix: str, builder: Builder) -> str:
    attrs = [builder.tensor_attr("components", f"{prefix}_components", np.asarray(t.components_))]
    doc_prior = getattr(t, "doc_topic_prior_", None) or getattr(t, "doc_topic_prior", None)
    if doc_prior is not None:
        attrs.append(_attr("doc_topic_prior", f64=float(doc_prior)))
    word_prior = getattr(t, "topic_word_prior_", None) or getattr(t, "topic_word_prior", None)
    if word_prior is not None:
        attrs.append(_attr("topic_word_prior", f64=float(word_prior)))
    out = builder.unique_name(f"{prefix}_lda")
    return _feature_node("LatentDirichletAllocation", input_name, out, attrs, builder,
                         output_type=out_type(t.components_.shape[0]))
