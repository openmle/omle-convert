"""sklearn.svm → OMLE: SVC, SVR, NuSVC, NuSVR, LinearSVC, LinearSVR."""
from __future__ import annotations

from typing import Optional

import numpy as np

import omle
from omle.ir.bodies import SVM, KernelSVM, LinearSVM
from omle.ir.enums import SVMKernelType, SVMMulticlassStrategy, TaskType
from omle.ir.types import Scalar, TensorRef, TensorValue

from .._common import make_node_outputs
from ._builder import Builder, _node_inputs

_SVM_CLASSES: frozenset[str] = frozenset({
    "SVC", "SVR", "NuSVC", "NuSVR", "LinearSVC", "LinearSVR",
})

_SVM_KERNEL_MAP: dict[str, SVMKernelType] = {
    "linear":  SVMKernelType.LINEAR,
    "poly":    SVMKernelType.POLY,
    "rbf":     SVMKernelType.RBF,
    "sigmoid": SVMKernelType.SIGMOID,
}


def _convert_svm(estimator, input_name, task: TaskType, n_classes: int, builder: Builder) -> None:
    """Convert an sklearn SVM estimator to an SVM node."""
    cls_name = type(estimator).__name__
    ir_task = task

    if cls_name in ("LinearSVC", "LinearSVR"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)
        body = SVM(
            task_type=ir_task,
            linear=LinearSVM(
                coefficients=builder.body_tensor_value("svm_coef", coef),
                intercept=builder.body_tensor_value("svm_intercept", intercept),
            ),
        )
    else:
        # SVC / SVR / NuSVC / NuSVR — kernel SVM
        try:
            import scipy.sparse as sp
            sv_is_sparse = sp.issparse(estimator.support_vectors_)
        except ImportError:
            sv_is_sparse = False

        if sv_is_sparse:
            sv_name = builder.add_sparse_tensor("svm_sv", estimator.support_vectors_)
            sv_tv = TensorValue(tensor_ref=TensorRef(id=sv_name))
        else:
            sv_tv = builder.body_tensor_value(
                "svm_sv", np.asarray(estimator.support_vectors_, dtype=np.float64)
            )

        dual_coef_tv = builder.body_tensor_value(
            "svm_dual_coef", np.asarray(estimator.dual_coef_, dtype=np.float64)
        )
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)

        # Computed gamma (after fit, stored as _gamma for kernel types that use it)
        gamma: Optional[float] = None
        raw_gamma = getattr(estimator, "_gamma", None)
        if raw_gamma is not None:
            gamma = float(raw_gamma)
        elif isinstance(getattr(estimator, "gamma", None), (int, float)):
            gamma = float(estimator.gamma)

        degree = int(estimator.degree) if hasattr(estimator, "degree") else None
        coef0 = float(estimator.coef0) if hasattr(estimator, "coef0") else None
        n_support = list(int(x) for x in estimator.n_support_) \
            if hasattr(estimator, "n_support_") else []
        kernel_type = _SVM_KERNEL_MAP.get(estimator.kernel, SVMKernelType.KERNEL_TYPE_UNSPECIFIED)

        multiclass_strategy: Optional[SVMMulticlassStrategy] = None
        if task == TaskType.MULTICLASS:
            multiclass_strategy = SVMMulticlassStrategy.ONE_VS_ONE  # sklearn SVC always OVO

        # Platt scaling parameters (set when probability=True)
        prob_a: Optional[TensorValue] = None
        prob_b: Optional[TensorValue] = None
        prob_a_raw = getattr(estimator, "probA_", None)
        prob_b_raw = getattr(estimator, "probB_", None)
        if prob_a_raw is not None and len(prob_a_raw) > 0:
            prob_a = builder.body_tensor_value("svm_prob_a",
                                               np.asarray(prob_a_raw, dtype=np.float64))
            prob_b = builder.body_tensor_value("svm_prob_b",
                                               np.asarray(prob_b_raw, dtype=np.float64))

        body = SVM(
            task_type=ir_task,
            multiclass_strategy=multiclass_strategy,
            kernel=KernelSVM(
                kernel_type=kernel_type,
                support_vectors=sv_tv,
                dual_coefficients=dual_coef_tv,
                intercept=builder.body_tensor_value("svm_intercept", intercept),
                gamma=Scalar(double_value=gamma) if gamma is not None else None,
                degree=degree,
                coef0=Scalar(double_value=coef0) if coef0 is not None else None,
                n_support=n_support,
                prob_a=prob_a,
                prob_b=prob_b,
            ),
        )

    builder.add_node(omle.Node(
        name="svm",
        domain="omle.ml",
        op="SVM",
        inputs=_node_inputs(input_name),
        outputs=make_node_outputs(task, n_classes),
        svm=body,
    ))
