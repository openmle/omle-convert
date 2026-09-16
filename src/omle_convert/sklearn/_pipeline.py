"""sklearn.pipeline → OMLE: Pipeline (as transformer), FeatureUnion."""
from __future__ import annotations

import omle

from .._common import TaskType
from ._builder import Builder, out_type


def _infer_task_n_classes(estimator) -> tuple[TaskType, int]:
    """Lightweight task / n_classes inference for a single estimator.

    Used when the last step of a Pipeline-as-transformer turns out to be a
    predictor.  Mirrors the logic in ``sklearn/__init__.py`` without relying
    on the package-level imports (which would create a circular dependency).
    """
    cls = type(estimator).__name__

    _REG_NAMES = {
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "Lars", "LassoLars",
        "BayesianRidge", "HuberRegressor", "TheilSenRegressor",
        "KMeans", "MiniBatchKMeans", "GaussianMixture",
    }
    if "Regressor" in cls or cls in ("SVR", "NuSVR") or cls in _REG_NAMES:
        return TaskType.REGRESSION, 1

    classes = getattr(estimator, "classes_", None)
    # Multi-output: classes_ is a list of arrays
    if classes is not None and len(classes) > 0 and hasattr(classes[0], "__len__"):
        return TaskType.MULTICLASS, 2

    n = getattr(estimator, "n_classes_", None)
    if n is None:
        n = len(classes) if classes is not None else 2
    n = int(n)
    return (TaskType.MULTICLASS if n > 2 else TaskType.BINARY), n


def _pipeline_as_transformer(pipeline, input_name: str, prefix: str, builder: Builder) -> str:
    """Convert a Pipeline used as a transformer (e.g. inside ColumnTransformer or FeatureUnion).

    All steps except possibly the last are converted via *convert_transformer*.
    If the last step is not a recognised transformer, it is converted as a
    predictor via *convert_estimator_to_node* with auto-inferred task type.
    The first output tensor of the predictor (``y_pred``) is returned.
    """
    from ._estimators import (
        convert_estimator_to_node,  # lazy: avoids circular import
    )
    from ._transformers import (
        convert_transformer,  # lazy: avoids circular import
    )

    steps = pipeline.steps
    current = input_name

    for idx, (step_name, step) in enumerate(steps):
        sub_prefix = f"{prefix}_{step_name}"
        is_last = idx == len(steps) - 1

        if not is_last:
            current = convert_transformer(step, current, sub_prefix, builder)
            continue

        # Last step: try transformer first, fall back to predictor
        try:
            current = convert_transformer(step, current, sub_prefix, builder)
        except NotImplementedError:
            from ._estimators import cls_snake as _cls_snake
            task, n_classes = _infer_task_n_classes(step)
            n_est_before = len(builder.nodes)
            out_names = convert_estimator_to_node(
                step, current, task, n_classes, builder, name_prefix=f"{sub_prefix}_",
            )
            # Override final node name: {prefix}_{step_name}_{cls_snake}
            new_est_nodes = builder.nodes[n_est_before:]
            if new_est_nodes:
                new_est_nodes[-1].name = f"{sub_prefix}_{_cls_snake(step)}"
            current = out_names[0]  # first output (y_pred, already prefixed)

    return current


def _feature_union(fu, input_name: str, prefix: str, builder: Builder) -> str:
    """Convert FeatureUnion to parallel transformer branches joined by Concat.

    Each non-dropped branch transformer receives the same *input_name*.
    Branch outputs are concatenated column-wise.  ``transformer_weights`` are
    not applied (feature scaling is not representable with current ops).
    """
    from ._transformers import (
        convert_transformer,  # lazy: avoids circular import
    )

    branch_outputs: list[str] = []
    for t_name, transformer in fu.transformer_list:
        if transformer == "drop" or transformer is None:
            continue
        sub_prefix = f"{prefix}_{t_name}"
        out = convert_transformer(transformer, input_name, sub_prefix, builder)
        branch_outputs.append(out)

    if not branch_outputs:
        return input_name
    if len(branch_outputs) == 1:
        return branch_outputs[0]

    out_name = builder.unique_name(f"{prefix}_concat")
    concat_type = None
    try:
        concat_type = out_type(len(fu.get_feature_names_out()))
    except Exception:
        pass
    builder.add_node(omle.Node(
        name=out_name,
        domain="omle.core",
        op="Concat",
        inputs=[omle.NodeInput(name=n) for n in branch_outputs],
        outputs=[omle.NodeOutput(name=out_name, type=concat_type)],
        attributes=[],
    ))
    return out_name
