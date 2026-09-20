"""Converter: scikit-learn → OMLE.

Supports:
  - DecisionTree{Classifier,Regressor}
  - {RandomForest,ExtraTrees}{Classifier,Regressor}
  - GradientBoosting{Classifier,Regressor}
  - HistGradientBoosting{Classifier,Regressor}
  - SVC, SVR, NuSVC, NuSVR  (kernel SVM)
  - LinearSVC, LinearSVR    (linear SVM)
  - MLPClassifier, MLPRegressor
  - KNeighborsClassifier, KNeighborsRegressor
  - LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    BayesianRidge, HuberRegressor, TheilSenRegressor,
    SGDRegressor, PassiveAggressiveRegressor
  - LogisticRegression, LogisticRegressionCV,
    RidgeClassifier, RidgeClassifierCV,
    SGDClassifier, PassiveAggressiveClassifier, Perceptron
  - KMeans, MiniBatchKMeans, GaussianMixture
  - GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB
  - VotingClassifier, VotingRegressor
  - StackingClassifier, StackingRegressor
  - OneVsRestClassifier
  - MultiOutputClassifier, MultiOutputRegressor
  - ClassifierChain, RegressorChain
  - Pipeline with preprocessing steps:
      StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
      Normalizer, SimpleImputer, Binarizer, PCA,
      OneHotEncoder, OrdinalEncoder, ColumnTransformer,
      FunctionTransformer (passthrough), FeatureUnion,
      nested Pipeline

Submodules
----------
_builder      : Builder — accumulates nodes/constants/imports
_trees        : parse_dt_tree, parse_histgb_nodes — low-level tree parsing
_estimators   : convert_estimator_to_node — tree ensemble model converters
_transformers : convert_transformer — preprocessing node converters

Usage::

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from omle_convert.sklearn import from_sklearn
    import omle

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=50)),
    ]).fit(X_train, y_train)

    model = from_sklearn(pipe)
    omle.save(model, "model.omle")
"""

from __future__ import annotations

from typing import Optional, Union

import omle

from .._common import (
    TaskType,
    _df_uniform_dtype,
    _is_str_dtype,
    make_auxiliary_data,
    make_input_spec,
    make_input_specs,
    make_metadata,
    make_model_schema,
    make_output_specs,
    numpy_input_dtype,
)
from ._builder import Builder
from ._estimators import (
    _ANOMALY_DETECTION_CLASSES,
    _CLUSTERING_CLASSES,
    _LINEAR_ESTIMATOR_CLASSES,
    _MULTICLASS_CLASSES,
    _MULTIOUTPUT_CLASSES,
    _NAIVE_BAYES_CLASSES,
    _SKLEARN_FLOAT32_CLASSES,
    _STACKING_CLASSES,
    _VOTING_CLASSES,
    convert_estimator_to_node,
)
from ._transformers import convert_transformer
from ._xgboost_lgbm import _LGB_CLASSES, _XGB_CLASSES

# ── Public API ────────────────────────────────────────────────────────────────

def from_sklearn(
    model,
    *,
    X=None,
    feature_names: Optional[list[str]] = None,
    target_name: str = "y",
    class_labels: Optional[list] = None,
    model_name: str = "",
    model_version: str = "",
    copyright: str = "",
    n_verify: Union[int, float] = 10,
    n_warmup: Union[int, float] = 10,
    n_warmup_repeat: int = 3,
    n_sample: int = 3,
    verify_atol: float = 1e-4,
    verify_rtol: float = 1e-4,
    random_state: Optional[int] = None,
) -> omle.OMLEModel:
    """Convert a scikit-learn model or Pipeline to an OMLEModel.

    Parameters
    ----------
    model:
        A fitted scikit-learn estimator or ``Pipeline``.  Supported estimators:
        ``DecisionTree{Classifier,Regressor}``,
        ``{RandomForest,ExtraTrees}{Classifier,Regressor}``,
        ``GradientBoosting{Classifier,Regressor}``,
        ``HistGradientBoosting{Classifier,Regressor}``.

        Supported preprocessing steps inside a ``Pipeline`` or
        ``ColumnTransformer``:
        ``StandardScaler``, ``MinMaxScaler``, ``RobustScaler``, ``MaxAbsScaler``,
        ``Normalizer``, ``SimpleImputer``, ``Binarizer``,
        ``OneHotEncoder``, ``OrdinalEncoder``, ``PCA``, ``ColumnTransformer``,
        ``FunctionTransformer`` (passthrough only).
    X:
        Optional data (numpy array or DataFrame) used for schema inference and
        to populate verification cases, warmup inputs, and sample inputs.  When
        a DataFrame is provided (and no preprocessing pipeline), per-column
        InputSpecs are generated.  When ``None`` (the default), those sections
        are omitted.
    feature_names:
        Column names for the raw input tensor.  Inferred from the model when
        possible (``feature_names_in_``).
    target_name:
        Name for the target variable in ``ModelSchema``.
    class_labels:
        Class label strings for classification models.  Inferred from
        ``model.classes_`` when available.
    model_name:
        Human-readable name stored in ``ModelMetadata.name``.
    model_version:
        Version string for the model artifact (e.g. ``"1.0.0"``), stored in
        ``ModelMetadata.version``.  Distinct from the framework version.
    copyright:
        Copyright notice stored in ``ModelMetadata.copyright``
        (e.g. ``"© 2025 Acme Corp"``).
    n_verify:
        Rows for verification — ``int`` (absolute) or ``float`` fraction of
        ``len(X)`` (e.g. ``0.1`` = 10 %).  ``0`` / ``0.0`` skips the section.
    n_warmup:
        Rows for warmup — same int-or-float semantics as ``n_verify``.
    n_warmup_repeat:
        How many times the runtime should repeat the warmup pass.
    n_sample:
        Number of single-row sample inputs to store (0 = skip).
    verify_atol:
        Absolute tolerance for verification comparisons.
    verify_rtol:
        Relative tolerance for verification comparisons.
    random_state:
        Seed for random row selection from ``X``.  ``None`` uses the first N
        rows; an integer seeds a reproducible draw.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from e

    builder = Builder()

    raw_feat_names = feature_names or _infer_input_feature_names(model)
    estimator, pre_steps = _decompose_model(model)

    # Detect text-vectorizer pipelines: first step consumes raw string documents.
    _text_input = _is_text_pipeline(pre_steps)
    if _text_input and raw_feat_names is None:
        raw_feat_names = ["text"]

    # For pipelines with preprocessing, use the pipeline's own n_features_in_
    # (which reflects the raw input), not the final estimator's (which reflects
    # the pre-processed input, e.g. PCA output dimension).
    n_features = _n_features_in(model if pre_steps else estimator, raw_feat_names)
    if raw_feat_names is None:
        raw_feat_names = [f"x{i}" for i in range(n_features)]

    task = _task_type(estimator)
    n_classes = _n_classes(estimator, task)
    if class_labels is None:
        class_labels = _class_labels(estimator, n_classes)

    # Per-column mode: use DataFrame columns when X is a DataFrame.
    # Pipelines with ColumnTransformer also benefit: string columns are routed correctly.
    try:
        import pandas as pd
        _df = X if isinstance(X, pd.DataFrame) else None
    except ImportError:
        _df = None

    # Track which input column names hold string values so _column_transformer
    # can choose StringHStack vs Concat for each group.
    if _df is not None:
        for col in _df.columns:
            # Not `dtype == object or str(dtype).startswith("string")`: pandas
            # 3.0 reports a string column's dtype as `str`, which satisfies
            # neither test, and the column would then be treated as numeric.
            if _is_str_dtype(_df[col].dtype):
                builder.str_columns.add(str(col))

    # Chain preprocessing nodes: X → ... → pre_out
    # For homogeneous DataFrames (all same non-string dtype), pass the whole matrix as "X"
    # so the estimator node receives a single tensor input rather than per-column inputs.
    if _df is not None:
        current = "X" if _df_uniform_dtype(_df) is not None else list(_df.columns)
    else:
        current = "X"
    for step_name, transformer in pre_steps:
        current = convert_transformer(transformer, current, step_name, builder)

    # Final estimator node — rename node (not outputs) with pipeline step name
    est_step_name: str = ""
    try:
        from sklearn.pipeline import Pipeline as _Pipeline
        if isinstance(model, _Pipeline):
            est_step_name = model.steps[-1][0]
    except ImportError:
        pass
    n_nodes_before = len(builder.nodes)
    convert_estimator_to_node(estimator, current, task, n_classes, builder)
    if est_step_name:
        for node in builder.nodes[n_nodes_before:]:
            node.name = f"{est_step_name}_{node.name}"

    feat_dtype = _feature_dtype(estimator)
    if _text_input:
        input_specs = [omle.InputSpec(
            name="X",
            type=omle.TensorType(dtype=omle.DataType.STRING, shape=[-1]),
        )]
    else:
        input_specs = make_input_specs(raw_feat_names, df=_df, feature_dtype=feat_dtype,
                                       input_dtype=numpy_input_dtype(X))
    metadata = make_metadata("sklearn", _sk_version(), model_name, version=model_version, copyright=copyright)
    sec = _secondary_framework(estimator)
    if sec is not None:
        sec_name, sec_ver = sec
        metadata.source_frameworks.append(
            omle.SourceFramework(name=sec_name, version=sec_ver, role="model")
        )

    top_cls = type(estimator).__name__
    if top_cls in _MULTIOUTPUT_CLASSES:
        n_outputs = len(getattr(estimator, "estimators_", []))
        is_reg = "Regressor" in top_cls
        output_specs = [omle.OutputSpec(
            name="y_pred",
            role=omle.OutputRole.PREDICTION,
            type=omle.TensorType(
                dtype=omle.DataType.FLOAT64 if is_reg else omle.DataType.INT64,
                shape=[-1, n_outputs],
            ),
        )]
        if top_cls in ("ClassifierChain", "MultiOutputClassifier"):
            # Export y_prob when the converter produces it (binary children only).
            child_estimators = getattr(estimator, "estimators_", [])
            all_binary = all(
                len(getattr(e, "classes_", [])) <= 2 for e in child_estimators
            )
            if top_cls == "ClassifierChain" or all_binary:
                output_specs.append(omle.OutputSpec(
                    name="y_prob",
                    role=omle.OutputRole.PROBABILITY,
                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_outputs]),
                ))
    elif (top_cls == "VotingClassifier" and
          getattr(estimator, "voting", "hard") == "hard"):
        # Hard voting produces no y_prob — only y_pred.
        output_specs = [omle.OutputSpec(
            name="y_pred",
            role=omle.OutputRole.PREDICTION,
            type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1]),
        )]
    else:
        output_specs = make_output_specs(task, n_classes)

    omle_model = omle.OMLEModel(
        metadata=metadata,
        operator_imports=builder.namespace_imports(),
        inputs=input_specs,
        outputs=output_specs,
        model_schema=make_model_schema(raw_feat_names, task, target_name, class_labels,
                                       df=_df,
                                       feature_dtype=omle.DataType.STRING if _text_input else feat_dtype),
        nodes=builder.nodes,
        tensor_entries=builder.tensor_entries,
    )

    if X is not None:
        output_map = _get_sklearn_outputs(model, X, task) if n_verify > 0 else None
        make_auxiliary_data(
            omle_model, X, output_map,
            n_verify=n_verify, n_warmup=n_warmup, n_warmup_repeat=n_warmup_repeat,
            n_sample=n_sample, verify_atol=verify_atol, verify_rtol=verify_rtol,
            random_state=random_state,
        )

    return omle_model


# ── Decomposition helpers ─────────────────────────────────────────────────────

def _decompose_model(model):
    """Return (final_estimator, pre_steps).

    For a Pipeline, pre_steps are all steps except the last.
    For a plain estimator, pre_steps is empty.
    """
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            return model.steps[-1][1], model.steps[:-1]
    except ImportError:
        pass
    return model, []


def _infer_input_feature_names(model) -> Optional[list[str]]:
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None:
        return [str(f) for f in fn]
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline) and model.steps:
            fn = getattr(model.steps[0][1], "feature_names_in_", None)
            if fn is not None:
                return [str(f) for f in fn]
    except ImportError:
        pass
    return None


def _n_features_in(estimator, feat_names) -> int:
    if feat_names:
        return len(feat_names)
    n = getattr(estimator, "n_features_in_", None)
    if n is not None:
        return int(n)
    tree_ = getattr(estimator, "tree_", None)
    if tree_ is not None:
        return int(tree_.n_features)
    return 0


# ── Task / label helpers ──────────────────────────────────────────────────────

def _task_type(estimator) -> TaskType:
    cls = type(estimator).__name__

    if cls in _ANOMALY_DETECTION_CLASSES:
        return TaskType.ANOMALY_DETECTION

    if cls in _CLUSTERING_CLASSES:
        return TaskType.CLUSTERING

    # Multi-output wrappers: task determined by class name suffix
    if cls in _MULTIOUTPUT_CLASSES:
        if "Regressor" in cls:
            return TaskType.REGRESSION
        return TaskType.MULTICLASS

    # Voting/Stacking/multiclass meta-estimators
    if cls in _VOTING_CLASSES | _STACKING_CLASSES | _MULTICLASS_CLASSES:
        if "Regressor" in cls:
            return TaskType.REGRESSION
        classes = getattr(estimator, "classes_", None)
        if classes is not None:
            return TaskType.MULTICLASS if len(classes) > 2 else TaskType.BINARY
        # Fall through to generic detection below

    if "Regressor" in cls or "Ranker" in cls or cls in ("SVR", "NuSVR", "LinearSVR"):
        return TaskType.REGRESSION

    # sklearn linear regressors that don't have "Regressor" in the class name
    _REGRESSION_BY_NAME = {
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "Lars", "LassoLars",
        "BayesianRidge", "HuberRegressor", "TheilSenRegressor",
        "QuantileRegressor", "OrthogonalMatchingPursuit", "IsotonicRegression",
    }
    if cls in _REGRESSION_BY_NAME:
        return TaskType.REGRESSION

    n = getattr(estimator, "n_classes_", None)
    if n is None:
        classes = getattr(estimator, "classes_", None)
        n = len(classes) if classes is not None else 2
    return TaskType.MULTICLASS if int(n) > 2 else TaskType.BINARY


def _n_classes(estimator, task: TaskType) -> int:
    if task in (TaskType.REGRESSION, TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION):
        return 1
    n = getattr(estimator, "n_classes_", None)
    if n is not None:
        return int(n)
    classes = getattr(estimator, "classes_", None)
    return len(classes) if classes is not None else 2


def _class_labels(estimator, n_classes: int) -> list:
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        return list(range(n_classes))
    # Multi-output: classes_ is a list of per-output arrays — not a single label list
    if len(classes) > 0 and hasattr(classes[0], "__len__"):
        return list(range(n_classes))
    return list(classes)


def _is_text_pipeline(pre_steps: list) -> bool:
    """Return True if the first preprocessing step is a text vectorizer."""
    if not pre_steps:
        return False
    from ._text import _TEXT_TRANSFORMER_CLASSES
    return type(pre_steps[0][1]).__name__ in _TEXT_TRANSFORMER_CLASSES


def _feature_dtype(estimator) -> omle.DataType:
    """Return the framework's internal float precision for *estimator*."""
    if type(estimator).__name__ in _SKLEARN_FLOAT32_CLASSES:
        return omle.DataType.FLOAT32
    return omle.DataType.FLOAT64


def _sk_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"


def _secondary_framework(estimator) -> Optional[tuple[str, str]]:
    """Return (name, version) for the underlying boosting library, or None."""
    cls = type(estimator).__name__
    if cls in _XGB_CLASSES:
        try:
            import xgboost
            return ("xgboost", xgboost.__version__)
        except ImportError:
            pass
    if cls in _LGB_CLASSES:
        try:
            import lightgbm
            return ("lightgbm", lightgbm.__version__)
        except ImportError:
            pass
    return None


# ── Auxiliary-data helpers ────────────────────────────────────────────────────

def _get_sklearn_outputs(model, X, task: TaskType) -> dict:
    """Run native sklearn predictions and return a dict of output_name → np.ndarray."""
    import numpy as np

    # Multi-output wrappers produce y_pred as (n, n_outputs).
    # ClassifierChain and all-binary MultiOutputClassifier also produce y_prob
    # as (n, n_outputs) positive-class probabilities.
    cls_name = type(model).__name__
    if cls_name in _MULTIOUTPUT_CLASSES:
        pred = np.asarray(model.predict(X), dtype=np.float32)
        if cls_name == "ClassifierChain":
            prob = np.asarray(model.predict_proba(X), dtype=np.float32)
            return {"y_pred": pred, "y_prob": prob}
        if cls_name == "MultiOutputClassifier":
            raw = model.predict_proba(X)  # list of (n, k_i) arrays
            if isinstance(raw, list) and all(a.shape[1] == 2 for a in raw):
                prob = np.column_stack([a[:, 1] for a in raw]).astype(np.float32)
                return {"y_pred": pred, "y_prob": prob}
        return {"y_pred": pred}

    if task == TaskType.ANOMALY_DETECTION:
        score_fn = getattr(model, "score_samples", None)
        if score_fn is None:
            return {}
        try:
            score = np.asarray(score_fn(X), dtype=np.float32)
            return {"anomaly_score": score}
        except Exception:
            return {}

    if task == TaskType.REGRESSION:
        return {"y_pred": np.asarray(model.predict(X), dtype=np.float32)}
    if task == TaskType.CLUSTERING:
        return {"cluster_id": np.asarray(model.predict(X), dtype=np.int32)}

    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is not None:
        raw = predict_proba(X)
        if isinstance(raw, list):
            # MultiOutputClassifier: predict_proba returns a list of (n, k) arrays,
            # one per output. Only y_pred is meaningful here; skip y_prob entirely.
            pred = np.asarray(model.predict(X), dtype=np.float32)
            return {"y_pred": pred}
        raw = np.asarray(raw, dtype=np.float32)
        if task == TaskType.BINARY:
            return {
                "y_pred": np.asarray(model.predict(X), dtype=np.float32),
                "y_prob": raw,
            }
        return {
            "y_pred": np.argmax(raw, axis=1).astype(np.float32),
            "y_prob": raw,
        }

    # Models without predict_proba (e.g. LinearSVC)
    pred = np.asarray(model.predict(X), dtype=np.float32)
    return {"y_pred": pred}
