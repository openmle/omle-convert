"""omle_convert.spark — convert Spark ML models to OMLE.

Two user-facing entry points:

``from_spark(path, ...)``
    Reads a saved PipelineModel or single model from disk using pyarrow.
    No SparkSession or JVM required.  The pipeline-vs-model distinction is
    detected automatically from the saved metadata.

``from_spark_live(model, ...)``
    Converts an in-memory fitted PySpark object (PipelineModel, model, or
    transformer).  Requires an active SparkSession.  Saves the object to a
    temporary directory then delegates to ``from_spark``.  Accepts an optional
    PySpark DataFrame ``dataset`` to populate verification, warmup, and sample
    inputs.

"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import Optional, Union

import omle

from ._reader import (
    _read_metadata,
    _short_class,
    from_saved_model,
    from_saved_pipeline,
)

# ── Public API ────────────────────────────────────────────────────────────────

def from_spark(
    spark_save_path: str,
    *,
    feature_names: Optional[list[str]] = None,
    n_features: int = -1,
    target_name: str = "label",
    class_labels: Optional[list] = None,
    model_name: str = "",
    model_version: str = "",
    copyright: str = "",
) -> omle.OMLEModel:
    """Load and convert a saved Spark ML model or PipelineModel to OMLE.

    No SparkSession or JVM is required.  The model type (pipeline vs single
    model) is detected automatically from the saved metadata.

    Parameters
    ----------
    spark_save_path:
        Path to the directory produced by ``model.save(path)`` or
        ``pipeline.save(path)``.
    feature_names:
        Names of the raw input scalar columns fed into the model, in the
        order they appear in the assembled feature vector.
    n_features:
        Total number of features for the input shape.  Automatically inferred
        from the saved model weights when omitted; only needed for pure
        transformer models/pipelines with no estimator stage (e.g. Normalizer,
        VectorSlicer).
    target_name:
        Name for the prediction output column in ``ModelSchema`` (default ``"label"``,
        matching Spark ML convention).
    class_labels:
        Class label values for classifiers (e.g. ``[0, 1]`` or ``["cat", "dog"]``).
        Stored in ``ModelSchema.targets[0].class_labels``.
    model_name:
        Human-readable name stored in ``ModelMetadata.name``.
    model_version:
        Version string for the model artifact (e.g. ``"1.0.0"``), stored in
        ``ModelMetadata.version``.
    copyright:
        Copyright notice stored in ``ModelMetadata.copyright``
        (e.g. ``"© 2025 Acme Corp"``).
    """
    meta = _read_metadata(spark_save_path)
    if _short_class(meta.get("class", "")) == "PipelineModel":
        return from_saved_pipeline(
            spark_save_path,
            feature_names=feature_names,
            n_features=n_features,
            target_name=target_name,
            class_labels=class_labels,
            model_name=model_name,
            model_version=model_version,
            copyright=copyright,
        )
    return from_saved_model(
        spark_save_path,
        feature_names=feature_names,
        n_features=n_features,
        target_name=target_name,
        class_labels=class_labels,
        model_name=model_name,
        model_version=model_version,
        copyright=copyright,
    )


def from_spark_live(
    model,
    *,
    dataset=None,
    feature_names: Optional[list[str]] = None,
    n_features: int = -1,
    target_name: str = "label",
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
    """Convert an in-memory fitted PySpark ML object to OMLE.

    Requires an active SparkSession.  The object is saved to a temporary
    directory and converted via the no-JVM disk reader.

    Parameters
    ----------
    model:
        A fitted ``pyspark.ml.PipelineModel``, a single model (e.g.
        ``LogisticRegressionModel``), or a transformer (e.g.
        ``StandardScalerModel``).
    dataset:
        Optional PySpark DataFrame used to populate verification cases, warmup
        inputs, and sample inputs.  ``model.transform(dataset)`` is called to obtain
        expected outputs.  When ``None``, those sections are omitted.
    feature_names:
        Names of the raw input scalar columns, in the order they appear in
        the assembled feature vector.
    n_features:
        Total number of features for the input shape.  Automatically inferred
        from the saved model weights when omitted; only needed for pure
        transformer models/pipelines with no estimator stage (e.g. Normalizer,
        VectorSlicer).
    target_name:
        Name for the prediction output column in ``ModelSchema`` (default ``"label"``,
        matching Spark ML convention).
    class_labels:
        Class label values for classifiers (e.g. ``[0, 1]`` or ``["cat", "dog"]``).
    model_name:
        Human-readable name stored in ``ModelMetadata.name``.
    model_version:
        Version string for the model artifact (e.g. ``"1.0.0"``).
    copyright:
        Copyright notice stored in ``ModelMetadata.copyright``.
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
    if n_features < 0 and dataset is not None:
        n_features = _infer_n_features(model, dataset)
    tmp = tempfile.mkdtemp(prefix="omle_spark_")
    save_path = os.path.join(tmp, "model")
    try:
        model.save(save_path)
        omle_model = from_spark(
            save_path,
            feature_names=feature_names,
            n_features=n_features,
            target_name=target_name,
            class_labels=class_labels,
            model_name=model_name,
            model_version=model_version,
            copyright=copyright,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if dataset is not None:
        try:
            _attach_auxiliary(
                model, omle_model, dataset, feature_names,
                n_verify=n_verify, n_warmup=n_warmup, n_warmup_repeat=n_warmup_repeat,
                n_sample=n_sample, verify_atol=verify_atol, verify_rtol=verify_rtol,
                random_state=random_state,
            )
        except Exception as e:
            # Auxiliary data is optional, so this stays non-fatal — but swallowing
            # it silently leaves the caller with verification=None and no way to
            # tell a deliberate skip from a bug. Surface the cause as a warning.
            warnings.warn(
                f"Could not attach verification/warmup/sample data: "
                f"{type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
    return omle_model


def _attach_auxiliary(
    spark_model,
    omle_model: omle.OMLEModel,
    dataset,
    feature_cols,
    n_verify,
    n_warmup,
    n_warmup_repeat,
    n_sample,
    verify_atol,
    verify_rtol,
    random_state,
) -> None:
    """Populate omle_model verification/warmup/sample sections from a Spark DataFrame."""
    import numpy as np

    from omle_convert._common import make_auxiliary_data

    # Build the feature input for make_auxiliary_data.
    # Per-column inputs (feature_cols provided): pass a pandas DataFrame.
    # Multi-col transformer (e.g. Imputer): stack all inputCols into a 2-D matrix.
    # Single-tensor input: extract the assembled feature vector as numpy.
    if feature_cols:
        X_pd = dataset.select(feature_cols).toPandas()
        X_in = X_pd
    else:
        input_cols = _get_stage_cols(spark_model, "getInputCols")
        if input_cols and all(c in dataset.columns for c in input_cols):
            # Pass as pandas DataFrame so make_auxiliary_data can map columns by name.
            X_in = dataset.select(input_cols).toPandas()
        else:
            features_col = _infer_features_col(spark_model)
            if features_col not in dataset.columns:
                # _infer_features_col returns an intermediate or generic column that
                # is not in the raw input dataset.  Resolution order:
                # 1. OMLE model's declared input names (text pipelines: "text")
                # 2. Forward-walk Spark stages to find the pipeline entry column
                #    (numeric pipelines: model input is "X", actual col is "features")
                omle_cols = [s.name for s in omle_model.inputs if s.name in dataset.columns]
                if omle_cols:
                    _x_pd = dataset.select(omle_cols).toPandas()
                    features_col = ""  # signal: X_in already built
                    if len(omle_cols) == 1:
                        _first = _x_pd[omle_cols[0]].iloc[0] if len(_x_pd) > 0 else None
                        if _first is not None and hasattr(_first, "toArray"):
                            X_in = np.stack(
                                _x_pd[omle_cols[0]].apply(lambda v: v.toArray())
                            ).astype(np.float64)
                        else:
                            X_in = _x_pd
                    else:
                        X_in = _x_pd
                else:
                    stages = list(getattr(spark_model, "stages", None) or [spark_model])
                    for stage in stages:
                        for method in ("getInputCol", "getFeaturesCol"):
                            try:
                                col = getattr(stage, method)()
                                if isinstance(col, str) and col in dataset.columns:
                                    features_col = col
                                    break
                            except Exception:
                                pass
                        if features_col and features_col in dataset.columns:
                            break
                    if not features_col or features_col not in dataset.columns:
                        return

            if features_col and features_col in dataset.columns:
                try:
                    X_pd = dataset.select(features_col).toPandas()
                except Exception:
                    return
                first_val = X_pd[features_col].iloc[0] if len(X_pd) > 0 else None
                if first_val is not None and hasattr(first_val, "toArray"):
                    # Dense vector column → float64 matrix
                    X_in = np.stack(
                        X_pd[features_col].apply(lambda v: v.toArray())
                    ).astype(np.float64)
                else:
                    # Scalar or string column → pass as pandas DataFrame
                    X_in = X_pd

    # Resolve output column names via standard Has*Col mixin methods.
    pred_col    = _get_stage_col(spark_model, "getPredictionCol",    "prediction")
    prob_col    = _get_stage_col(spark_model, "getProbabilityCol",   "probability")
    raw_col     = _get_stage_col(spark_model, "getRawPredictionCol", "rawPrediction")

    # Run Spark predictions.
    transformed_pd = spark_model.transform(dataset).toPandas()

    # Map Spark output columns → OMLE output spec names.
    output_map = {}
    for spec in omle_model.outputs:
        if spec.role == omle.OutputRole.PREDICTION:
            if pred_col in transformed_pd.columns:
                output_map[spec.name] = np.asarray(transformed_pd[pred_col])
        elif spec.role == omle.OutputRole.ENTITY_ID:
            # Clustering: cluster assignment stored as int32 in predictionCol.
            if pred_col in transformed_pd.columns:
                output_map[spec.name] = np.asarray(transformed_pd[pred_col], dtype=np.int32)
        elif spec.role == omle.OutputRole.PROBABILITY:
            if prob_col in transformed_pd.columns:
                output_map[spec.name] = np.stack(
                    transformed_pd[prob_col].apply(lambda v: v.toArray())
                ).astype(np.float64)
        elif spec.role == omle.OutputRole.SCORE:
            for col in (raw_col, pred_col):
                if col in transformed_pd.columns:
                    output_map[spec.name] = np.asarray(
                        transformed_pd[col], dtype=np.float64
                    )
                    break
        elif spec.role == omle.OutputRole.TRANSFORMED_VALUE:
            # With variadic outputs, spec.name equals the Spark column name directly.
            # Fall back to getOutputCol() for single-col transformers whose output
            # was renamed via explicit_pm but the internal name differs.
            col = spec.name if spec.name in transformed_pd.columns else \
                  _get_stage_col(spark_model, "getOutputCol", "")
            if col and col in transformed_pd.columns:
                col_data  = transformed_pd[col]
                first_val = col_data.iloc[0] if len(col_data) > 0 else None
                if first_val is not None and hasattr(first_val, "toArray"):
                    # Spark DenseVector / SparseVector → float64 matrix
                    output_map[spec.name] = np.stack(
                        col_data.apply(lambda v: v.toArray())
                    ).astype(np.float64)
                elif isinstance(first_val, list):
                    # Token sequence (list of strings) → keep as Python list for
                    # _make_output_entries to serialize as space-joined STRING tensor.
                    output_map[spec.name] = col_data.tolist()
                elif isinstance(first_val, str):
                    output_map[spec.name] = col_data.tolist()
                else:
                    output_map[spec.name] = np.asarray(col_data, dtype=np.float64)

    make_auxiliary_data(
        omle_model, X_in, output_map or None,
        n_verify=n_verify, n_warmup=n_warmup, n_warmup_repeat=n_warmup_repeat,
        n_sample=n_sample, verify_atol=verify_atol, verify_rtol=verify_rtol,
        random_state=random_state,
    )


def _infer_n_features(spark_model, dataset) -> int:
    """Return feature vector size (>0), 0 for scalar/string columns, or -1 when unknown."""
    best = getattr(spark_model, "bestModel", None)
    if best is not None:
        return _infer_n_features(best, dataset)
    candidates = [
        _get_stage_col(spark_model, "getFeaturesCol", ""),
        _get_stage_col(spark_model, "getInputCol", ""),
        *_get_stage_cols(spark_model, "getInputCols"),
    ]
    for col in candidates:
        if not col or col not in dataset.columns:
            continue
        try:
            first = dataset.select(col).first()
            if first and first[0] is not None:
                if hasattr(first[0], "size"):
                    return first[0].size  # DenseVector → vector width
                else:
                    return 0              # scalar or string → 1-D
        except Exception:
            pass
    return -1


def _get_stage_col(spark_model, method: str, default: str) -> str:
    """Call a Has*Col getter on the last pipeline stage that supports it.

    Walks stages in reverse so the final estimator takes precedence over
    intermediate transformers that expose the same method name.
    Falls back to calling the method on the model itself, then to *default*.
    """
    stages = getattr(spark_model, "stages", None)
    if stages:
        for stage in reversed(stages):
            try:
                val = getattr(stage, method)()
                if isinstance(val, str):
                    return val
            except Exception:
                pass
    try:
        val = getattr(spark_model, method)()
        if isinstance(val, str):
            return val
    except Exception:
        pass
    return default


def _get_stage_cols(spark_model, method: str) -> list:
    """Like _get_stage_col but for multi-valued getters (e.g. getInputCols/getOutputCols)."""
    stages = getattr(spark_model, "stages", None)
    if stages:
        for stage in reversed(stages):
            try:
                val = getattr(stage, method)()
                if isinstance(val, (list, tuple)) and val:
                    return list(val)
            except Exception:
                pass
    try:
        val = getattr(spark_model, method)()
        if isinstance(val, (list, tuple)) and val:
            return list(val)
    except Exception:
        pass
    return []


def _infer_features_col(spark_model) -> str:
    """Infer the primary input column name, preferring plural form when available."""
    # getFeaturesCol / getInputCols take precedence over the scalar getInputCol default
    col = (
        _get_stage_col(spark_model, "getFeaturesCol", "") or
        (_get_stage_cols(spark_model, "getInputCols") or [""])[0]
    )
    return col or _get_stage_col(spark_model, "getInputCol", "features")


__all__ = [
    "from_spark",
    "from_spark_live",
]
