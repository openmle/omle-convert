"""omle-convert: converters from popular ML frameworks to the OMLE format."""
from __future__ import annotations

from pathlib import Path

from omle_convert.catboost import from_catboost, from_catboost_file
from omle_convert.lightgbm import from_lightgbm, from_lightgbm_text
from omle_convert.sklearn import from_sklearn
from omle_convert.spark import from_spark, from_spark_live
from omle_convert.xgboost import from_xgboost, from_xgboost_json

__all__ = [
    "to_omle",
    "export_omle",
    "from_xgboost", "from_xgboost_json",
    "from_lightgbm", "from_lightgbm_text",
    "from_sklearn",
    "from_catboost", "from_catboost_file",
    "from_spark", "from_spark_live",
]
__version__ = "0.1.0"


def to_omle(model, **kwargs):
    """Convert a model to OMLE format, auto-detecting the framework.

    Accepts either a live model object or a file path (str/Path).

    Supported objects:
    - any sklearn BaseEstimator or Pipeline, including XGBClassifier and LGBMClassifier
    - xgboost.Booster (raw booster, not wrapped in sklearn API)
    - lightgbm.Booster (raw booster, not wrapped in sklearn API)

    Supported file paths (by extension):
    - ``.json``  → XGBoost JSON model
    - ``.ubj``   → XGBoost binary (ubj)
    - ``.bin`` / ``.txt`` → LightGBM text model
    - directory  → Spark saved model / saved pipeline (auto-detected)

    Extra keyword arguments are forwarded to the underlying converter.
    """
    if isinstance(model, (str, Path)):
        return _convert_from_path(model, **kwargs)
    return _convert_object(model, **kwargs)


def export_omle(model, path, **kwargs) -> None:
    """Export *model* to an OMLE file at *path*.

    If *model* is already an ``OMLEModel`` it is saved directly.
    Otherwise it is converted via ``to_omle`` first.
    """
    import omle
    if not isinstance(model, omle.OMLEModel):
        model = to_omle(model, **kwargs)
    omle.save(model, path)



# Sample-data argument names differ by framework: the array-based converters take
# ``X`` (scikit-learn's convention), the PySpark ones take ``dataset``
# (``fit(dataset)`` / ``transform(dataset)``). Each is idiomatic where it is used,
# but the unified entry points auto-detect the framework, so requiring the caller
# to already know which spelling applies would defeat the point. Accept either
# here and forward whichever the resolved converter declares.
_SAMPLE_ALIASES = ("X", "dataset")


def _sample_kwarg(kwargs: dict, wanted: str) -> dict:
    """Rename the sample-data kwarg to *wanted*.

    The destination is named explicitly rather than discovered by inspecting the
    target's signature: converters get monkey-patched in places (the test suite
    wraps them for instrumentation), and a wrapper's ``(*args, **kwargs)``
    signature would make introspection silently find nothing.
    """
    given = [k for k in _SAMPLE_ALIASES if k in kwargs]
    if not given:
        return kwargs
    if len(given) > 1:
        raise TypeError(
            "Pass either 'X' or 'dataset' for sample data, not both."
        )
    if given[0] == wanted:
        return kwargs
    kwargs = dict(kwargs)
    kwargs[wanted] = kwargs.pop(given[0])
    return kwargs


def _convert_object(model, **kwargs):
    """Dispatch a live model object to the right converter."""
    module = type(model).__module__ or ""
    cls_name = type(model).__name__

    # sklearn (includes XGBClassifier, LGBMClassifier which are BaseEstimators)
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            return from_sklearn(model, **_sample_kwarg(kwargs, "X"))
    except ImportError:
        pass

    # CatBoost (CatBoostClassifier/Regressor/CatBoost are not sklearn BaseEstimators
    # when catboost is not installed alongside sklearn, but handle both cases)
    if module.startswith("catboost"):
        return from_catboost(model, **_sample_kwarg(kwargs, "X"))

    # xgboost Booster (not a BaseEstimator)
    if module.startswith("xgboost"):
        return from_xgboost(model, **_sample_kwarg(kwargs, "X"))

    # lightgbm Booster (not a BaseEstimator)
    if module.startswith("lightgbm"):
        return from_lightgbm(model, **_sample_kwarg(kwargs, "X"))

    # live PySpark PipelineModel or single estimator
    if module.startswith("pyspark"):
        return from_spark_live(model, **_sample_kwarg(kwargs, "dataset"))

    raise TypeError(
        f"Cannot auto-detect converter for {cls_name} (module={module!r}). "
        "Use a specific from_* function instead."
    )


def _convert_from_path(path, **kwargs):
    """Dispatch a file path to the right converter by extension / content."""
    p = Path(path)

    if p.is_dir():
        return from_spark(str(p), **kwargs)

    suffix = p.suffix.lower()

    if suffix in {".bin", ".txt"}:
        return from_lightgbm_text(str(p), **_sample_kwarg(kwargs, "X"))

    if suffix in {".pkl", ".pickle", ".joblib"}:
        try:
            import joblib
            obj = joblib.load(p)
        except ImportError:
            import pickle
            with open(p, "rb") as f:
                obj = pickle.load(f)
        return _convert_object(obj, **kwargs)

    if suffix == ".cbm":
        return from_catboost_file(str(p), **_sample_kwarg(kwargs, "X"))

    if suffix in {".json", ".ubj"}:
        # Both frameworks use `.json`, so dispatch on a top-level key: CatBoost
        # writes "oblivious_trees", XGBoost writes "learner".
        #
        # The whole file is scanned, not a fixed prefix: CatBoost emits
        # "features_info" first and "oblivious_trees" only tens of kilobytes in,
        # so a prefix peek misses it and silently routes to the wrong loader.
        if suffix == ".json":
            try:
                raw = p.read_bytes()
                if b'"oblivious_trees"' in raw:
                    return from_catboost_file(str(p), **_sample_kwarg(kwargs, "X"))
                if b'"learner"' in raw or b'"objective"' in raw:
                    return from_xgboost_json(str(p), **_sample_kwarg(kwargs, "X"))
            except Exception:
                pass
        return from_xgboost_json(str(p), **_sample_kwarg(kwargs, "X"))

    raise ValueError(
        f"Cannot auto-detect converter for path {str(p)!r} (suffix={suffix!r}). "
        "Use a specific from_* function instead."
    )
