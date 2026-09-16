"""Tests for the unified entry points: ``to_omle`` and ``export_omle``.

These cover the auto-detection layer itself — object dispatch in
``_convert_object`` and path dispatch in ``_convert_from_path`` — rather than the
per-framework conversion logic, which the framework-specific suites cover.
"""

import json

import numpy as np
import pytest

import omle
from omle_convert import export_omle, to_omle

# ── Object dispatch ───────────────────────────────────────────────────────────

class TestObjectDispatch:
    """`to_omle(estimator)` routes on the object's module."""

    def test_sklearn_estimator(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        m = to_omle(LogisticRegression(max_iter=300).fit(X, y), X=X)
        assert [f.name for f in m.metadata.source_frameworks] == ["sklearn"]
        assert omle.validate(m).is_valid

    def test_sklearn_pipeline(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = binary_data
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(X, y)
        m = to_omle(pipe, X=X)
        assert [n.op for n in m.nodes] == ["StandardScaler", "Linear"]
        assert omle.validate(m).is_valid

    def test_xgboost_sklearn_wrapper(self, binary_data):
        xgb = pytest.importorskip("xgboost")
        X, y = binary_data
        model = xgb.XGBClassifier(n_estimators=4, eval_metric="logloss").fit(X, y)
        m = to_omle(model, X=X)
        # Wrappers are sklearn BaseEstimators, so the sklearn path runs and
        # records xgboost as the model-producing framework.
        names = [f.name for f in m.metadata.source_frameworks]
        assert "sklearn" in names and "xgboost" in names
        assert omle.validate(m).is_valid

    def test_xgboost_native_booster(self, binary_data):
        xgb = pytest.importorskip("xgboost")
        X, y = binary_data
        booster = xgb.train({"objective": "binary:logistic"}, xgb.DMatrix(X, label=y),
                            num_boost_round=4)
        m = to_omle(booster)
        assert [f.name for f in m.metadata.source_frameworks] == ["xgboost"]
        assert omle.validate(m).is_valid

    def test_lightgbm_native_booster(self, binary_data):
        lgb = pytest.importorskip("lightgbm")
        X, y = binary_data
        booster = lgb.train({"objective": "binary", "verbose": -1},
                            lgb.Dataset(X, label=y), num_boost_round=4)
        m = to_omle(booster)
        assert [f.name for f in m.metadata.source_frameworks] == ["lightgbm"]
        assert omle.validate(m).is_valid

    def test_lightgbm_sklearn_wrapper(self, binary_data):
        lgb = pytest.importorskip("lightgbm")
        X, y = binary_data
        model = lgb.LGBMClassifier(n_estimators=4, verbose=-1).fit(X, y)
        m = to_omle(model, X=X)
        # The wrapper is an sklearn BaseEstimator, so the sklearn path runs and
        # records lightgbm as the model-producing framework.
        names = [f.name for f in m.metadata.source_frameworks]
        assert "sklearn" in names and "lightgbm" in names
        assert omle.validate(m).is_valid

    def test_catboost_estimator(self, binary_data):
        cb = pytest.importorskip("catboost")
        X, y = binary_data
        model = cb.CatBoostClassifier(iterations=4, depth=3, verbose=0).fit(X, y)
        m = to_omle(model, X=X)
        assert "catboost" in [f.name for f in m.metadata.source_frameworks]
        assert omle.validate(m).is_valid

    def test_unsupported_object_raises_typeerror(self):
        with pytest.raises(TypeError, match="Cannot auto-detect converter"):
            to_omle(object())


# ── Path dispatch ─────────────────────────────────────────────────────────────

class TestPathDispatch:
    """`to_omle(path)` routes on extension, and on content when they collide."""

    def test_xgboost_json(self, tmp_path, binary_data):
        xgb = pytest.importorskip("xgboost")
        X, y = binary_data
        booster = xgb.train({"objective": "binary:logistic"}, xgb.DMatrix(X, label=y),
                            num_boost_round=4)
        p = tmp_path / "model.json"
        booster.save_model(str(p))
        m = to_omle(str(p))
        assert [f.name for f in m.metadata.source_frameworks] == ["xgboost"]

    def test_catboost_json_distinguished_from_xgboost_by_content(self, tmp_path, binary_data):
        """Both frameworks use `.json`; dispatch sniffs the file contents."""
        cb = pytest.importorskip("catboost")
        X, y = binary_data
        model = cb.CatBoostClassifier(iterations=4, depth=3, verbose=0).fit(X, y)
        p = tmp_path / "model.json"
        model.save_model(str(p), format="json")
        raw = p.read_bytes()
        # Regression guard: CatBoost writes "oblivious_trees" tens of kilobytes
        # in, so dispatch must scan the file rather than a fixed prefix.
        assert raw.find(b'"oblivious_trees"') > 512
        m = to_omle(str(p))
        assert "catboost" in [f.name for f in m.metadata.source_frameworks]

    def test_lightgbm_txt(self, tmp_path, binary_data):
        lgb = pytest.importorskip("lightgbm")
        X, y = binary_data
        booster = lgb.train({"objective": "binary", "verbose": -1},
                            lgb.Dataset(X, label=y), num_boost_round=4)
        p = tmp_path / "model.txt"
        booster.save_model(str(p))
        m = to_omle(str(p))
        assert [f.name for f in m.metadata.source_frameworks] == ["lightgbm"]

    def test_catboost_cbm(self, tmp_path, binary_data):
        cb = pytest.importorskip("catboost")
        X, y = binary_data
        model = cb.CatBoostClassifier(iterations=4, depth=3, verbose=0).fit(X, y)
        p = tmp_path / "model.cbm"
        model.save_model(str(p))
        m = to_omle(str(p))
        assert "catboost" in [f.name for f in m.metadata.source_frameworks]

    def test_joblib_roundtrips_through_object_dispatch(self, tmp_path, binary_data):
        joblib = pytest.importorskip("joblib")
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        p = tmp_path / "model.joblib"
        joblib.dump(LogisticRegression(max_iter=300).fit(X, y), p)
        m = to_omle(str(p))
        assert [f.name for f in m.metadata.source_frameworks] == ["sklearn"]

    def test_pickle_roundtrips_through_object_dispatch(self, tmp_path, binary_data):
        import pickle

        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        p = tmp_path / "model.pkl"
        p.write_bytes(pickle.dumps(LogisticRegression(max_iter=300).fit(X, y)))
        m = to_omle(str(p))
        assert [f.name for f in m.metadata.source_frameworks] == ["sklearn"]

    def test_unknown_suffix_raises_valueerror(self, tmp_path):
        p = tmp_path / "model.zzz"
        p.write_bytes(b"nope")
        with pytest.raises(ValueError, match="Cannot auto-detect converter for path"):
            to_omle(str(p))

    def test_missing_framework_error_names_the_package(self, tmp_path, monkeypatch):
        """A dynamically loaded framework that is absent must fail informatively."""
        p = tmp_path / "model.txt"
        p.write_bytes(b"tree\n")
        monkeypatch.setitem(__import__("sys").modules, "lightgbm", None)
        with pytest.raises(Exception) as exc:
            to_omle(str(p))
        assert "lightgbm" in str(exc.value).lower()


# ── export_omle ────────────────────────────────────────────────────────────

class TestExportOmle:
    """`export_omle` converts when needed, then writes by extension."""

    def test_writes_protobuf_by_default(self, tmp_path, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        out = tmp_path / "model.omle"
        export_omle(LogisticRegression(max_iter=300).fit(X, y), str(out), X=X)
        assert out.exists() and out.stat().st_size > 0
        assert omle.validate(omle.load(str(out))).is_valid

    def test_writes_json_for_json_extension(self, tmp_path, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        out = tmp_path / "model.json"
        export_omle(LogisticRegression(max_iter=300).fit(X, y), str(out), X=X)
        json.loads(out.read_text())  # must be valid JSON, not protobuf bytes
        assert omle.validate(omle.load(str(out))).is_valid

    def test_accepts_an_already_converted_model_without_reconverting(self, tmp_path, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        model = to_omle(LogisticRegression(max_iter=300).fit(X, y), X=X)
        out = tmp_path / "passthrough.omle"
        export_omle(model, str(out))
        assert omle.to_proto_bytes(omle.load(str(out))) == omle.to_proto_bytes(model)

    def test_roundtrip_preserves_predictions_shape(self, tmp_path, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        X, y = binary_data
        out = tmp_path / "rf.omle"
        export_omle(RandomForestClassifier(n_estimators=4, random_state=0).fit(X, y),
                       str(out), X=X)
        loaded = omle.load(str(out))
        assert [o.role.name for o in loaded.outputs] == ["PREDICTION", "PROBABILITY"]
        assert loaded.outputs[1].type.shape == [-1, len(np.unique(y))]

# ── Spark object dispatch ─────────────────────────────────────────────────────

try:
    import pyspark  # noqa: F401
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


@pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")
class TestSparkObjectDispatch:
    """`to_omle(pyspark_object)` routes to the live Spark path.

    These use the session-scoped ``spark`` fixture from conftest: PySpark's
    SparkContext is a JVM singleton and must be built exactly once per process.
    """

    def test_live_pipeline_model(self, spark, binary_df):
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression
        pipeline_model = Pipeline(stages=[
            LogisticRegression(featuresCol="features", labelCol="label"),
        ]).fit(binary_df)
        m = to_omle(pipeline_model, dataset=binary_df)
        assert [f.name for f in m.metadata.source_frameworks] == ["spark"]
        assert omle.validate(m).is_valid

    def test_live_single_model(self, spark, binary_df):
        from pyspark.ml.classification import LogisticRegression
        model = LogisticRegression(featuresCol="features", labelCol="label").fit(binary_df)
        m = to_omle(model, dataset=binary_df)
        assert [f.name for f in m.metadata.source_frameworks] == ["spark"]
        assert [n.op for n in m.nodes] == ["Linear"]
        assert omle.validate(m).is_valid

    def test_accepts_the_X_alias(self, spark, binary_df):
        """A Spark model converted with `X=` must match `dataset=`."""
        from pyspark.ml.classification import LogisticRegression
        model = LogisticRegression(featuresCol="features", labelCol="label").fit(binary_df)
        via_dataset = to_omle(model, dataset=binary_df)
        via_x = to_omle(model, X=binary_df)
        assert [n.op for n in via_dataset.nodes] == [n.op for n in via_x.nodes]
        assert omle.validate(via_x).is_valid

    def test_pipeline_and_single_model_are_distinguished(self, spark, binary_df):
        """Auto-detection must not collapse the two shapes."""
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.feature import StandardScaler
        single = LogisticRegression(featuresCol="features", labelCol="label").fit(binary_df)
        pipeline_model = Pipeline(stages=[
            StandardScaler(inputCol="features", outputCol="scaled"),
            LogisticRegression(featuresCol="scaled", labelCol="label"),
        ]).fit(binary_df)
        assert len(to_omle(single, dataset=binary_df).nodes) == 1
        assert len(to_omle(pipeline_model, dataset=binary_df).nodes) == 2

# ── Sample-data argument aliasing ─────────────────────────────────────────────

class TestSampleKwargAlias:
    """The unified entry points accept either `X` or `dataset`.

    Framework converters keep their native spelling — `X` for the array-based
    ones, `dataset` for PySpark — but `to_omle` auto-detects the framework,
    so it must not require the caller to already know which spelling applies.
    """

    def test_sklearn_accepts_dataset_alias(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        est = LogisticRegression(max_iter=300).fit(X, y)
        via_x = to_omle(est, X=X)
        via_dataset = to_omle(est, dataset=X)
        assert omle.to_proto_bytes(via_x) == omle.to_proto_bytes(via_dataset)

    def test_export_omle_accepts_dataset_alias(self, tmp_path, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        out = tmp_path / "aliased.omle"
        export_omle(LogisticRegression(max_iter=300).fit(X, y), str(out), dataset=X)
        assert omle.validate(omle.load(str(out))).is_valid

    def test_passing_both_is_rejected(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        est = LogisticRegression(max_iter=300).fit(X, y)
        with pytest.raises(TypeError, match="either 'X' or 'dataset'"):
            to_omle(est, X=X, dataset=X)

    def test_framework_functions_keep_their_native_spelling(self):
        """The alias is a unified-layer convenience, not a rename.

        Read the declarations from source: conftest monkey-patches the converters
        with an instrumentation wrapper whose signature is ``(*args, **kwargs)``,
        so inspecting the live attribute would tell us nothing.
        """
        import ast
        import pathlib

        import omle_convert

        root = pathlib.Path(omle_convert.__file__).parent

        def kwonly_args(path, func):
            tree = ast.parse((root / path).read_text())
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == func)
            return {a.arg for a in fn.args.kwonlyargs}

        sk = kwonly_args("sklearn/__init__.py", "from_sklearn")
        assert "X" in sk and "dataset" not in sk
        sp = kwonly_args("spark/__init__.py", "from_spark_live")
        assert "dataset" in sp and "X" not in sp
