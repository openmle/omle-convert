"""Tests for the unified entry points: ``to_omle`` and ``export_omle``.

These cover the auto-detection layer itself — object dispatch in
``_convert_object`` and path dispatch in ``_convert_from_path`` — rather than the
per-framework conversion logic, which the framework-specific suites cover.
"""

import json
import pickle
import subprocess
import sys

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


# ── A real-world mixed-type pipeline ──────────────────────────────────────────
#
# sklearn's own "Column Transformer with Mixed Types" example, with an
# XGBClassifier head: median imputation and scaling for the numeric columns,
# one-hot plus a chi2 percentile selector for the categorical ones. This is the
# shape users actually bring to a converter, and it exercises the dispatch layer
# end to end — to_omle must route a Pipeline whose final step is an XGBoost
# estimator through the sklearn walker, so the preprocessing is converted too
# rather than only the ensemble.

@pytest.fixture(scope="module")
def titanic_pipeline():
    """Fit the mixed-type Titanic pipeline; skip if OpenML is unreachable."""
    pytest.importorskip("pandas", reason="fetch_openml needs pandas for as_frame")
    xgb = pytest.importorskip("xgboost")
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.feature_selection import SelectPercentile, chi2
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    try:
        X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    except Exception as exc:  # network down, OpenML outage, no cache
        pytest.skip(f"titanic could not be fetched from OpenML: {exc}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, ["age", "fare"]),
        ("cat", categorical_transformer, ["embarked", "sex", "pclass"]),
    ])
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(n_estimators=100, max_depth=10,
                                         learning_rate=1, objective="binary:logistic",
                                         verbosity=0)),
    ])

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    clf.fit(X_train, y_train.astype(np.int64))
    return clf, X_train, X_test


class TestTitanicMixedPipeline:
    """to_omle on sklearn's mixed-type example with an XGBoost head."""

    def test_converts(self, titanic_pipeline):
        clf, X_train, _ = titanic_pipeline
        assert isinstance(to_omle(clf, X=X_train), omle.OMLEModel)

    def test_routes_through_the_sklearn_walker(self, titanic_pipeline):
        clf, X_train, _ = titanic_pipeline
        m = to_omle(clf, X=X_train)
        names = [f.name for f in m.metadata.source_frameworks]
        assert "sklearn" in names and "xgboost" in names

    def test_preprocessing_is_converted_not_dropped(self, titanic_pipeline):
        """The ColumnTransformer must survive as its own node.

        Calling from_xgboost directly emits only the ensemble, which silently
        feeds raw columns into a model trained on transformed ones. Dispatching
        through to_omle is what keeps the preprocessing.
        """
        clf, X_train, _ = titanic_pipeline
        ops = [f"{n.domain}.{n.op}" for n in to_omle(clf, X=X_train).nodes]
        assert "omle.core.Composite" in ops
        assert "omle.ml.TreeEnsemble" in ops

    def test_model_is_valid(self, titanic_pipeline):
        clf, X_train, _ = titanic_pipeline
        assert omle.validate(to_omle(clf, X=X_train)).is_valid

    def test_predict_proba_matches_native(self, titanic_pipeline, tmp_path):
        """Scores must reproduce the native pipeline's probabilities exactly.

        Still run in a subprocess: this model used to segfault the runtime on
        load, and an in-process crash takes the whole session down rather than
        failing one test. The isolation costs a process and keeps a regression
        legible.
        """
        pytest.importorskip("omle_runtime")
        clf, X_train, X_test = titanic_pipeline

        model_path = tmp_path / "titanic.omle"
        omle.save(to_omle(clf, X=X_train), model_path)
        x_path, out_path = tmp_path / "x.pkl", tmp_path / "proba.npy"
        with open(x_path, "wb") as fh:
            pickle.dump(X_test, fh)

        script = (
            "import pickle, sys, numpy as np, omle_runtime as omr\n"
            "model_path, x_path, out_path = sys.argv[1:4]\n"
            "X = pickle.load(open(x_path, 'rb'))\n"
            "np.save(out_path, omr.Model.load(model_path).predict_proba(X))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script, str(model_path), str(x_path), str(out_path)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
            pytest.fail(
                f"omle-runtime exited with {proc.returncode} "
                f"({'SIGSEGV' if proc.returncode == -11 else 'error'}): {detail}"
            )

        np.testing.assert_allclose(
            np.load(out_path), clf.predict_proba(X_test), rtol=1e-4, atol=1e-4)


# ── Single-column categorical pipelines ──────────────────────────────────────
#
# A schema whose features all share one source in column order is eligible for
# the runtime's batch path, which emits one wide tensor instead of one per
# feature. That path only produces a tensor for numeric sources, so claiming it
# for a string source left consumers wired to a value nothing produced and the
# model failed to load with "graph value 'sex__batch__' not found". One string
# column is the smallest shape that trips it.

@pytest.mark.parametrize(
    "frame,label",
    [
        ({"sex": ["m", "f", "m", "f"] * 8}, "one string column"),
        ({"sex": ["m", "f", "m", "f"] * 8, "port": ["S", "C", "Q", "S"] * 8},
         "two string columns"),
        ({"sex": ["m", "f", "m", "f"] * 8, "pclass": [1, 2, 3, 1] * 8},
         "string and integer"),
        ({"pclass": [1, 2, 3, 1] * 8}, "one integer column"),
    ],
    ids=["one_string", "two_strings", "string_and_int", "one_int"],
)
def test_categorical_pipeline_loads_and_matches(frame, label):
    pd = pytest.importorskip("pandas")
    xgb = pytest.importorskip("xgboost")
    omr = pytest.importorskip("omle_runtime")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from omle.proto.convert import ir_to_proto

    df = pd.DataFrame(frame)
    y = np.array([0, 1, 0, 1] * 8)
    pipe = Pipeline([
        ("pre", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), list(df.columns))])),
        ("clf", xgb.XGBClassifier(n_estimators=10, max_depth=3, verbosity=0)),
    ]).fit(df, y)

    # load_bytes runs the model's embedded verification, so this also asserts
    # the converted model reproduces its own recorded outputs.
    rt = omr.load_bytes(ir_to_proto(to_omle(pipe, X=df)).SerializeToString())
    np.testing.assert_allclose(
        rt.predict_proba(df), pipe.predict_proba(df), rtol=1e-6, atol=1e-6)
