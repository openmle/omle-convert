"""Tests for omle_convert.spark live PySpark converter.

Requires a local SparkSession (pyspark installed). Run with:
    pytest tests/test_spark_jvm.py

The TestPredictionMatch class additionally requires the omle-spark JAR;
those tests are automatically skipped when the JAR is not present.
"""
import numpy as np
import pytest

try:
    import pyspark.ml.classification as C
    import pyspark.ml.clustering as K
    import pyspark.ml.feature as F
    import pyspark.ml.regression as R
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.linalg import DenseVector, Vectors
    from pyspark.sql import SparkSession
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

pytestmark = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")

import os
import re
import tempfile
from pathlib import Path

import omle
from omle_convert.spark import from_spark_live

# ── omle-spark availability (for runtime prediction-match tests) ───────────
#
# The omle-spark wheel ships the JARs the JVM side needs and selects the ones
# matching the installed PySpark, so there is nothing to discover here. conftest
# puts them on the Spark class-path; this module only needs to know whether they
# are there.
#
# This replaced a hand-rolled search of a sibling omle-runtime checkout, which
# meant the tests only ran for someone who had built the Scala JAR with
# `sbt +package` — never in CI, where all 22 of them skipped.
try:
    import omle_spark
    from omle_spark import OMLEModel as _OMLEModel

    _OMLE_SPARK_JARS = omle_spark.jars()
    _HAS_OMLE_SPARK_PKG = True
except ImportError:
    _OMLEModel = None
    _OMLE_SPARK_JARS = []
    _HAS_OMLE_SPARK_PKG = False
except Exception:
    # Importable but shipping no usable JAR — a wheel built without
    # stage_jars.py, or one with no build for this PySpark's Scala version.
    _OMLEModel = None
    _OMLE_SPARK_JARS = []
    _HAS_OMLE_SPARK_PKG = True

HAS_OMLE_SPARK = bool(_OMLE_SPARK_JARS)

skip_no_omle_spark = pytest.mark.skipif(
    not HAS_OMLE_SPARK,
    reason=("omle-spark is installed but ships no JAR for this PySpark's Scala "
            "version — see spark/scripts/stage_jars.py"
            if _HAS_OMLE_SPARK_PKG
            else "omle-spark not installed (pip install omle-spark)"),
)


# ── XGBoost4J / SynapseML JAR detection (for optional Scala/SynapseML tests) ──

def _find_xgboost4j_spark_jar():
    """Same logic as conftest._find_xgboost4j_spark_jar — duplicated so skip marks
    can be evaluated at collection time without importing conftest directly."""
    env = os.environ.get("XGBOOST4J_SPARK_JAR")
    if env and Path(env).exists():
        return Path(env)
    try:
        import xgboost as _xgb
        v = _xgb.__version__
    except ImportError:
        return None
    name = f"xgboost4j-spark_2.12-{v}.jar"
    candidates = [
        Path.home() / ".m2" / "repository" / "ml" / "dmlc" / "xgboost4j-spark_2.12" / v / name,
        *Path.home().glob(f"Library/Caches/Coursier/**/{name}"),
        *Path.home().glob(f".ivy2/**/ml.dmlc/xgboost4j-spark_2.12/**/{name}"),
    ]
    return next((p for p in candidates if p.exists()), None)


def _synapseml_pyspark_compatible():
    """Return True when SynapseML publishes a build for the running Spark line.

    SynapseML ships a separate artifact per Spark line (Spark 3.5 on Scala 2.12,
    Spark 4.0 and 4.1 on Scala 2.13), so support is not "Spark 3 only" — it is
    "whichever lines currently have a published build". conftest owns the
    coordinate selection; this mirrors the availability check so skip marks can
    be evaluated at collection time.
    """
    try:
        import pyspark as _pyspark
        parts = _pyspark.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except Exception:
        return True  # assume compatible if the version cannot be read
    return (major == 3 and minor >= 5) or major == 4


def _find_synapseml_jar():
    """Same logic as conftest._find_synapseml_jar — duplicated for collection-time evaluation."""
    env = os.environ.get("SYNAPSEML_JAR")
    if env and Path(env).exists():
        return Path(env)
    try:
        import synapse.ml.lightgbm as _lgbm  # noqa: F401
        if _lgbm.__file__:
            pkg_dir = Path(os.path.dirname(_lgbm.__file__))
            for pattern in ("**/synapseml-lightgbm*.jar", "**/synapseml_*.jar"):
                jars = list(pkg_dir.rglob(pattern))
                if jars:
                    return jars[0]
        if not _synapseml_pyspark_compatible():
            return None
        return "maven"
    except ImportError:
        pass
    return None


_XGBOOST4J_JAR = _find_xgboost4j_spark_jar() if HAS_PYSPARK else None
_SYNAPSEML_JAR = _find_synapseml_jar() if HAS_PYSPARK else None
HAS_XGBOOST4J_SPARK = _XGBOOST4J_JAR is not None
HAS_SYNAPSEML = _SYNAPSEML_JAR is not None

# No skip marker for XGBoost4J: the only tests that needed one were removed
# along with the `sparkxgb` dependency (see the note further down). The JAR
# discovery above is still used — conftest puts it on the Spark classpath.

skip_no_synapseml = pytest.mark.skipif(
    not HAS_SYNAPSEML,
    reason=(
        "SynapseML not available — install via `pip install synapseml` or set "
        "SYNAPSEML_JAR to the JAR path. Note: synapseml 1.x (Scala 2.12) is "
        "incompatible with PySpark 4.x (Scala 2.13)."
    ),
)

# TEMPORARY. Remove once _load_lightgbm_spark is reworked.
#
# `_load_lightgbm_spark` pulls the booster text out of the saved model's
# metadata paramMap under the key "modelStr". That key does not exist on a
# SynapseML >= 1.1 fitted model: `modelStr` is an *estimator* param (a
# warm-start input), while the fitted model holds the booster in
# `lightGBMBooster`, declared as
#     LightGBMBoosterParam extends ComplexParam[LightGBMBooster]
# and ComplexParam values are deliberately written outside metadata by
# ComplexParamsWriter. So the booster is not in paramMap under any key, and
# reading it back needs a different approach — saveNativeModel()/getNativeModel()
# on the live model, or loading whatever ComplexParamsWriter wrote to disk.
#
# Reproducing this locally needs an x86_64 host: every entry point into
# SynapseML LightGBM (including loadNativeModelFromString) initializes the
# bundled lib_lightgbm native library, which ships x86_64-only.
skip_lightgbm_synapseml = pytest.mark.skip(
    reason=(
        "SynapseML >= 1.1 stores the booster in the lightGBMBooster ComplexParam, "
        "not in the metadata paramMap as 'modelStr' — _load_lightgbm_spark needs "
        "reworking before these can pass."
    ),
)


# ── Round-trip helpers ─────────────────────────────────────────────────────────

def _spark_pred(df_out):
    return np.array([r.prediction for r in df_out.select("prediction").collect()])

def _spark_prob(df_out):
    return np.array([r.probability.toArray() for r in df_out.select("probability").collect()])

def _to_omle(ir_model):
    pb_bytes = omle.to_proto_bytes(ir_model)
    fd, pb_path = tempfile.mkstemp(suffix=".pb")
    os.close(fd)
    with open(pb_path, "wb") as f:
        f.write(pb_bytes)
    return _OMLEModel(modelPath=pb_path)

def _omle_pred(df_out):
    return np.array([r.prediction for r in df_out.select("prediction").collect()])

def _omle_prob(df_out):
    return np.array([r.probability.toArray() for r in df_out.select("probability").collect()])



# ── Transformers ────────────────────────────────────────────────────────────────

class TestTransformers:
    def test_standard_scaler(self, binary_df):
        model = (F.StandardScaler(inputCol="features", outputCol="out", withMean=True, withStd=True)
                 .fit(binary_df))
        # n_features inferred from df; inputCol/outputCol used as model input/output names
        m = from_spark_live(model, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "StandardScaler" in ops
        node = next(n for n in m.nodes if n.op == "StandardScaler")
        attr_names = {a.name for a in node.attributes}
        assert "mean" in attr_names
        assert "scale" in attr_names
        # inputCol → model input name; outputCol → model output name
        assert m.inputs[0].name == "features"
        assert m.outputs[0].name == "out"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]
        assert m.model_schema is None

    def test_standard_scaler_std_only(self, binary_df):
        """Default StandardScaler (withMean=False, withStd=True): only 'scale' attr emitted."""
        model = F.StandardScaler(inputCol="features", outputCol="out").fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        node = next(n for n in m.nodes if n.op == "StandardScaler")
        attr_names = {a.name for a in node.attributes}
        assert "scale" in attr_names
        assert "mean" not in attr_names
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]

    def test_standard_scaler_mean_only(self, binary_df):
        """StandardScaler(withMean=True, withStd=False): only 'mean' attr emitted."""
        model = F.StandardScaler(inputCol="features", outputCol="out",
                                 withMean=True, withStd=False).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        node = next(n for n in m.nodes if n.op == "StandardScaler")
        attr_names = {a.name for a in node.attributes}
        assert "mean" in attr_names
        assert "scale" not in attr_names
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]
        # pure transformer — no ML estimator, so no targets
        assert m.model_schema is None
        # verification rows populated from df
        assert m.verification is not None

    def test_minmax_scaler(self, binary_df):
        model = F.MinMaxScaler(inputCol="features", outputCol="out").fit(binary_df)
        # n_features inferred from df's feature vector
        m = from_spark_live(model, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "MinMaxScaler" in ops
        node = next(n for n in m.nodes if n.op == "MinMaxScaler")
        attr_names = {a.name for a in node.attributes}
        assert "data_min" in attr_names
        assert "data_max" in attr_names
        assert m.inputs[0].name == "features"
        assert m.outputs[0].name == "out"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]
        assert m.model_schema is None
        assert m.verification is not None

    def test_minmax_scaler_custom_range(self, binary_df):
        """MinMaxScaler with custom feature range: feature_range_min/max attrs are set."""
        model = F.MinMaxScaler(inputCol="features", outputCol="out",
                               min=-1.0, max=1.0).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        node = next(n for n in m.nodes if n.op == "MinMaxScaler")
        attrs = {a.name: a for a in node.attributes}
        assert attrs["feature_range_min"].f64 == -1.0
        assert attrs["feature_range_max"].f64 == 1.0
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]

    def test_string_indexer(self, spark):
        data = spark.createDataFrame([("cat",), ("dog",), ("cat",), ("bird",)], ["animal"])
        model = F.StringIndexer(inputCol="animal", outputCol="idx").fit(data)
        m = from_spark_live(model, dataset=data)
        ops = [n.op for n in m.nodes]
        assert "LabelEncoder" in ops
        node = next(n for n in m.nodes if n.op == "LabelEncoder")
        assert node.name == "string_indexer_model"
        labels_attr = next(a for a in node.attributes if a.name == "labels")
        if labels_attr.tensor_ref is not None:
            strings = m.get_tensor_entry(labels_attr.tensor_ref.id).dense.string_data
        else:
            strings = labels_attr.tensor.string_data
        assert len(strings) == 3
        assert m.inputs[0].name == "animal"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "idx"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]

    def test_string_indexer_multi(self, spark):
        """StringIndexer with inputCols/outputCols: single LabelEncoder node with
        variadic xs/ys — one node handles all columns, no Concat needed."""
        data = spark.createDataFrame(
            [("cat", "red"), ("dog", "blue"), ("cat", "red"), ("bird", "green")],
            ["animal", "color"],
        )
        model = F.StringIndexer(inputCols=["animal", "color"], outputCols=["animal_idx", "color_idx"]).fit(data)
        m = from_spark_live(model, dataset=data)
        ops = [n.op for n in m.nodes]
        # single LabelEncoder node with two inputs/outputs — no separate nodes per column
        assert ops.count("LabelEncoder") == 1
        node = next(n for n in m.nodes if n.op == "LabelEncoder")
        assert node.name == "string_indexer_model"
        assert {inp.name for inp in node.inputs} == {"animal", "color"}
        # inputs: one per inputCol, typed as STRING
        input_names = {i.name for i in m.inputs}
        assert "animal" in input_names
        assert "color" in input_names
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.STRING
        for inp in m.inputs:
            assert inp.type.shape == [-1]
        # labels tensor: concatenated strings for both columns (3 + 3 = 6 total)
        labels_attr = next(a for a in node.attributes if a.name == "labels")
        if labels_attr.tensor_ref is not None:
            all_labels = list(m.get_tensor_entry(labels_attr.tensor_ref.id).dense.string_data)
        else:
            all_labels = list(labels_attr.tensor.string_data)
        assert len(all_labels) == 6  # 3 animal labels + 3 color labels
        # label_offsets has 3 entries: [0, 3, 6]
        offsets_attr = next(a for a in node.attributes if a.name == "label_offsets")
        if offsets_attr.tensor_ref is not None:
            offsets = list(m.get_tensor_entry(offsets_attr.tensor_ref.id).dense.int64_data)
        else:
            offsets = list(offsets_attr.tensor.int64_data)
        assert offsets == [0, 3, 6]
        # node outputs match outputCols
        node_out_names = {o.name for o in node.outputs}
        assert node_out_names == {"animal_idx", "color_idx"}
        # model outputs: one per outputCol
        out_map = {o.name: o for o in m.outputs}
        assert "animal_idx" in out_map
        assert "color_idx" in out_map
        for out in m.outputs:
            assert out.type.dtype == omle.DataType.FLOAT64
            assert out.type.shape == [-1]
        # pure transformer — no ML estimator, so no model schema
        assert m.model_schema is None

    def test_one_hot_encoder(self, spark):
        data = spark.createDataFrame([(0.0,), (1.0,), (2.0,), (0.0,)], ["idx"])
        model = F.OneHotEncoder(inputCol="idx", outputCol="ohe").fit(data)
        model.transform(data).printSchema()
        model.transform(data).show()
        m = from_spark_live(model, dataset=data)
        ops = [n.op for n in m.nodes]
        assert "OneHotEncoder" in ops
        assert m.inputs[0].name == "idx"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "ohe"
        # 3 categories with dropLast=True → output vector has 2 elements per row
        assert m.outputs[0].type is not None
        assert m.outputs[0].type.shape == [-1, 2]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64

    def test_one_hot_encoder_input_cols_multi(self, spark):
        """OHE with plural inputCols/outputCols: single variadic node with named ys outputs."""
        data = spark.createDataFrame(
            [(0.0, 1.0), (1.0, 0.0), (2.0, 2.0), (0.0, 1.0)],
            ["idx1", "idx2"],
        )
        model = F.OneHotEncoder(inputCols=["idx1", "idx2"], outputCols=["ohe1", "ohe2"]).fit(data)
        m = from_spark_live(model, dataset=data)
        ops = [n.op for n in m.nodes]
        # single variadic node — no per-column nodes
        assert ops.count("OneHotEncoder") == 1
        node = next(n for n in m.nodes if n.op == "OneHotEncoder")
        assert node.name == "one_hot_encoder_model"
        assert {inp.name for inp in node.inputs} == {"idx1", "idx2"}
        # categories and category_offsets attributes present
        attr_names = {a.name for a in node.attributes}
        assert "categories" in attr_names
        assert "category_offsets" in attr_names
        # two named outputs, one per inputCol, shape [-1, 2] each (3 cats, dropLast=True)
        node_out_map = {o.name: o for o in node.outputs}
        assert "ohe1" in node_out_map
        assert "ohe2" in node_out_map
        assert node_out_map["ohe1"].type.shape == [-1, 2]
        assert node_out_map["ohe2"].type.shape == [-1, 2]
        # model outputs: one OutputSpec per outputCols entry
        out_map = {o.name: o for o in m.outputs}
        assert "ohe1" in out_map
        assert "ohe2" in out_map
        assert out_map["ohe1"].type.shape == [-1, 2]
        assert out_map["ohe2"].type.shape == [-1, 2]
        in_map = {i.name: i for i in m.inputs}
        assert "idx1" in in_map
        assert "idx2" in in_map
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        assert m.model_schema is None

    def test_imputer_single_col(self, spark):
        """Imputer with singular inputCol/outputCol."""
        data = spark.createDataFrame([(1.0,), (None,), (5.0,), (3.0,)], ["a"])
        model = F.Imputer(inputCol="a", outputCol="a_imp").fit(data)
        m = from_spark_live(model, dataset=data)
        assert any(n.op == "Imputer" for n in m.nodes)
        assert m.inputs[0].name == "a"
        assert m.outputs[0].name == "a_imp"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]

    def test_imputer_multi(self, spark):
        """Imputer with inputCols/outputCols: single variadic node with named xs/ys."""
        data = spark.createDataFrame(
            [(1.0, 10.0), (None, 20.0), (3.0, None), (2.0, 40.0)], ["a", "b"]
        )
        model = F.Imputer(inputCols=["a", "b"], outputCols=["a_imp", "b_imp"]).fit(data)
        m = from_spark_live(model, dataset=data)
        assert [n.op for n in m.nodes].count("Imputer") == 1
        node = next(n for n in m.nodes if n.op == "Imputer")
        assert node.name == "imputer_model"
        assert {inp.name for inp in node.inputs} == {"a", "b"}
        assert {o.name for o in node.outputs} == {"a_imp", "b_imp"}
        assert any(a.name == "fill_tensor" for a in node.attributes)
        out_map = {o.name: o for o in m.outputs}
        assert "a_imp" in out_map
        assert "b_imp" in out_map
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        for out in m.outputs:
            assert out.type.dtype == omle.DataType.FLOAT64
            assert out.type.shape == [-1]
        assert m.model_schema is None

    def test_binarizer(self, binary_df):
        t = F.Binarizer(inputCol="features", outputCol="bin", threshold=3.0)
        m = from_spark_live(t, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "Binarizer" in ops
        node = next(n for n in m.nodes if n.op == "Binarizer")
        thr_attr = next(a for a in node.attributes if a.name == "thresholds")
        thr_vals = (thr_attr.tensor.float64_data if thr_attr.tensor_ref is None
                    else m.get_tensor_entry(thr_attr.tensor_ref.id).dense.float64_data)
        assert list(thr_vals) == [3.0]
        assert m.inputs[0].name == "features"
        assert m.outputs[0].name == "bin"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]

    def test_binarizer_multi(self, spark):
        """Binarizer with inputCols/outputCols: single variadic node, per-column thresholds."""
        data = spark.createDataFrame([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], ["a", "b"])
        t = F.Binarizer(inputCols=["a", "b"], outputCols=["a_bin", "b_bin"], thresholds=[2.5, 4.0])
        m = from_spark_live(t, dataset=data)
        assert [n.op for n in m.nodes].count("Binarizer") == 1
        node = next(n for n in m.nodes if n.op == "Binarizer")
        assert node.name == "binarizer"
        assert {inp.name for inp in node.inputs} == {"a", "b"}
        assert {o.name for o in node.outputs} == {"a_bin", "b_bin"}
        thr_attr = next(a for a in node.attributes if a.name == "thresholds")
        thr_vals = (thr_attr.tensor.float64_data if thr_attr.tensor_ref is None
                    else m.get_tensor_entry(thr_attr.tensor_ref.id).dense.float64_data)
        assert list(thr_vals) == [2.5, 4.0]
        out_map = {o.name: o for o in m.outputs}
        assert "a_bin" in out_map
        assert "b_bin" in out_map
        in_map = {i.name: i for i in m.inputs}
        assert "a" in in_map
        assert "b" in in_map
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        for out in m.outputs:
            assert out.type.dtype == omle.DataType.FLOAT64
            assert out.type.shape == [-1]
        assert m.model_schema is None

    def test_pca(self, binary_df):
        model = F.PCA(k=2, inputCol="features", outputCol="pca").fit(binary_df)
        # n_features inferred from df's feature vector
        m = from_spark_live(model, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "PCA" in ops
        node = next(n for n in m.nodes if n.op == "PCA")
        attr_names = {a.name for a in node.attributes}
        assert "components" in attr_names
        assert "n_components" in attr_names
        assert m.inputs[0].name == "features"
        assert m.outputs[0].name == "pca"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 2]

    def test_max_abs_scaler(self, binary_df):
        model = F.MaxAbsScaler(inputCol="features", outputCol="out").fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "MaxAbsScaler" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "MaxAbsScaler")
        assert any(a.name == "scale" for a in node.attributes)
        assert m.inputs[0].name == "features"
        assert m.outputs[0].name == "out"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]
        assert m.model_schema is None

    def test_normalizer(self, binary_df):
        t = F.Normalizer(inputCol="features", outputCol="norm_out", p=2.0)
        m = from_spark_live(t, dataset=binary_df)
        assert any(n.op == "Normalizer" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "Normalizer")
        assert any(a.name == "p" for a in node.attributes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "norm_out"
        assert m.outputs[0].type.shape == [-1]  # stateless transformer; n_features not available at load time
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema is None

    def test_bucketizer(self, spark):
        data = spark.createDataFrame([(float(x),) for x in range(-4, 5)], ["val"])
        t = F.Bucketizer(inputCol="val", outputCol="bucket",
                         splits=[-float("inf"), -1.0, 1.0, float("inf")])
        m = from_spark_live(t, dataset=data)
        assert any(n.op == "Bucketizer" for n in m.nodes)
        assert m.inputs[0].name == "val"
        assert m.inputs[0].type is not None
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "bucket"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert m.model_schema is None

    def test_bucketizer_multi(self, spark):
        """Bucketizer with inputCols/outputCols/splitsArray: single variadic node."""
        data = spark.createDataFrame([(1.0, 5.0), (3.0, 15.0), (7.0, 25.0)], ["x", "y"])
        t = F.Bucketizer(
            inputCols=["x", "y"], outputCols=["x_buck", "y_buck"],
            splitsArray=[[-float("inf"), 2.0, 6.0, float("inf")],
                         [-float("inf"), 10.0, 20.0, float("inf")]],
        )
        m = from_spark_live(t, dataset=data)
        assert [n.op for n in m.nodes].count("Bucketizer") == 1
        node = next(n for n in m.nodes if n.op == "Bucketizer")
        assert node.name == "bucketizer"
        assert {inp.name for inp in node.inputs} == {"x", "y"}
        assert {o.name for o in node.outputs} == {"x_buck", "y_buck"}
        attr_names = {a.name for a in node.attributes}
        assert "boundaries" in attr_names
        assert "boundary_offsets" in attr_names
        out_map = {o.name: o for o in m.outputs}
        assert "x_buck" in out_map
        assert "y_buck" in out_map
        in_map = {i.name: i for i in m.inputs}
        assert "x" in in_map
        assert "y" in in_map
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        for out in m.outputs:
            assert out.type.dtype == omle.DataType.FLOAT64
            assert out.type.shape == [-1]
        assert m.model_schema is None

    def test_elementwise_product(self, binary_df):
        t = F.ElementwiseProduct(
            inputCol="features", outputCol="scaled",
            scalingVec=Vectors.dense(1.0, 2.0, 3.0),
        )
        m = from_spark_live(t, dataset=binary_df)
        assert any(n.op == "WeightedSum" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "WeightedSum")
        assert node.name == "elementwise_product"
        assert node.inputs[0].name == "features"
        assert node.outputs[0].name == "scaled"
        weights_attr = next(a for a in node.attributes if a.name == "weights")
        weights = weights_attr.tensor.float64_data
        assert weights == [1.0, 2.0, 3.0]
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "scaled"
        assert m.outputs[0].type.shape == [-1, 3]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema is None

    def test_vector_slicer(self, binary_df):
        t = F.VectorSlicer(inputCol="features", outputCol="sliced", indices=[0, 2])
        m = from_spark_live(t, dataset=binary_df)
        assert any(n.op == "TakeSlots" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TakeSlots")
        assert node.name == "vector_slicer"
        assert node.inputs[0].name == "features"
        assert node.outputs[0].name == "sliced"
        assert list(node.attributes[0].ints) == [0, 2]
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "sliced"
        assert m.outputs[0].type.shape == [-1, 2]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema is None

    def test_chi_sq_selector(self, nonneg_df):
        model = F.ChiSqSelector(
            numTopFeatures=2, featuresCol="features", labelCol="label", outputCol="selected"
        ).fit(nonneg_df)
        m = from_spark_live(model, dataset=nonneg_df)
        assert any(n.op == "TakeSlots" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TakeSlots")
        assert len(node.attributes[0].ints) == 2
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "selected"
        assert m.outputs[0].type.shape == [-1, 2]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema is None

    def test_univariate_feature_selector(self, nonneg_df):
        if not hasattr(F, "UnivariateFeatureSelector"):
            pytest.skip("UnivariateFeatureSelector requires Spark 3.1+")
        model = (F.UnivariateFeatureSelector(
            featuresCol="features", outputCol="selected",
            labelCol="label", selectionMode="numTopFeatures",
        ).setFeatureType("categorical").setLabelType("categorical")
         .setSelectionThreshold(2).fit(nonneg_df))
        m = from_spark_live(model, dataset=nonneg_df)
        assert any(n.op == "TakeSlots" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TakeSlots")
        assert len(node.attributes[0].ints) == 2
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "selected"
        assert m.outputs[0].type.shape == [-1, 2]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema is None

    def test_variance_threshold_selector(self, binary_df):
        if not hasattr(F, "VarianceThresholdSelector"):
            pytest.skip("VarianceThresholdSelector requires Spark 3.1+")
        model = F.VarianceThresholdSelector(
            varianceThreshold=0.0, featuresCol="features", outputCol="selected"
        ).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "TakeSlots" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TakeSlots")
        assert len(node.attributes[0].ints) == 3  # threshold=0.0 keeps all 3 features
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert m.outputs[0].name == "selected"
        assert m.outputs[0].type.shape == [-1, 3]
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema is None


# ── Text Transformers ────────────────────────────────────────────────────────────

class TestTextTransformers:
    def test_tokenizer(self, text_df):
        t = F.Tokenizer(inputCol="text", outputCol="tokens")
        m = from_spark_live(t, dataset=text_df)
        assert any(n.op == "Tokenizer" for n in m.nodes)
        assert m.inputs[0].name == "text"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "tokens"
        assert m.outputs[0].type.dtype == omle.DataType.STRING
        assert m.outputs[0].type.shape == [-1, -1]
        assert m.model_schema is None

    def test_regex_tokenizer(self, text_df):
        t = F.RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r"\W+")
        m = from_spark_live(t, dataset=text_df)
        assert any(n.op == "RegexTokenizer" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "RegexTokenizer")
        assert any(a.name == "pattern" for a in node.attributes)
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].name == "text"
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "tokens"
        assert m.outputs[0].type.dtype == omle.DataType.STRING
        assert m.outputs[0].type.shape == [-1, -1]
        assert m.model_schema is None

    def test_stop_words_remover(self, text_df):
        tokens_df = F.Tokenizer(inputCol="text", outputCol="tokens").transform(text_df)
        t = F.StopWordsRemover(
            inputCol="tokens", outputCol="filtered",
            stopWords=["hello", "foo", "bar"],
        )
        m = from_spark_live(t, dataset=tokens_df)
        assert any(n.op == "StopWordsRemover" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "StopWordsRemover")
        assert any(a.name == "stop_words" for a in node.attributes)
        assert m.inputs[0].name == "tokens"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1, -1]
        assert m.outputs[0].name == "filtered"
        assert m.outputs[0].type.dtype == omle.DataType.STRING
        assert m.outputs[0].type.shape == [-1, -1]
        assert m.model_schema is None

    def test_ngram(self, text_df):
        tokens_df = F.Tokenizer(inputCol="text", outputCol="tokens").transform(text_df)
        t = F.NGram(inputCol="tokens", outputCol="ngrams", n=2)
        m = from_spark_live(t, dataset=tokens_df)
        assert any(n.op == "NGram" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "NGram")
        assert any(a.name in ("n_min", "n_max") for a in node.attributes)
        assert m.inputs[0].name == "tokens"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1, -1]
        assert m.outputs[0].name == "ngrams"
        assert m.outputs[0].type.dtype == omle.DataType.STRING
        assert m.outputs[0].type.shape == [-1, -1]
        assert m.model_schema is None

    def test_count_vectorizer(self, text_df):
        tokens_df = F.Tokenizer(inputCol="text", outputCol="tokens").transform(text_df)
        model = F.CountVectorizer(inputCol="tokens", outputCol="tf", minDF=1.0).fit(tokens_df)
        m = from_spark_live(model, dataset=tokens_df)
        assert any(n.op == "CountVectorizer" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "CountVectorizer")
        assert any(a.name == "vocabulary" for a in node.attributes)
        assert m.inputs[0].name == "tokens"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1, -1]
        assert m.outputs[0].name == "tf"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert m.model_schema is None

    def test_hashing_tf(self, text_df):
        tokens_df = F.Tokenizer(inputCol="text", outputCol="tokens").transform(text_df)
        t = F.HashingTF(inputCol="tokens", outputCol="tf", numFeatures=100)
        m = from_spark_live(t, dataset=tokens_df)
        assert any(n.op == "HashingVectorizer" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "HashingVectorizer")
        assert any(a.name == "num_features" for a in node.attributes)
        assert m.inputs[0].name == "tokens"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1, -1]
        assert m.outputs[0].name == "tf"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert m.model_schema is None

    def test_idf(self, text_df):
        tokens_df = F.Tokenizer(inputCol="text", outputCol="tokens").transform(text_df)
        tf_df = F.HashingTF(inputCol="tokens", outputCol="tf", numFeatures=100).transform(tokens_df)
        model = F.IDF(inputCol="tf", outputCol="tfidf").fit(tf_df)
        m = from_spark_live(model, dataset=tf_df)
        assert any(n.op == "TfIdfTransformer" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TfIdfTransformer")
        assert any(a.name == "idf" for a in node.attributes)
        assert m.inputs[0].name == "tf"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 100]
        assert m.outputs[0].name == "tfidf"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert m.model_schema is None

    def test_word2vec(self, text_df):
        tokens_df = F.Tokenizer(inputCol="text", outputCol="tokens").transform(text_df)
        model = F.Word2Vec(
            inputCol="tokens", outputCol="w2v", vectorSize=10, minCount=0, seed=42
        ).fit(tokens_df)
        m = from_spark_live(model, dataset=tokens_df)
        assert any(n.op == "Word2Vec" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "Word2Vec")
        assert any(a.name == "vocabulary" for a in node.attributes)
        assert any(a.name == "embeddings" for a in node.attributes)
        assert m.inputs[0].name == "tokens"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1, -1]
        assert m.outputs[0].name == "w2v"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert m.model_schema is None

    # ── Pipeline tests ────────────────────────────────────────────────────────────

    def test_tokenizer_count_vectorizer_pipeline(self, text_df):
        pipeline = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.CountVectorizer(inputCol="tokens", outputCol="tf", minDF=1.0),
        ])
        model = pipeline.fit(text_df)
        m = from_spark_live(model, dataset=text_df)
        assert any(n.op == "Tokenizer" for n in m.nodes)
        assert any(n.op == "CountVectorizer" for n in m.nodes)
        tok_node = next(n for n in m.nodes if n.op == "Tokenizer")
        assert tok_node.outputs[0].name == "tokens"
        cv_node = next(n for n in m.nodes if n.op == "CountVectorizer")
        assert cv_node.outputs[0].name == "tf"
        assert m.inputs[0].name == "text"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "tf"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert m.model_schema is None

    def test_tfidf_pipeline(self, text_df):
        pipeline = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.HashingTF(inputCol="tokens", outputCol="tf", numFeatures=100),
            F.IDF(inputCol="tf", outputCol="tfidf"),
        ])
        model = pipeline.fit(text_df)
        m = from_spark_live(model, dataset=text_df)
        assert any(n.op == "Tokenizer" for n in m.nodes)
        assert any(n.op == "HashingVectorizer" for n in m.nodes)
        assert any(n.op == "TfIdfTransformer" for n in m.nodes)
        tok_node = next(n for n in m.nodes if n.op == "Tokenizer")
        assert tok_node.outputs[0].name == "tokens"
        hv_node = next(n for n in m.nodes if n.op == "HashingVectorizer")
        assert hv_node.outputs[0].name == "tf"
        tfidf_node = next(n for n in m.nodes if n.op == "TfIdfTransformer")
        assert tfidf_node.outputs[0].name == "tfidf"
        assert m.inputs[0].name == "text"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "tfidf"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert m.model_schema is None

    def test_stop_words_ngram_pipeline(self, text_df):
        pipeline = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.StopWordsRemover(
                inputCol="tokens", outputCol="filtered",
                stopWords=["hello", "foo", "bar"],
            ),
            F.NGram(inputCol="filtered", outputCol="ngrams", n=2),
        ])
        model = pipeline.fit(text_df)
        m = from_spark_live(model, dataset=text_df)
        assert any(n.op == "Tokenizer" for n in m.nodes)
        assert any(n.op == "StopWordsRemover" for n in m.nodes)
        assert any(n.op == "NGram" for n in m.nodes)
        tok_node = next(n for n in m.nodes if n.op == "Tokenizer")
        assert tok_node.outputs[0].name == "tokens"
        swr_node = next(n for n in m.nodes if n.op == "StopWordsRemover")
        assert swr_node.outputs[0].name == "filtered"
        ng_node = next(n for n in m.nodes if n.op == "NGram")
        assert ng_node.outputs[0].name == "ngrams"
        assert m.inputs[0].name == "text"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        assert m.outputs[0].name == "ngrams"
        assert m.outputs[0].type.dtype == omle.DataType.STRING
        assert m.outputs[0].type.shape == [-1, -1]
        assert m.model_schema is None


# ── Estimators ─────────────────────────────────────────────────────────────────

class TestEstimators:
    def test_logistic_regression_binary(self, binary_df):
        model = C.LogisticRegression(
            maxIter=10, featuresCol="features",
            predictionCol="lr_pred", probabilityCol="lr_prob",
        ).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "Linear" in ops
        node = next(n for n in m.nodes if n.op == "Linear")
        assert node.linear is not None
        assert node.linear.post_transform.name == "SIGMOID"
        output_names = {o.name for o in m.outputs}
        assert "lr_pred" in output_names
        assert "lr_prob" in output_names
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        lr_pred_out = next(o for o in m.outputs if o.name == "lr_pred")
        assert lr_pred_out.type.dtype == omle.DataType.INT64
        assert lr_pred_out.type.shape == [-1]
        lr_prob_out = next(o for o in m.outputs if o.name == "lr_prob")
        assert lr_prob_out.type.dtype == omle.DataType.FLOAT64
        assert lr_prob_out.type.shape == [-1, 2]

    def test_linear_regression(self, regression_df):
        model = R.LinearRegression(
            maxIter=10, featuresCol="features", labelCol="label",
            predictionCol="reg_pred",
        ).fit(regression_df)
        m = from_spark_live(model, dataset=regression_df)
        ops = [n.op for n in m.nodes]
        assert "Linear" in ops
        output_names = {o.name for o in m.outputs}
        assert "reg_pred" in output_names
        assert "probability" not in output_names
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        reg_pred_out = next(o for o in m.outputs if o.name == "reg_pred")
        assert reg_pred_out.type.dtype == omle.DataType.FLOAT64
        assert reg_pred_out.type.shape == [-1]
        assert m.model_schema.targets[0].name == "label"

    def test_decision_tree_classifier(self, binary_df):
        model = C.DecisionTreeClassifier(maxDepth=3).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "TreeEnsemble" in ops
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble is not None
        trees = node.tree_ensemble.trees
        assert len(trees) == 1
        assert trees[0].num_nodes > 0
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        dt_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert dt_pred_out.type.dtype == omle.DataType.INT64
        assert dt_pred_out.type.shape == [-1]
        dt_prob_out = next(o for o in m.outputs if o.name == "probability")
        assert dt_prob_out.type.dtype == omle.DataType.FLOAT64
        assert dt_prob_out.type.shape == [-1, 2]

    def test_decision_tree_regressor(self, regression_df):
        df = regression_df.withColumnRenamed("features", "feats")
        model = R.DecisionTreeRegressor(maxDepth=3, featuresCol="feats").fit(df)
        m = from_spark_live(model, dataset=df)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        assert m.inputs[0].name == "feats"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        dtr_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert dtr_pred_out.type.dtype == omle.DataType.FLOAT64
        assert dtr_pred_out.type.shape == [-1]

    def test_random_forest_classifier_binary(self, binary_df):
        model = C.RandomForestClassifier(
            numTrees=3, maxDepth=2, seed=42,
            predictionCol="rf_pred", probabilityCol="rf_prob",
        ).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert len(node.tree_ensemble.trees) == 3
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "rf_pred" in output_names
        assert "rf_prob" in output_names
        rf_pred_out = next(o for o in m.outputs if o.name == "rf_pred")
        assert rf_pred_out.type.dtype == omle.DataType.INT64
        assert rf_pred_out.type.shape == [-1]
        rf_prob_out = next(o for o in m.outputs if o.name == "rf_prob")
        assert rf_prob_out.type.dtype == omle.DataType.FLOAT64
        assert rf_prob_out.type.shape == [-1, 2]

    def test_random_forest_regressor(self, regression_df):
        model = R.RandomForestRegressor(numTrees=3, maxDepth=2, seed=42).fit(regression_df)
        m = from_spark_live(model, dataset=regression_df)
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble.aggregation.name == "AVERAGE"
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"prediction"}
        rfr_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert rfr_pred_out.type.dtype == omle.DataType.FLOAT64
        assert rfr_pred_out.type.shape == [-1]

    def test_gbt_classifier(self, binary_df):
        df = binary_df.withColumnRenamed("features", "raw_feats")
        model = C.GBTClassifier(
            maxIter=3, maxDepth=2, seed=42, featuresCol="raw_feats",
        ).fit(df)
        m = from_spark_live(model, dataset=df)
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble.post_transform.name == "SIGMOID"
        assert node.tree_ensemble.aggregation.name == "SUM"
        assert m.inputs[0].name == "raw_feats"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert "prediction" in {o.name for o in m.outputs}
        gbt_cls_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert gbt_cls_pred_out.type.dtype == omle.DataType.INT64
        assert gbt_cls_pred_out.type.shape == [-1]

    def test_gbt_regressor(self, regression_df):
        model = R.GBTRegressor(
            maxIter=3, maxDepth=2, seed=42, predictionCol="gbt_pred",
        ).fit(regression_df)
        m = from_spark_live(model, dataset=regression_df)
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble.aggregation.name == "SUM"
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"gbt_pred"}
        gbt_pred_out = next(o for o in m.outputs if o.name == "gbt_pred")
        assert gbt_pred_out.type.dtype == omle.DataType.FLOAT64
        assert gbt_pred_out.type.shape == [-1]

    def test_naive_bayes(self, nonneg_df):
        model = C.NaiveBayes(modelType="multinomial").fit(nonneg_df)
        m = from_spark_live(model, dataset=nonneg_df)
        assert any(n.op == "NaiveBayes" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert "prediction" in {o.name for o in m.outputs}
        assert "probability" in {o.name for o in m.outputs}
        nb_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert nb_pred_out.type.dtype == omle.DataType.INT64
        assert nb_pred_out.type.shape == [-1]
        nb_prob_out = next(o for o in m.outputs if o.name == "probability")
        assert nb_prob_out.type.dtype == omle.DataType.FLOAT64
        assert nb_prob_out.type.shape == [-1, 2]

    def test_kmeans(self, binary_df):
        model = K.KMeans(k=2, seed=42, featuresCol="features").fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "Clustering" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "Clustering")
        assert node.clustering is not None
        assert node.clustering.prototype is not None
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_roles = {o.role for o in m.outputs}
        assert omle.OutputRole.ENTITY_ID in output_roles
        assert {o.name for o in m.outputs} == {"prediction"}  # predictionCol default
        km_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert km_pred_out.type.dtype == omle.DataType.INT32
        assert km_pred_out.type.shape == [-1]
        assert m.model_schema.targets == []  # clustering is unsupervised

    def test_gaussian_mixture(self, binary_df):
        model = K.GaussianMixture(k=2, seed=42, featuresCol="features").fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "Clustering" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "Clustering")
        assert node.clustering is not None
        assert node.clustering.gaussian_mixture is not None
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        gm_pred_out = next(o for o in m.outputs if o.role == omle.OutputRole.ENTITY_ID)
        assert gm_pred_out.type.dtype == omle.DataType.INT32
        assert gm_pred_out.type.shape == [-1]
        assert m.model_schema.targets == []  # clustering is unsupervised

    def test_linear_svc(self, binary_df):
        model = C.LinearSVC(maxIter=10).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "SVM" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        svc_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert svc_pred_out.type.dtype == omle.DataType.INT64
        assert svc_pred_out.type.shape == [-1]

    def test_mlp_classifier(self, binary_df):
        model = C.MultilayerPerceptronClassifier(
            layers=[3, 4, 2], maxIter=10, seed=42
        ).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "NeuralNetwork" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "NeuralNetwork")
        assert node.neural_network is not None
        assert len(node.neural_network.layers) == 2  # 3→4 layer, 4→2 layer
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert "prediction" in {o.name for o in m.outputs}
        assert "probability" in {o.name for o in m.outputs}
        mlp_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert mlp_pred_out.type.dtype == omle.DataType.INT64
        assert mlp_pred_out.type.shape == [-1]
        mlp_prob_out = next(o for o in m.outputs if o.name == "probability")
        assert mlp_prob_out.type.dtype == omle.DataType.FLOAT64
        assert mlp_prob_out.type.shape == [-1, 2]

    def test_generalized_linear_regression(self, regression_df):
        model = R.GeneralizedLinearRegression(
            family="gaussian", link="identity", maxIter=10
        ).fit(regression_df)
        m = from_spark_live(model, dataset=regression_df)
        assert any(n.op == "Linear" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" not in output_names
        glr_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert glr_pred_out.type.dtype == omle.DataType.FLOAT64
        assert glr_pred_out.type.shape == [-1]

    def test_isotonic_regression(self, regression_df):
        model = R.IsotonicRegression(featuresCol="features", featureIndex=0).fit(regression_df)
        m = from_spark_live(model, dataset=regression_df)
        assert any(n.op == "NormContinuous" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        iso_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert iso_pred_out.type.dtype == omle.DataType.FLOAT64
        assert iso_pred_out.type.shape == [-1]

    def test_one_vs_rest(self, multiclass_df):
        lr = C.LogisticRegression(maxIter=5)
        model = C.OneVsRest(classifier=lr).fit(multiclass_df)
        m = from_spark_live(model, dataset=multiclass_df)
        # Spark's OneVsRest has no probabilityCol — only a prediction is produced.
        assert any(n.op == "ArgMax" for n in m.nodes)
        assert not any(n.op == "SoftVote" for n in m.nodes)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]

        # One binary sub-model per class: the Concat node must have 3 inputs (K=3).
        concat_node = next(n for n in m.nodes if n.op == "Concat")
        assert len(concat_node.inputs) == 3, \
            f"expected 3 class scores in Concat, got {len(concat_node.inputs)}"

        # Only the winning-class prediction should be exported — no sub-classifier outputs,
        # no probability (Spark OvR does not compute normalised class probabilities).
        pred_outputs = [o for o in m.outputs if o.role == omle.OutputRole.PREDICTION]
        prob_outputs = [o for o in m.outputs if o.role == omle.OutputRole.PROBABILITY]
        assert len(pred_outputs) == 1, \
            f"expected exactly 1 PREDICTION output, got {[o.name for o in pred_outputs]}"
        assert len(prob_outputs) == 0, \
            f"expected no PROBABILITY output, got {[o.name for o in prob_outputs]}"

        # Model schema: 3 class labels [0, 1, 2] inferred from numClasses.
        target = m.model_schema.targets[0]
        assert len(target.class_labels) == 3, \
            f"expected 3 class labels, got {target.class_labels}"

        # Multiclass prediction: winning class index (INT64, shape [n]).
        pred = pred_outputs[0]
        assert pred.type.dtype == omle.DataType.INT64
        assert pred.type.shape == [-1]

    def test_cross_validator(self, binary_df):
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        lr = C.LogisticRegression(maxIter=5)
        grid = ParamGridBuilder().addGrid(lr.maxIter, [5]).build()
        cv = CrossValidator(
            estimator=lr,
            estimatorParamMaps=grid,
            evaluator=BinaryClassificationEvaluator(),
            numFolds=2,
            seed=42,
        )
        model = cv.fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "Linear" for n in m.nodes)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_train_validation_split(self, binary_df):
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
        lr = C.LogisticRegression(maxIter=5)
        grid = ParamGridBuilder().addGrid(lr.maxIter, [5]).build()
        tvs = TrainValidationSplit(
            estimator=lr,
            estimatorParamMaps=grid,
            evaluator=BinaryClassificationEvaluator(),
            trainRatio=0.8,
            seed=42,
        )
        model = tvs.fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "Linear" for n in m.nodes)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]


# ── Pipeline ────────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_scaler_plus_lr_pipeline(self, binary_df):
        pipeline = Pipeline(stages=[
            F.StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True),
            C.LogisticRegression(featuresCol="features_scaled", maxIter=10),
        ])
        model = pipeline.fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "StandardScaler" in ops
        assert "Linear" in ops
        ns = {ni.namespace for ni in m.operator_imports}
        assert "omle.feature" in ns
        assert "omle.ml" in ns
        # input: first stage's inputCol; outputs: last estimator's predictionCol/probabilityCol
        assert m.inputs[0].name == "features"
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]
        assert m.model_schema.targets[0].name == "label"
        # node name: snake_case of Spark class (uniform across standalone + pipeline)
        scaler_node = next(n for n in m.nodes if n.op == "StandardScaler")
        assert scaler_node.name == "standard_scaler_model"
        # node output tensor: outputCol value (matches sklearn {input}_{alias} convention)
        assert scaler_node.outputs[0].name == "features_scaled"
        assert scaler_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert scaler_node.outputs[0].type.shape == [-1, 3]

    def test_rf_pipeline(self, binary_df):
        pipeline = Pipeline(stages=[
            C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42),
        ])
        model = pipeline.fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        assert m.inputs[0].name == "features"
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]
        # node name: snake_case of Spark class
        rf_node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert rf_node.name == "random_forest_classification_model"

    def test_pipeline_with_vector_assembler(self, spark):
        data = spark.createDataFrame([
            (1.0, 2.0, 0.0),
            (3.0, 4.0, 1.0),
            (5.0, 6.0, 0.0),
            (7.0, 8.0, 1.0),
        ], ["a", "b", "label"])
        data.printSchema()
        pipeline = Pipeline(stages=[
            F.VectorAssembler(inputCols=["a", "b"], outputCol="features"),
            C.LogisticRegression(maxIter=5, featuresCol="features"),
        ])
        model = pipeline.fit(data)
        m = from_spark_live(model, dataset=data, feature_names=["a", "b"])
        ops = [n.op for n in m.nodes]
        assert "Concat" in ops
        assert "Linear" in ops
        # Inputs: per-column FLOAT64 scalars (a, b)
        assert {i.name for i in m.inputs} == {"a", "b"}
        for i in m.inputs:
            assert i.type.dtype == omle.DataType.FLOAT64
            assert i.type.shape == [-1]
        # Schema features also use FLOAT64
        assert all(f.type.dtype == omle.DataType.FLOAT64 for f in m.model_schema.features)
        # Concat (VectorAssembler): node name, output name, dtype, shape, field_names
        concat_node = next(n for n in m.nodes if n.op == "Concat")
        assert concat_node.name == "vector_assembler"
        co = concat_node.outputs[0]
        assert co.name == "features"                     # outputCol
        assert co.type.dtype  == omle.DataType.FLOAT64
        assert co.type.shape  == [-1, 2]                 # 2 input columns
        assert co.field_names == ["a", "b"]              # inputCols
        # Model outputs: Spark column names, correct dtypes and shapes
        out_map = {o.name: o for o in m.outputs}
        assert "prediction"  in out_map
        assert "probability" in out_map
        assert out_map["prediction"].type.dtype  == omle.DataType.INT64
        assert out_map["prediction"].type.shape  == [-1]
        assert out_map["probability"].type.dtype == omle.DataType.FLOAT64
        assert out_map["probability"].type.shape == [-1, 2]

    def test_mixed_numeric_assembler_rf_pipeline(self, spark_mixed_df):
        """VectorAssembler from scalar numeric/int columns + RandomForest."""
        pipeline = Pipeline(stages=[
            F.VectorAssembler(inputCols=["size", "weight", "is_premium"], outputCol="features"),
            C.RandomForestClassifier(featuresCol="features", numTrees=5, maxDepth=2, seed=42),
        ])
        model = pipeline.fit(spark_mixed_df)
        m = from_spark_live(model, dataset=spark_mixed_df, feature_names=["size", "weight", "is_premium"])
        ops = [n.op for n in m.nodes]
        assert "Concat" in ops
        assert "TreeEnsemble" in ops
        input_names = {inp.name for inp in m.inputs}
        assert {"size", "weight", "is_premium"} <= input_names
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        # VectorAssembler: size(1) + weight(1) + is_premium(1) = 3
        concat_out = next(o for nd in m.nodes if nd.op == "Concat" for o in nd.outputs)
        assert concat_out.name == "features"
        assert concat_out.type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_mixed_string_indexer_assembler_lr_pipeline(self, spark_mixed_df):
        """StringIndexer (string→float index) + VectorAssembler + LR on mixed input.
        Spark's StringIndexer maps to LabelEncoder in the OMLE IR."""
        pipeline = Pipeline(stages=[
            F.StringIndexer(inputCol="color", outputCol="color_idx"),
            F.VectorAssembler(inputCols=["color_idx", "size", "weight"], outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=10),
        ])
        model = pipeline.fit(spark_mixed_df)
        m = from_spark_live(model, dataset=spark_mixed_df)
        ops = [n.op for n in m.nodes]
        assert "LabelEncoder" in ops  # StringIndexer → LabelEncoder in OMLE IR
        assert "Concat" in ops
        assert "Linear" in ops
        # ALL raw pipeline inputs — color (STRING scalar), size and weight (FLOAT64 scalars)
        input_names = {i.name for i in m.inputs}
        assert input_names == {"color", "size", "weight"}
        # color is the StringIndexer input: STRING, scalar
        color_spec = next(i for i in m.inputs if i.name == "color")
        assert color_spec.type.dtype == omle.DataType.STRING
        assert color_spec.type.shape == [-1]
        # size and weight are numeric scalars fed into VectorAssembler
        for col in ("size", "weight"):
            s = next(i for i in m.inputs if i.name == col)
            assert s.type.dtype == omle.DataType.FLOAT64
            assert s.type.shape == [-1]
        # StringIndexer output: float index scalar
        le_out = next(o for nd in m.nodes if nd.op == "LabelEncoder" for o in nd.outputs)
        assert le_out.name == "color_idx"
        assert le_out.type.dtype == omle.DataType.FLOAT64
        assert le_out.type.shape == [-1]
        # VectorAssembler: color_idx(1) + size(1) + weight(1) = 3
        concat_out = next(o for nd in m.nodes if nd.op == "Concat" for o in nd.outputs)
        assert concat_out.name == "features"
        assert concat_out.type.dtype == omle.DataType.FLOAT64
        assert concat_out.type.shape == [-1, 3]
        # Schema features: color is STRING/NOMINAL with categories; others are FLOAT64/CONTINUOUS
        feat_map = {f.name: f for f in m.model_schema.features}
        assert feat_map["color"].type.dtype == omle.DataType.STRING
        assert feat_map["color"].measure_level == omle.MeasureLevel.NOMINAL
        color_cats = {v.value.string_value for v in feat_map["color"].domain.discrete.values}
        assert color_cats == {"red", "green", "blue"}  # categories from StringIndexer labels
        assert feat_map["size"].type.dtype   == omle.DataType.FLOAT64
        assert feat_map["weight"].type.dtype == omle.DataType.FLOAT64
        # LR outputs
        assert "prediction" in {o.name for o in m.outputs}
        assert "probability" in {o.name for o in m.outputs}
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_mixed_ohe_assembler_lr_pipeline(self, spark_mixed_df):
        """Full categorical encoding: StringIndexer→OHE→VectorAssembler→LR."""
        pipeline = Pipeline(stages=[
            F.StringIndexer(inputCol="color", outputCol="color_idx"),
            F.OneHotEncoder(inputCols=["color_idx"], outputCols=["color_vec"]),
            F.VectorAssembler(inputCols=["color_vec", "size", "weight", "is_premium"],
                              outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=10),
        ])
        model = pipeline.fit(spark_mixed_df)
        m = from_spark_live(model, dataset=spark_mixed_df)
        ops = [n.op for n in m.nodes]
        assert "LabelEncoder" in ops  # StringIndexer → LabelEncoder in OMLE IR
        assert "OneHotEncoder" in ops
        assert "Linear" in ops
        assert m.inputs[0].name == "color"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        # StringIndexer output: float index scalar
        le_out = next(o for nd in m.nodes if nd.op == "LabelEncoder" for o in nd.outputs)
        assert le_out.name == "color_idx"
        assert le_out.type.dtype == omle.DataType.FLOAT64
        assert le_out.type.shape == [-1]
        # OHE output: color (3 cats, dropLast=True) → 2 dims
        ohe_out = next(o for nd in m.nodes if nd.op == "OneHotEncoder" for o in nd.outputs)
        assert ohe_out.name == "color_vec"
        assert ohe_out.type.shape == [-1, 2]
        # VectorAssembler: color_vec(2) + size(1) + weight(1) + is_premium(1) = 5
        concat_out = next(o for nd in m.nodes if nd.op == "Concat" for o in nd.outputs)
        assert concat_out.name == "features"
        assert concat_out.type.shape == [-1, 5]
        assert "prediction" in {o.name for o in m.outputs}
        assert "probability" in {o.name for o in m.outputs}
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_multi_col_ohe_pipeline(self, spark):
        """Multi-column OHE inside a pipeline: single variadic OHE node with named ys outputs."""
        data = spark.createDataFrame([
            ("red",   "small",  1.0, 0.0),
            ("blue",  "large",  2.0, 1.0),
            ("red",   "medium", 3.0, 0.0),
            ("green", "small",  4.0, 1.0),
            ("blue",  "medium", 5.0, 0.0),
            ("green", "large",  6.0, 1.0),
        ], ["color", "size_cat", "weight", "label"])
        pipeline = Pipeline(stages=[
            F.StringIndexer(inputCols=["color", "size_cat"], outputCols=["color_idx", "size_idx"]),
            F.OneHotEncoder(inputCols=["color_idx", "size_idx"], outputCols=["color_vec", "size_vec"]),
            F.VectorAssembler(inputCols=["color_vec", "size_vec", "weight"],
                              outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=10),
        ])
        model = pipeline.fit(data)
        m = from_spark_live(model, dataset=data)
        ops = [n.op for n in m.nodes]
        # Multi-col StringIndexer → single LabelEncoder node with variadic inputs
        assert ops.count("LabelEncoder") == 1
        enc_node = next(n for n in m.nodes if n.op == "LabelEncoder")
        enc_inputs = {inp.name for inp in enc_node.inputs}
        assert "color" in enc_inputs
        assert "size_cat" in enc_inputs
        # LabelEncoder outputs: color_idx and size_idx are float scalars
        le_out_map = {o.name: o for o in enc_node.outputs}
        assert "color_idx" in le_out_map
        assert "size_idx" in le_out_map
        for le_name in ("color_idx", "size_idx"):
            assert le_out_map[le_name].type.dtype == omle.DataType.FLOAT64
            assert le_out_map[le_name].type.shape == [-1]
        # Single variadic OHE node with two named outputs
        assert ops.count("OneHotEncoder") == 1
        ohe_node = next(n for n in m.nodes if n.op == "OneHotEncoder")
        assert {inp.name for inp in ohe_node.inputs} == {"color_idx", "size_idx"}
        assert {o.name for o in ohe_node.outputs} == {"color_vec", "size_vec"}
        # OHE outputs: color(3 cats, dropLast=True)→2 dims, size_cat(3 cats)→2 dims
        ohe_out_map = {o.name: o for o in ohe_node.outputs}
        assert ohe_out_map["color_vec"].type.shape == [-1, 2]
        assert ohe_out_map["size_vec"].type.shape == [-1, 2]
        # VectorAssembler: color_vec(2) + size_vec(2) + weight(1) = 5
        concat_out = next(o for nd in m.nodes if nd.op == "Concat" for o in nd.outputs)
        assert concat_out.name == "features"
        assert concat_out.type.dtype == omle.DataType.FLOAT64
        assert concat_out.type.shape == [-1, 5]
        assert "Linear" in ops
        in_map = {i.name: i for i in m.inputs}
        assert "color" in in_map
        assert "size_cat" in in_map
        assert "weight" in in_map
        for name in ("color", "size_cat"):
            assert in_map[name].type.dtype == omle.DataType.STRING
            assert in_map[name].type.shape == [-1]
        assert in_map["weight"].type.dtype == omle.DataType.FLOAT64
        assert in_map["weight"].type.shape == [-1]
        out_map = {o.name: o for o in m.outputs}
        assert "prediction" in out_map
        assert "probability" in out_map
        assert out_map["prediction"].type.dtype == omle.DataType.INT64
        assert out_map["prediction"].type.shape == [-1]
        assert out_map["probability"].type.dtype == omle.DataType.FLOAT64
        assert out_map["probability"].type.shape == [-1, 2]
        assert m.verification is not None

    def test_sql_transformer_pipeline(self, spark):
        """SQLTransformer computing a derived column, then VectorAssembler + LR."""
        pytest.importorskip("sqlglot", reason="sqlglot not installed")
        data = spark.createDataFrame([
            (1.0, 2.0, 0.0), (3.0, 4.0, 1.0),
            (5.0, 6.0, 0.0), (7.0, 8.0, 1.0),
        ], ["a", "b", "label"])
        pipeline = Pipeline(stages=[
            F.SQLTransformer(statement="SELECT *, a + b AS ab FROM __THIS__"),
            F.VectorAssembler(inputCols=["a", "b", "ab"], outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=5),
        ])
        model = pipeline.fit(data)
        m = from_spark_live(model, dataset=data, feature_names=["a", "b"])
        ops = [n.op for n in m.nodes]
        assert "Derive" in ops
        assert "Concat" in ops
        assert "Linear" in ops
        in_map = {i.name: i for i in m.inputs}
        assert "a" in in_map
        assert "b" in in_map
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        # Derive node creates intermediate column named after the SQL alias
        derive_node = next(n for n in m.nodes if n.op == "Derive")
        assert derive_node.outputs[0].name == "ab"
        # VectorAssembler: a(1) + b(1) + ab(1) = 3
        concat_out = next(o for nd in m.nodes if nd.op == "Concat" for o in nd.outputs)
        assert concat_out.name == "features"
        assert concat_out.type.dtype == omle.DataType.FLOAT64
        assert concat_out.type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]


# ── SQLTransformer ──────────────────────────────────────────────────────────────

class TestSQLTransformer:
    """Verify SQLTransformer → Derive node conversion for every supported expression type.

    Each test creates a minimal Pipeline(SQLTransformer), converts it, and checks:
      • the correct Derive nodes are emitted
      • output tensor names exactly match the SQL aliases
      • the expression attribute is present on every Derive node
    """

    @pytest.fixture(autouse=True)
    def require_sqlglot(self):
        pytest.importorskip("sqlglot", reason="sqlglot not installed")

    @pytest.fixture
    def num_df(self, spark):
        return spark.createDataFrame(
            [(1.0, 2.0), (3.0, -4.0), (5.0, 6.0)], ["a", "b"]
        )

    @pytest.fixture
    def str_df(self, spark):
        return spark.createDataFrame(
            [("hello ", "world"), ("FOO", "bar")], ["s", "t"]
        )

    def _run(self, df, stmt, feature_names=None):
        p = Pipeline(stages=[F.SQLTransformer(statement=stmt)])
        return from_spark_live(p.fit(df), dataset=df, feature_names=feature_names)

    def _derive_map(self, m):
        return {
            o.name: nd
            for nd in m.nodes if nd.op == "Derive"
            for o in nd.outputs
        }

    # ── Arithmetic ───────────────────────────────────────────────────────────────

    def test_arithmetic_ops(self, num_df):
        m = self._run(num_df, "SELECT *, a + b AS add, a - b AS sub, a * b AS mul, a / b AS div FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"add", "sub", "mul", "div"}
        for nd in dm.values():
            assert any(attr.name == "expr" for attr in nd.attributes)
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        in_map = {i.name: i for i in m.inputs}
        assert in_map["a"].type.dtype == omle.DataType.FLOAT64
        assert in_map["a"].type.shape == [-1]
        assert in_map["b"].type.dtype == omle.DataType.FLOAT64
        assert in_map["b"].type.shape == [-1]
        out_map = {o.name: o for o in m.outputs}
        assert set(out_map) == {"add", "sub", "mul", "div"}
        for o in m.outputs:
            assert o.role == omle.OutputRole.TRANSFORMED_VALUE
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_unary_minus(self, num_df):
        m = self._run(num_df, "SELECT *, -a AS neg_a FROM __THIS__")
        assert "neg_a" in self._derive_map(m)
        nd = self._derive_map(m)["neg_a"]
        assert nd.inputs[0].name == "a"
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        neg_out = next(o for o in m.outputs if o.name == "neg_a")
        assert neg_out.type.dtype == omle.DataType.FLOAT64
        assert neg_out.type.shape == [-1]
        assert m.verification is not None

    def test_nested_arithmetic(self, num_df):
        m = self._run(num_df, "SELECT *, (a + b) * (a - b) AS diff_sq FROM __THIS__")
        assert "diff_sq" in self._derive_map(m)
        nd = self._derive_map(m)["diff_sq"]
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        diff_out = next(o for o in m.outputs if o.name == "diff_sq")
        assert diff_out.type.dtype == omle.DataType.FLOAT64
        assert diff_out.type.shape == [-1]
        assert m.verification is not None

    # ── Math functions ───────────────────────────────────────────────────────────

    def test_abs_sqrt_floor_ceil(self, num_df):
        m = self._run(num_df, "SELECT *, ABS(b) AS abs_b, SQRT(a) AS sqrt_a, FLOOR(b) AS floor_b, CEIL(b) AS ceil_b FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"abs_b", "sqrt_a", "floor_b", "ceil_b"}
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_exp_log(self, num_df):
        m = self._run(num_df, "SELECT *, EXP(a) AS exp_a, LN(a) AS ln_a, LOG2(a) AS lg2, LOG10(a) AS lg10 FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"exp_a", "ln_a", "lg2", "lg10"}
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_round_sign_pow(self, num_df):
        m = self._run(num_df, "SELECT *, ROUND(b, 0) AS rnd, SIGN(b) AS sgn, POW(a, 2.0) AS sq FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"rnd", "sgn", "sq"}
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    # ── Comparison & boolean ─────────────────────────────────────────────────────

    def test_comparison_ops(self, num_df):
        m = self._run(num_df, "SELECT *, a > b AS gt, a < b AS lt, a = b AS eq, a != b AS ne FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"gt", "lt", "eq", "ne"}
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        out_map = {o.name: o for o in m.outputs}
        assert set(out_map) == {"gt", "lt", "eq", "ne"}
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_boolean_ops(self, num_df):
        m = self._run(num_df, "SELECT *, a > 0 AND b > 0 AS both_pos, a > 0 OR b > 0 AS either_pos FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"both_pos", "either_pos"}
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_not_operator(self, num_df):
        m = self._run(num_df, "SELECT *, NOT (a > b) AS not_gt FROM __THIS__")
        assert "not_gt" in self._derive_map(m)
        nd = self._derive_map(m)["not_gt"]
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        not_out = next(o for o in m.outputs if o.name == "not_gt")
        assert not_out.type.dtype == omle.DataType.FLOAT64
        assert not_out.type.shape == [-1]
        assert m.verification is not None

    # ── Conditionals ─────────────────────────────────────────────────────────────

    def test_if_conditional(self, num_df):
        m = self._run(num_df, "SELECT *, IF(a > 2.0, 1.0, 0.0) AS flag FROM __THIS__")
        assert "flag" in self._derive_map(m)
        nd = self._derive_map(m)["flag"]
        assert nd.inputs[0].name == "a"
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        flag_out = next(o for o in m.outputs if o.name == "flag")
        assert flag_out.type.dtype == omle.DataType.FLOAT64
        assert flag_out.type.shape == [-1]
        assert m.verification is not None

    def test_case_when_else(self, num_df):
        m = self._run(num_df, "SELECT *, CASE WHEN a > 4.0 THEN 2.0 WHEN a > 2.0 THEN 1.0 ELSE 0.0 END AS tier FROM __THIS__")
        assert "tier" in self._derive_map(m)
        nd = self._derive_map(m)["tier"]
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        tier_out = next(o for o in m.outputs if o.name == "tier")
        assert tier_out.type.dtype == omle.DataType.FLOAT64
        assert tier_out.type.shape == [-1]
        assert m.verification is not None

    def test_coalesce(self, spark):
        df = spark.createDataFrame([(1.0, None), (None, 2.0)], ["a", "b"])
        m = self._run(df, "SELECT *, COALESCE(a, b) AS coal FROM __THIS__")
        assert "coal" in self._derive_map(m)
        nd = self._derive_map(m)["coal"]
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        coal_out = next(o for o in m.outputs if o.name == "coal")
        assert coal_out.type.dtype == omle.DataType.FLOAT64
        assert coal_out.type.shape == [-1]
        assert m.verification is not None

    # ── Null checks ──────────────────────────────────────────────────────────────

    def test_null_checks(self, spark):
        df = spark.createDataFrame([(1.0,), (None,)], ["a"])
        m = self._run(df, "SELECT *, a IS NULL AS is_null, a IS NOT NULL AS not_null FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"is_null", "not_null"}
        for nd in dm.values():
            assert nd.inputs[0].name == "a"
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        out_map = {o.name: o for o in m.outputs}
        assert set(out_map) == {"is_null", "not_null"}
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    # ── String functions ─────────────────────────────────────────────────────────

    def test_lower_upper(self, str_df):
        m = self._run(str_df, "SELECT *, LOWER(s) AS low_s, UPPER(t) AS up_t FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"low_s", "up_t"}
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.STRING
            assert nd.outputs[0].type.shape == [-1]
        in_map = {i.name: i for i in m.inputs}
        assert in_map["s"].type.dtype == omle.DataType.STRING
        assert in_map["t"].type.dtype == omle.DataType.STRING
        out_map = {o.name: o for o in m.outputs}
        assert set(out_map) == {"low_s", "up_t"}
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.STRING
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_concat_trim(self, str_df):
        m = self._run(str_df, "SELECT *, CONCAT(s, t) AS cat_st, TRIM(s) AS trimmed FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"cat_st", "trimmed"}
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.STRING
            assert nd.outputs[0].type.shape == [-1]
        in_map = {i.name: i for i in m.inputs}
        assert in_map["s"].type.dtype == omle.DataType.STRING
        assert in_map["t"].type.dtype == omle.DataType.STRING
        out_map = {o.name: o for o in m.outputs}
        assert set(out_map) == {"cat_st", "trimmed"}
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.STRING
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_substring(self, str_df):
        m = self._run(str_df, "SELECT *, SUBSTRING(s, 1, 3) AS sub3 FROM __THIS__")
        assert "sub3" in self._derive_map(m)
        nd = self._derive_map(m)["sub3"]
        assert nd.inputs[0].name == "s"
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.STRING
        assert nd.outputs[0].type.shape == [-1]
        in_map = {i.name: i for i in m.inputs}
        assert in_map["s"].type.dtype == omle.DataType.STRING
        sub_out = next(o for o in m.outputs if o.name == "sub3")
        assert sub_out.type.dtype == omle.DataType.STRING
        assert sub_out.type.shape == [-1]
        assert m.verification is not None

    # ── Multi-column and passthrough ─────────────────────────────────────────────

    def test_multi_derived_cols(self, num_df):
        """Multiple derived columns → one Derive node per alias."""
        m = self._run(num_df, "SELECT *, a + b AS ab, a * 2.0 AS double_a, b * b AS b_sq FROM __THIS__")
        dm = self._derive_map(m)
        assert set(dm) == {"ab", "double_a", "b_sq"}
        assert [n.op for n in m.nodes].count("Derive") == 3
        for nd in dm.values():
            assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
            assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
            assert nd.outputs[0].type.shape == [-1]
        out_map = {o.name: o for o in m.outputs}
        assert set(out_map) == {"ab", "double_a", "b_sq"}
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        assert m.verification is not None

    def test_passthrough_columns_unchanged(self, num_df):
        """SELECT * passes through all input columns; only aliased expressions become Derive nodes."""
        m = self._run(num_df, "SELECT *, a + b AS ab FROM __THIS__", feature_names=["a", "b"])
        assert {i.name for i in m.inputs} == {"a", "b"}
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        assert set(self._derive_map(m)) == {"ab"}
        nd = self._derive_map(m)["ab"]
        assert nd.outputs[0].role == omle.OutputRole.TRANSFORMED_VALUE
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        ab_out = next(o for o in m.outputs if o.name == "ab")
        assert ab_out.type.dtype == omle.DataType.FLOAT64
        assert ab_out.type.shape == [-1]
        assert m.verification is not None

    # ── Pipeline integration ──────────────────────────────────────────────────────

    def test_pipeline_with_assembler_and_lr(self, spark):
        """SQLTransformer → VectorAssembler → LR: full end-to-end IR check."""
        data = spark.createDataFrame(
            [(1.0, 2.0, 0.0), (3.0, 4.0, 1.0), (5.0, 6.0, 0.0), (7.0, 8.0, 1.0)],
            ["a", "b", "label"],
        )
        pipeline = Pipeline(stages=[
            F.SQLTransformer(statement="SELECT *, a + b AS ab FROM __THIS__"),
            F.VectorAssembler(inputCols=["a", "b", "ab"], outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=5),
        ])
        m = from_spark_live(pipeline.fit(data), dataset=data, feature_names=["a", "b"])
        ops = [n.op for n in m.nodes]
        assert "Derive" in ops
        assert "Concat" in ops
        assert "Linear" in ops
        derive_node = next(n for n in m.nodes if n.op == "Derive")
        assert derive_node.outputs[0].name == "ab"
        assert derive_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert derive_node.outputs[0].type.shape == [-1]
        concat_out = next(o for nd in m.nodes if nd.op == "Concat" for o in nd.outputs)
        assert concat_out.name == "features"
        assert concat_out.type.dtype == omle.DataType.FLOAT64
        assert concat_out.type.shape == [-1, 3]
        out_map = {o.name: o for o in m.outputs}
        assert out_map["prediction"].type.dtype == omle.DataType.INT64
        assert out_map["prediction"].type.shape == [-1]
        assert out_map["probability"].type.dtype == omle.DataType.FLOAT64
        assert out_map["probability"].type.shape == [-1, 2]


# ── Saved-pipeline conversion via PySpark load ──────────────────────────────────

class TestSavedPipeline:
    def test_from_saved_pipeline(self, binary_df, tmp_path):
        """PipelineModel saved to disk → loaded via PySpark → converted."""
        pipeline = Pipeline(stages=[
            F.StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True),
            C.LogisticRegression(featuresCol="features_scaled", maxIter=5),
        ])
        fitted = pipeline.fit(binary_df)
        save_path = str(tmp_path / "pipeline")
        fitted.save(save_path)

        loaded = PipelineModel.load(save_path)
        m = from_spark_live(loaded, dataset=binary_df)
        ops = [n.op for n in m.nodes]
        assert "StandardScaler" in ops
        assert "Linear" in ops
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert "prediction" in {o.name for o in m.outputs}
        assert "probability" in {o.name for o in m.outputs}
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_saved_pipeline_writes_json(self, binary_df, tmp_path):
        """Saved pipeline loaded via PySpark → converted → JSON round-trip."""
        fitted = Pipeline(stages=[
            C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42),
        ]).fit(binary_df)
        save_path   = str(tmp_path / "rf_pipeline")
        output_json = str(tmp_path / "rf_pipeline.json")
        fitted.save(save_path)

        m = from_spark_live(PipelineModel.load(save_path), dataset=binary_df)
        assert m.inputs[0].name == "features"
        assert "prediction" in {o.name for o in m.outputs}
        omle.save_json(m, output_json)

        loaded = omle.load_json(output_json)
        assert any(n.op == "TreeEnsemble" for n in loaded.nodes)
        assert loaded.inputs[0].name == "features"
        assert "prediction" in {o.name for o in loaded.outputs}
        pred_out = next(o for o in loaded.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in loaded.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_from_saved_model(self, binary_df, tmp_path):
        """Single model saved to disk → loaded via PySpark → converted."""
        from pyspark.ml.classification import RandomForestClassificationModel
        model = C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42).fit(binary_df)
        save_path = str(tmp_path / "rf_model")
        model.save(save_path)

        loaded = RandomForestClassificationModel.load(save_path)
        m = from_spark_live(loaded, dataset=binary_df)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert "prediction" in {o.name for o in m.outputs}
        assert "probability" in {o.name for o in m.outputs}
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_saved_model_writes_json(self, regression_df, tmp_path):
        """Single model saved to disk → loaded → converted → JSON round-trip."""
        from pyspark.ml.regression import LinearRegressionModel
        model = R.LinearRegression(maxIter=5).fit(regression_df)
        save_path   = str(tmp_path / "lr_model")
        output_json = str(tmp_path / "lr.json")
        model.save(save_path)

        m = from_spark_live(LinearRegressionModel.load(save_path), dataset=regression_df)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert {o.name for o in m.outputs} == {"prediction"}
        omle.save_json(m, output_json)

        loaded = omle.load_json(output_json)
        assert any(n.op == "Linear" for n in loaded.nodes)
        assert loaded.inputs[0].name == "features"
        assert {o.name for o in loaded.outputs} == {"prediction"}
        lr_pred_out = next(o for o in loaded.outputs if o.name == "prediction")
        assert lr_pred_out.type.dtype == omle.DataType.FLOAT64
        assert lr_pred_out.type.shape == [-1]

    def test_saved_pipeline_feature_cols_preserved(self, spark, tmp_path):
        """feature_names passed to from_spark_live appear as model inputs."""
        data = spark.createDataFrame([
            (1.0, 2.0, 0.0), (3.0, 4.0, 1.0),
            (5.0, 6.0, 0.0), (7.0, 8.0, 1.0),
        ], ["a", "b", "label"])
        fitted = Pipeline(stages=[
            F.VectorAssembler(inputCols=["a", "b"], outputCol="features"),
            C.LogisticRegression(maxIter=5, featuresCol="features"),
        ]).fit(data)
        save_path = str(tmp_path / "va_lr_pipeline")
        fitted.save(save_path)

        m = from_spark_live(PipelineModel.load(save_path), dataset=data, feature_names=["a", "b"])
        input_names = {inp.name for inp in m.inputs}
        assert "a" in input_names
        assert "b" in input_names
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]


# ── XGBoost / LightGBM Spark (optional — requires extra libraries / JARs) ────────

class TestXGBoostLightGBMSpark:
    def test_xgboost_spark_classifier(self, binary_df):
        """Python xgboost.spark SparkXGBClassifier → SparkXGBClassifierModel."""
        xgb_spark = pytest.importorskip("xgboost.spark", reason="xgboost.spark not available")
        model = xgb_spark.SparkXGBClassifier(
            num_workers=1, n_estimators=5, max_depth=3, use_gpu=False
        ).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        assert "probability" in output_names
        xgb_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert xgb_pred_out.type.dtype == omle.DataType.INT64
        assert xgb_pred_out.type.shape == [-1]
        xgb_prob_out = next(o for o in m.outputs if o.name == "probability")
        assert xgb_prob_out.type.dtype == omle.DataType.FLOAT64
        assert xgb_prob_out.type.shape == [-1, 2]

    def test_xgboost_spark_regressor(self, regression_df):
        """Python xgboost.spark SparkXGBRegressor → SparkXGBRegressorModel."""
        xgb_spark = pytest.importorskip("xgboost.spark", reason="xgboost.spark not available")
        model = xgb_spark.SparkXGBRegressor(
            num_workers=1, n_estimators=5, max_depth=3, use_gpu=False
        ).fit(regression_df)
        m = from_spark_live(model, dataset=regression_df)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"prediction"}
        xgbr_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert xgbr_pred_out.type.dtype == omle.DataType.FLOAT64
        assert xgbr_pred_out.type.shape == [-1]

    @skip_lightgbm_synapseml
    @skip_no_synapseml
    def test_lightgbm_spark_classifier(self, binary_df):
        """SynapseML LightGBMClassificationModel.

        Requires: pip install synapseml  (JAR is auto-detected from the package)
        """
        lgbm = pytest.importorskip("synapse.ml.lightgbm", reason="synapseml not installed")
        model = lgbm.LightGBMClassifier(
            numIterations=5, maxDepth=3, labelCol="label", featuresCol="features"
        ).fit(binary_df)
        m = from_spark_live(model, dataset=binary_df)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        output_names = {o.name for o in m.outputs}
        assert "prediction" in output_names
        lgbm_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert lgbm_pred_out.type.dtype == omle.DataType.INT64
        assert lgbm_pred_out.type.shape == [-1]

    @skip_lightgbm_synapseml
    @skip_no_synapseml
    def test_lightgbm_spark_regressor(self, regression_df):
        """SynapseML LightGBMRegressionModel.

        Requires: pip install synapseml  (JAR is auto-detected from the package)
        """
        lgbm = pytest.importorskip("synapse.ml.lightgbm", reason="synapseml not installed")
        model = lgbm.LightGBMRegressor(
            numIterations=5, maxDepth=3, labelCol="label", featuresCol="features"
        ).fit(regression_df)
        m = from_spark_live(model, dataset=regression_df)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"prediction"}
        lgbmr_pred_out = next(o for o in m.outputs if o.name == "prediction")
        assert lgbmr_pred_out.type.dtype == omle.DataType.FLOAT64
        assert lgbmr_pred_out.type.shape == [-1]


# ── Model roundtrip (JSON → load back) ──────────────────────────────────────────

class TestJsonRoundtrip:
    def test_lr_json_roundtrip(self, binary_df, tmp_path):
        model  = C.LogisticRegression(maxIter=5).fit(binary_df)
        m      = from_spark_live(model, dataset=binary_df)
        path   = str(tmp_path / "lr.json")
        omle.save_json(m, path)
        loaded = omle.load_json(path)
        assert any(n.op == "Linear" for n in loaded.nodes)
        assert loaded.inputs[0].name == "features"
        assert loaded.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert loaded.inputs[0].type.shape == [-1, 3]
        assert "prediction" in {o.name for o in loaded.outputs}
        assert "probability" in {o.name for o in loaded.outputs}
        lr_pred_out = next(o for o in loaded.outputs if o.name == "prediction")
        assert lr_pred_out.type.dtype == omle.DataType.INT64
        assert lr_pred_out.type.shape == [-1]
        lr_prob_out = next(o for o in loaded.outputs if o.name == "probability")
        assert lr_prob_out.type.dtype == omle.DataType.FLOAT64
        assert lr_prob_out.type.shape == [-1, 2]

    def test_rf_json_roundtrip(self, binary_df, tmp_path):
        model  = C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42).fit(binary_df)
        m      = from_spark_live(model, dataset=binary_df)
        path   = str(tmp_path / "rf.json")
        omle.save_json(m, path)
        loaded = omle.load_json(path)
        node   = next(n for n in loaded.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble is not None
        assert len(node.tree_ensemble.trees) == 3
        for tree in node.tree_ensemble.trees:
            assert tree.num_nodes > 0
        assert loaded.inputs[0].name == "features"
        assert loaded.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert loaded.inputs[0].type.shape == [-1, 3]
        assert "prediction" in {o.name for o in loaded.outputs}
        assert "probability" in {o.name for o in loaded.outputs}
        rf_pred_out = next(o for o in loaded.outputs if o.name == "prediction")
        assert rf_pred_out.type.dtype == omle.DataType.INT64
        assert rf_pred_out.type.shape == [-1]
        rf_prob_out = next(o for o in loaded.outputs if o.name == "probability")
        assert rf_prob_out.type.dtype == omle.DataType.FLOAT64
        assert rf_prob_out.type.shape == [-1, 2]


# ── Prediction match: Spark native vs omle-spark runtime ──────────────────
#
# Requires the omle-spark JAR (run `sbt package` in omle-runtime/spark).
# Automatically skipped when the JAR is not present.

@skip_no_omle_spark
class TestPredictionMatch:

    # ── Regression ───────────────────────────────────────────────────────────────

    def test_linear_regression(self, regression_df):
        spark_model = R.LinearRegression(maxIter=50).fit(regression_df)
        spark_out   = spark_model.transform(regression_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=regression_df)).transform(regression_df)
        np.testing.assert_allclose(
            _omle_pred(omle_out), _spark_pred(spark_out), atol=1e-4,
        )

    def test_decision_tree_regressor(self, regression_df):
        spark_model = R.DecisionTreeRegressor(maxDepth=3).fit(regression_df)
        spark_out   = spark_model.transform(regression_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=regression_df)).transform(regression_df)
        np.testing.assert_allclose(
            _omle_pred(omle_out), _spark_pred(spark_out), atol=1e-4,
        )

    def test_random_forest_regressor(self, regression_df):
        spark_model = R.RandomForestRegressor(numTrees=5, maxDepth=2, seed=42).fit(regression_df)
        spark_out   = spark_model.transform(regression_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=regression_df)).transform(regression_df)
        np.testing.assert_allclose(
            _omle_pred(omle_out), _spark_pred(spark_out), atol=1e-4,
        )

    def test_gbt_regressor(self, regression_df):
        spark_model = R.GBTRegressor(maxIter=5, maxDepth=2, seed=42).fit(regression_df)
        spark_out   = spark_model.transform(regression_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=regression_df)).transform(regression_df)
        np.testing.assert_allclose(
            _omle_pred(omle_out), _spark_pred(spark_out), atol=1e-4,
        )

    # ── Binary classification ─────────────────────────────────────────────────────

    def test_logistic_regression_binary_class(self, binary_df):
        spark_model = C.LogisticRegression(maxIter=20).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_logistic_regression_binary_probability(self, binary_df):
        spark_model = C.LogisticRegression(maxIter=20).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        spark_p1    = _spark_prob(spark_out)[:, 1]
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_allclose(_omle_prob(omle_out)[:, 1], spark_p1, atol=1e-4)

    def test_decision_tree_binary_class(self, binary_df):
        spark_model = C.DecisionTreeClassifier(maxDepth=3).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_decision_tree_binary_probability(self, binary_df):
        spark_model = C.DecisionTreeClassifier(maxDepth=3).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        spark_p1    = _spark_prob(spark_out)[:, 1]
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_allclose(_omle_prob(omle_out)[:, 1], spark_p1, atol=1e-4)

    def test_random_forest_binary_class(self, binary_df):
        spark_model = C.RandomForestClassifier(numTrees=5, maxDepth=2, seed=42).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_random_forest_binary_probability(self, binary_df):
        spark_model = C.RandomForestClassifier(numTrees=5, maxDepth=2, seed=42).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        spark_p1    = _spark_prob(spark_out)[:, 1]
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_allclose(_omle_prob(omle_out)[:, 1], spark_p1, atol=1e-4)

    def test_gbt_binary_class(self, binary_df):
        spark_model = C.GBTClassifier(maxIter=5, maxDepth=2, seed=42).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_gbt_binary_probability(self, binary_df):
        spark_model = C.GBTClassifier(maxIter=5, maxDepth=2, seed=42).fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        spark_p1    = _spark_prob(spark_out)[:, 1]
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_allclose(_omle_prob(omle_out)[:, 1], spark_p1, atol=1e-4)

    # ── Multiclass classification ─────────────────────────────────────────────────

    def test_logistic_regression_multiclass_class(self, multiclass_df):
        spark_model = C.LogisticRegression(maxIter=50).fit(multiclass_df)
        spark_out   = spark_model.transform(multiclass_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=multiclass_df)).transform(multiclass_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_logistic_regression_multiclass_probability(self, multiclass_df):
        spark_model = C.LogisticRegression(maxIter=50).fit(multiclass_df)
        spark_out   = spark_model.transform(multiclass_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=multiclass_df)).transform(multiclass_df)
        np.testing.assert_allclose(_omle_prob(omle_out), _spark_prob(spark_out), atol=1e-4)

    def test_decision_tree_multiclass_class(self, multiclass_df):
        spark_model = C.DecisionTreeClassifier(maxDepth=3).fit(multiclass_df)
        spark_out   = spark_model.transform(multiclass_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=multiclass_df)).transform(multiclass_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_random_forest_multiclass_class(self, multiclass_df):
        spark_model = C.RandomForestClassifier(numTrees=5, maxDepth=3, seed=42).fit(multiclass_df)
        spark_out   = spark_model.transform(multiclass_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=multiclass_df)).transform(multiclass_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    # ── Pipeline round-trip ───────────────────────────────────────────────────────

    def test_scaler_plus_lr_pipeline(self, binary_df):
        pipeline = Pipeline(stages=[
            F.StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True),
            C.LogisticRegression(featuresCol="features_scaled", maxIter=20),
        ])
        spark_model = pipeline.fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_minmax_scaler_plus_rf_pipeline(self, binary_df):
        pipeline = Pipeline(stages=[
            F.MinMaxScaler(inputCol="features", outputCol="features_scaled"),
            C.RandomForestClassifier(featuresCol="features_scaled", numTrees=5, maxDepth=2, seed=42),
        ])
        spark_model = pipeline.fit(binary_df)
        spark_out   = spark_model.transform(binary_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=binary_df)).transform(binary_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))

    def test_scaler_plus_lr_regression_pipeline(self, regression_df):
        pipeline = Pipeline(stages=[
            F.StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True),
            R.LinearRegression(featuresCol="features_scaled", maxIter=50),
        ])
        spark_model = pipeline.fit(regression_df)
        spark_out   = spark_model.transform(regression_df)
        omle_out    = _to_omle(from_spark_live(spark_model, dataset=regression_df)).transform(regression_df)
        np.testing.assert_allclose(_omle_pred(omle_out), _spark_pred(spark_out), atol=1e-4)

    # ── Mixed-type pipelines ──────────────────────────────────────────────────────

    def test_mixed_numeric_assembler_rf_class(self, spark_mixed_df):
        """VectorAssembler + RF pipeline on mixed DataFrame: class labels and probabilities match."""
        pipeline = Pipeline(stages=[
            F.VectorAssembler(inputCols=["size", "weight", "is_premium"], outputCol="features"),
            C.RandomForestClassifier(featuresCol="features", numTrees=10, maxDepth=3, seed=42),
        ])
        fitted    = pipeline.fit(spark_mixed_df)
        spark_out = fitted.transform(spark_mixed_df)
        omle_out  = _to_omle(from_spark_live(fitted, dataset=spark_mixed_df,
                                             feature_names=["size", "weight", "is_premium"])).transform(spark_mixed_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))
        np.testing.assert_allclose(_omle_prob(omle_out), _spark_prob(spark_out), atol=1e-4)

    def test_mixed_ohe_assembler_lr_binary(self, spark_mixed_df):
        """StringIndexer → OHE → VectorAssembler → LR pipeline: class labels and probabilities match."""
        pipeline = Pipeline(stages=[
            F.StringIndexer(inputCol="color", outputCol="color_idx"),
            F.OneHotEncoder(inputCols=["color_idx"], outputCols=["color_vec"]),
            F.VectorAssembler(inputCols=["color_vec", "size", "weight", "is_premium"],
                              outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=20),
        ])
        fitted    = pipeline.fit(spark_mixed_df)
        spark_out = fitted.transform(spark_mixed_df)
        omle_out  = _to_omle(from_spark_live(fitted, dataset=spark_mixed_df)).transform(spark_mixed_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))
        np.testing.assert_allclose(_omle_prob(omle_out), _spark_prob(spark_out), atol=1e-4)

    def test_mixed_ohe_assembler_gbt_binary(self, spark_mixed_df):
        """StringIndexer → OHE → VectorAssembler → GBT pipeline: class labels and probabilities match."""
        pipeline = Pipeline(stages=[
            F.StringIndexer(inputCol="color", outputCol="color_idx"),
            F.OneHotEncoder(inputCols=["color_idx"], outputCols=["color_vec"]),
            F.VectorAssembler(inputCols=["color_vec", "size", "weight"], outputCol="features"),
            C.GBTClassifier(featuresCol="features", maxIter=5, maxDepth=2, seed=42),
        ])
        fitted    = pipeline.fit(spark_mixed_df)
        spark_out = fitted.transform(spark_mixed_df)
        omle_out  = _to_omle(from_spark_live(fitted, dataset=spark_mixed_df)).transform(spark_mixed_df)
        np.testing.assert_array_equal(_omle_pred(omle_out), _spark_pred(spark_out))
        np.testing.assert_allclose(_omle_prob(omle_out), _spark_prob(spark_out), atol=1e-4)
