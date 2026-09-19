"""Tests for omle_convert.spark (no-Spark reader).

Covers every supported Spark ML stage listed in the README.
The save step requires PySpark; the load step uses only pyarrow — no JVM needed.
"""
import pytest

try:
    import pyspark.ml.classification as C
    import pyspark.ml.clustering as CL
    import pyspark.ml.feature as F
    import pyspark.ml.regression as R
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import SparkSession
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

pytestmark = pytest.mark.skipif(not HAS_PYSPARK, reason="pyspark not installed")

import numpy as np

import omle
from omle_convert.spark import from_spark

# ── Fixtures ───────────────────────────────────────────────────────────────────




# ── Feature transformers ───────────────────────────────────────────────────────

class TestFeatureTransformers:

    def test_standard_scaler(self, binary_df, tmp_path):
        model = F.StandardScaler(
            inputCol="features", outputCol="out", withMean=True, withStd=True,
        ).fit(binary_df)
        model.save(str(tmp_path / "std_scaler"))
        m = from_spark(str(tmp_path / "std_scaler"))
        assert any(n.op == "StandardScaler" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "StandardScaler")
        assert node.name == "standard_scaler_model"           # snake_case of Spark class
        # input: featuresCol name, dtype FLOAT64, shape [-1, n_features]
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape  == [-1, 3]
        # node output: outputCol name, dtype FLOAT64, same shape as input
        no = node.outputs[0]
        assert no.name == "out"
        assert no.type.dtype  == omle.DataType.FLOAT64
        assert no.type.shape  == [-1, 3]
        # model output: same as node output for transformer
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape  == [-1, 3]

    def test_min_max_scaler(self, binary_df, tmp_path):
        model = F.MinMaxScaler(inputCol="features", outputCol="out").fit(binary_df)
        model.save(str(tmp_path / "mm_scaler"))
        m = from_spark(str(tmp_path / "mm_scaler"))
        assert any(n.op == "MinMaxScaler" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape  == [-1, 3]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]
        nd = next(n for n in m.nodes if n.op == "MinMaxScaler")
        assert nd.outputs[0].name == "out"
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1, 3]

    def test_max_abs_scaler(self, binary_df, tmp_path):
        model = F.MaxAbsScaler(inputCol="features", outputCol="out").fit(binary_df)
        model.save(str(tmp_path / "ma_scaler"))
        m = from_spark(str(tmp_path / "ma_scaler"))
        assert any(n.op == "MaxAbsScaler" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape  == [-1, 3]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 3]
        nd = next(n for n in m.nodes if n.op == "MaxAbsScaler")
        assert nd.outputs[0].name == "out"
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1, 3]

    def test_normalizer(self, binary_df, tmp_path):
        # Normalizer is a Transformer (no fit step), save it directly
        F.Normalizer(inputCol="features", outputCol="out", p=2.0).save(str(tmp_path / "norm"))
        m = from_spark(str(tmp_path / "norm"), n_features=3)
        assert any(n.op == "Normalizer" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape  == [-1, 3]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Normalizer")
        assert nd.outputs[0].name == "out"

    def test_binarizer(self, binary_df, tmp_path):
        F.Binarizer(inputCol="features", outputCol="out", threshold=3.0).save(str(tmp_path / "bin"))
        m = from_spark(str(tmp_path / "bin"), n_features=3)
        assert any(n.op == "Binarizer" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape  == [-1, 3]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Binarizer")
        assert nd.outputs[0].name == "out"

    def test_string_indexer_single(self, spark, tmp_path):
        df = spark.createDataFrame([("cat",), ("dog",), ("cat",), ("bird",), ("dog",), ("cat",)], ["animal"])
        model = F.StringIndexer(inputCol="animal", outputCol="animal_idx").fit(df)
        model.save(str(tmp_path / "si"))
        m = from_spark(str(tmp_path / "si"))
        assert any(n.op == "LabelEncoder" for n in m.nodes)
        assert m.inputs[0].name == "animal"
        assert m.inputs[0].type.dtype  == omle.DataType.STRING
        assert m.inputs[0].type.shape  == [-1]
        assert m.outputs[0].name == "animal_idx"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        enc_node = next(n for n in m.nodes if n.op == "LabelEncoder")
        assert enc_node.outputs[0].name == "animal_idx"
        assert enc_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert enc_node.outputs[0].type.shape == [-1]

    def test_string_indexer_multi_column(self, spark, tmp_path):
        df = spark.createDataFrame(
            [("cat", "red"), ("dog", "blue"), ("cat", "red"), ("bird", "green"), ("dog", "blue"), ("cat", "red")],
            ["animal", "color"],
        )
        model = F.StringIndexer(inputCols=["animal", "color"], outputCols=["a_idx", "c_idx"]).fit(df)
        model.save(str(tmp_path / "si_multi"))
        m = from_spark(str(tmp_path / "si_multi"))
        # single LabelEncoder node handles all columns (variadic xs/ys), no Concat needed
        ops = [n.op for n in m.nodes]
        assert ops.count("LabelEncoder") == 1
        enc_node = next(n for n in m.nodes if n.op == "LabelEncoder")
        for inp in m.inputs:
            assert inp.type.dtype  == omle.DataType.STRING
            assert inp.type.shape  == [-1]
        input_names = {i.name for i in m.inputs}
        assert "animal" in input_names
        assert "color"  in input_names
        node_out_names = {o.name for o in enc_node.outputs}
        assert node_out_names == {"a_idx", "c_idx"}
        for o in enc_node.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]
        output_names = {o.name for o in m.outputs}
        assert output_names == {"a_idx", "c_idx"}
        for o in m.outputs:
            assert o.type.dtype == omle.DataType.FLOAT64
            assert o.type.shape == [-1]

    def test_one_hot_encoder(self, spark, tmp_path):
        from pyspark.ml import Pipeline
        df = spark.createDataFrame([("cat",), ("dog",), ("cat",), ("bird",), ("dog",), ("cat",)], ["animal"])
        fitted = Pipeline(stages=[
            F.StringIndexer(inputCol="animal", outputCol="animal_idx"),
            F.OneHotEncoder(inputCol="animal_idx", outputCol="animal_ohe"),
        ]).fit(df)
        fitted.save(str(tmp_path / "ohe"))
        m = from_spark(str(tmp_path / "ohe"))
        assert any(n.op == "LabelEncoder"   for n in m.nodes)
        assert any(n.op == "OneHotEncoder"  for n in m.nodes)
        # pipeline: first stage inputCol → model input
        assert m.inputs[0].name == "animal"
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        # LabelEncoder intermediate output
        le_node  = next(n for n in m.nodes if n.op == "LabelEncoder")
        assert le_node.outputs[0].name == "animal_idx"
        assert le_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert le_node.outputs[0].type.shape == [-1]
        # OHE output: 3 categories → 2-wide vector (drop-last)
        ohe_node = next(n for n in m.nodes if n.op == "OneHotEncoder")
        assert ohe_node.outputs[0].name == "animal_ohe"
        assert ohe_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert ohe_node.outputs[0].type.shape == [-1, 2]
        assert m.outputs[0].name == "animal_ohe"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 2]

    def test_bucketizer(self, binary_df, tmp_path):
        F.Bucketizer(
            inputCol="features", outputCol="out",
            splits=[-float("inf"), 2.0, 5.0, float("inf")],
        ).save(str(tmp_path / "bucket"))
        m = from_spark(str(tmp_path / "bucket"), n_features=3)
        assert any(n.op == "Bucketizer" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape  == [-1]
        assert m.outputs[0].name == "out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Bucketizer")
        assert nd.outputs[0].name == "out"

    def test_imputer(self, spark, tmp_path):
        from pyspark.sql.types import DoubleType, StructField, StructType
        schema = StructType([
            StructField("a", DoubleType(), True),
            StructField("b", DoubleType(), True),
        ])
        df = spark.createDataFrame([(1.0, 2.0), (None, 3.0), (3.0, None), (4.0, 5.0)], schema)
        model = F.Imputer(inputCols=["a", "b"], outputCols=["a_out", "b_out"]).fit(df)
        model.save(str(tmp_path / "imputer"))
        m = from_spark(str(tmp_path / "imputer"))
        assert any(n.op == "Imputer" for n in m.nodes)
        # inputCols → per-column InputSpecs (scalar FLOAT64)
        input_names  = {i.name for i in m.inputs}
        assert input_names == {"a", "b"}
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        output_names = {o.name for o in m.outputs}
        assert output_names == {"a_out", "b_out"}
        for out in m.outputs:
            assert out.type.dtype == omle.DataType.FLOAT64
            assert out.type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Imputer")
        for o in nd.outputs:
            assert o.name in {"a_out", "b_out"}

    def test_pca(self, binary_df, tmp_path):
        model = F.PCA(k=2, inputCol="features", outputCol="pca_out").fit(binary_df)
        model.save(str(tmp_path / "pca"))
        m = from_spark(str(tmp_path / "pca"))
        assert any(n.op == "PCA" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "PCA")
        n_comp = next(a.i for a in node.attributes if a.name == "n_components")
        assert n_comp == 2
        # input: featuresCol, dtype FLOAT64
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        # node output: outputCol, FLOAT64, shape [-1, k], field_names = ["pca0","pca1"]
        no = node.outputs[0]
        assert no.name == "pca_out"
        assert no.type.dtype  == omle.DataType.FLOAT64
        assert no.type.shape  == [-1, 2]               # k=2 components
        assert no.field_names == ["pca0", "pca1"]
        # model output reflects same type
        assert m.outputs[0].name == "pca_out"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1, 2]

    def test_vector_assembler_in_pipeline(self, spark, tmp_path):
        from pyspark.ml import Pipeline
        df = spark.createDataFrame([(1.0, 2.0, 0.0), (3.0, 4.0, 1.0), (5.0, 6.0, 0.0),
                                    (7.0, 8.0, 1.0), (2.0, 1.0, 0.0), (4.0, 3.0, 1.0)], ["a", "b", "label"])
        fitted = Pipeline(stages=[
            F.VectorAssembler(inputCols=["a", "b"], outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=5),
        ]).fit(df)
        fitted.save(str(tmp_path / "va"))
        m = from_spark(str(tmp_path / "va"), feature_names=["a", "b"])
        assert any(n.op == "Linear" for n in m.nodes)
        # scalar inputs: FLOAT64 [-1]
        assert m.inputs[0].name == "a" and m.inputs[1].name == "b"
        for inp in m.inputs:
            assert inp.type.dtype == omle.DataType.FLOAT64
            assert inp.type.shape == [-1]
        # Concat node assembles into [-1, 2]
        concat_node = next(n for n in m.nodes if n.op == "Concat")
        assert concat_node.outputs[0].name == "features"
        assert concat_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert concat_node.outputs[0].type.shape == [-1, 2]
        assert concat_node.outputs[0].field_names == ["a", "b"]
        # LR outputs
        output_names = {o.name for o in m.outputs}
        assert "prediction"  in output_names
        assert "probability" in output_names
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_vector_slicer(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.VectorSlicer(inputCol="features", outputCol="sliced", indices=[0, 2]),
            C.LogisticRegression(featuresCol="sliced", maxIter=5),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "vs"))
        m = from_spark(str(tmp_path / "vs"), n_features=3)
        ops = [n.op for n in m.nodes]
        assert "TakeSlots" in ops
        assert "Linear" in ops
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        # TakeSlots output: 2 selected indices
        ts_node = next(n for n in m.nodes if n.op == "TakeSlots")
        assert ts_node.outputs[0].name == "sliced"
        assert ts_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert ts_node.outputs[0].type.shape == [-1, 2]
        # LR final outputs
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64
        assert pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64
        assert prob_out.type.shape == [-1, 2]

    def test_chisq_selector(self, nonneg_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.ChiSqSelector(numTopFeatures=2, featuresCol="features",
                            labelCol="label", outputCol="selected"),
            C.LogisticRegression(featuresCol="selected", maxIter=5),
        ]).fit(nonneg_df)
        fitted.save(str(tmp_path / "chisq"))
        m = from_spark(str(tmp_path / "chisq"), n_features=3)
        assert any(n.op == "TakeSlots" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        # TakeSlots output: 2 top features
        ts_node = next(n for n in m.nodes if n.op == "TakeSlots")
        assert ts_node.outputs[0].name == "selected"
        assert ts_node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert ts_node.outputs[0].type.shape == [-1, 2]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64  and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]

    def test_univariate_feature_selector(self, binary_df, tmp_path):
        if not hasattr(F, "UnivariateFeatureSelector"):
            pytest.skip("UnivariateFeatureSelector requires Spark 3.1+")
        from pyspark.ml import Pipeline
        sel = (F.UnivariateFeatureSelector(
                    featuresCol="features", labelCol="label", outputCol="selected",
                    selectionMode="numTopFeatures")
               .setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(2))
        fitted = Pipeline(stages=[sel, C.LogisticRegression(featuresCol="selected", maxIter=5)]).fit(binary_df)
        fitted.save(str(tmp_path / "uvfs"))
        m = from_spark(str(tmp_path / "uvfs"), n_features=3)
        assert any(n.op == "TakeSlots" for n in m.nodes)

    def test_variance_threshold_selector(self, binary_df, tmp_path):
        if not hasattr(F, "VarianceThresholdSelector"):
            pytest.skip("VarianceThresholdSelector requires Spark 3.1+")
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.VarianceThresholdSelector(featuresCol="features", outputCol="selected",
                                        varianceThreshold=0.0),
            C.LogisticRegression(featuresCol="selected", maxIter=5),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "vts"))
        m = from_spark(str(tmp_path / "vts"), n_features=3)
        assert any(n.op == "TakeSlots" for n in m.nodes)


# ── Estimators ─────────────────────────────────────────────────────────────────

class TestEstimators:

    def test_logistic_regression_binary(self, binary_df, tmp_path):
        model = C.LogisticRegression(maxIter=5).fit(binary_df)
        model.save(str(tmp_path / "lr_bin"))
        m = from_spark(str(tmp_path / "lr_bin"))
        assert any(n.op == "Linear" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype  == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape  == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]
        nd = next(n for n in m.nodes if n.op == "Linear")
        nd_pred = next(o for o in nd.outputs if o.name == "prediction")
        nd_prob = next(o for o in nd.outputs if o.name == "probability")
        assert nd_pred.type.dtype == omle.DataType.INT64   and nd_pred.type.shape == [-1]
        assert nd_prob.type.dtype == omle.DataType.FLOAT64 and nd_prob.type.shape == [-1, 2]
        assert m.model_schema.targets[0].name == "label"

    def test_logistic_regression_multiclass(self, multiclass_df, tmp_path):
        model = C.LogisticRegression(maxIter=5).fit(multiclass_df)
        model.save(str(tmp_path / "lr_multi"))
        m = from_spark(str(tmp_path / "lr_multi"))
        assert any(n.op == "Linear" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 3]

    def test_linear_regression(self, regression_df, tmp_path):
        model = R.LinearRegression(maxIter=5).fit(regression_df)
        model.save(str(tmp_path / "lr_reg"))
        m = from_spark(str(tmp_path / "lr_reg"))
        assert any(n.op == "Linear" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"prediction"}
        pred_out = m.outputs[0]
        assert pred_out.type.dtype == omle.DataType.FLOAT64
        assert pred_out.type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Linear")
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]
        assert m.model_schema.targets[0].name == "label"

    def test_linear_svc(self, binary_df, tmp_path):
        model = C.LinearSVC(maxIter=5).fit(binary_df)
        model.save(str(tmp_path / "lsvc"))
        m = from_spark(str(tmp_path / "lsvc"))
        assert any(n.op == "SVM" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]
        nd = next(n for n in m.nodes if n.op == "SVM")
        nd_pred = next(o for o in nd.outputs if o.name == "prediction")
        nd_prob = next(o for o in nd.outputs if o.name == "probability")
        assert nd_pred.type.dtype == omle.DataType.INT64   and nd_pred.type.shape == [-1]
        assert nd_prob.type.dtype == omle.DataType.FLOAT64 and nd_prob.type.shape == [-1, 2]

    def test_generalized_linear_regression_identity(self, regression_df, tmp_path):
        model = R.GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=5).fit(regression_df)
        model.save(str(tmp_path / "glm_id"))
        m = from_spark(str(tmp_path / "glm_id"))
        assert any(n.op == "Linear" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert m.outputs[0].name == "prediction"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Linear")
        assert nd.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert nd.outputs[0].type.shape == [-1]

    def test_generalized_linear_regression_log(self, regression_df, tmp_path):
        model = R.GeneralizedLinearRegression(family="poisson", link="log", maxIter=5).fit(regression_df)
        model.save(str(tmp_path / "glm_log"))
        m = from_spark(str(tmp_path / "glm_log"))
        node = next(n for n in m.nodes if n.op == "Linear")
        assert node.linear.post_transform.name == "EXP"
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert m.outputs[0].name == "prediction"
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert node.outputs[0].type.shape == [-1]

    def test_dt_classifier_binary(self, binary_df, tmp_path):
        model = C.DecisionTreeClassifier(maxDepth=3).fit(binary_df)
        model.save(str(tmp_path / "dt_clf"))
        m = from_spark(str(tmp_path / "dt_clf"))
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert len(node.tree_ensemble.trees) == 1
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]
        nd_pred = next(o for o in node.outputs if o.name == "prediction")
        nd_prob = next(o for o in node.outputs if o.name == "probability")
        assert nd_pred.type.dtype == omle.DataType.INT64   and nd_pred.type.shape == [-1]
        assert nd_prob.type.dtype == omle.DataType.FLOAT64 and nd_prob.type.shape == [-1, 2]

    def test_dt_classifier_multiclass(self, multiclass_df, tmp_path):
        model = C.DecisionTreeClassifier(maxDepth=3).fit(multiclass_df)
        model.save(str(tmp_path / "dt_clf_mc"))
        m = from_spark(str(tmp_path / "dt_clf_mc"))
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert len(node.tree_ensemble.trees) == 3  # one tree per class
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 3]

    def test_dt_regressor(self, regression_df, tmp_path):
        model = R.DecisionTreeRegressor(maxDepth=3).fit(regression_df)
        model.save(str(tmp_path / "dt_reg"))
        m = from_spark(str(tmp_path / "dt_reg"))
        assert any(n.op == "TreeEnsemble" for n in m.nodes)
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble.trees[0].num_nodes > 0
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"prediction"}
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert node.outputs[0].type.shape == [-1]

    def test_rf_classifier_binary(self, binary_df, tmp_path):
        model = C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "rf_clf"))
        m = from_spark(str(tmp_path / "rf_clf"))
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert len(node.tree_ensemble.trees) == 3
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]
        nd_pred = next(o for o in node.outputs if o.name == "prediction")
        nd_prob = next(o for o in node.outputs if o.name == "probability")
        assert nd_pred.type.dtype == omle.DataType.INT64   and nd_pred.type.shape == [-1]
        assert nd_prob.type.dtype == omle.DataType.FLOAT64 and nd_prob.type.shape == [-1, 2]

    def test_rf_classifier_multiclass(self, multiclass_df, tmp_path):
        model = C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42).fit(multiclass_df)
        model.save(str(tmp_path / "rf_clf_mc"))
        m = from_spark(str(tmp_path / "rf_clf_mc"))
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert len(node.tree_ensemble.trees) == 9  # 3 trees × 3 classes
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 3]

    def test_rf_regressor(self, regression_df, tmp_path):
        model = R.RandomForestRegressor(numTrees=3, maxDepth=2, seed=42).fit(regression_df)
        model.save(str(tmp_path / "rf_reg"))
        m = from_spark(str(tmp_path / "rf_reg"))
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert len(node.tree_ensemble.trees) == 3
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"prediction"}
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert node.outputs[0].type.shape == [-1]

    def test_gbt_classifier(self, binary_df, tmp_path):
        model = C.GBTClassifier(maxIter=3, maxDepth=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "gbt_clf"))
        m = from_spark(str(tmp_path / "gbt_clf"))
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble.post_transform.name == "SIGMOID"
        assert node.tree_ensemble.aggregation.name == "SUM"
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]
        nd_pred = next(o for o in node.outputs if o.name == "prediction")
        nd_prob = next(o for o in node.outputs if o.name == "probability")
        assert nd_pred.type.dtype == omle.DataType.INT64   and nd_pred.type.shape == [-1]
        assert nd_prob.type.dtype == omle.DataType.FLOAT64 and nd_prob.type.shape == [-1, 2]

    def test_gbt_regressor(self, regression_df, tmp_path):
        model = R.GBTRegressor(maxIter=3, maxDepth=2, seed=42).fit(regression_df)
        model.save(str(tmp_path / "gbt_reg"))
        m = from_spark(str(tmp_path / "gbt_reg"))
        node = next(n for n in m.nodes if n.op == "TreeEnsemble")
        assert node.tree_ensemble.aggregation.name == "SUM"
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 2]
        assert {o.name for o in m.outputs} == {"prediction"}
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.outputs[0].type.shape == [-1]
        assert node.outputs[0].type.dtype == omle.DataType.FLOAT64
        assert node.outputs[0].type.shape == [-1]

    def test_naive_bayes_multinomial(self, nonneg_df, tmp_path):
        model = C.NaiveBayes(modelType="multinomial").fit(nonneg_df)
        model.save(str(tmp_path / "nb_multi"))
        m = from_spark(str(tmp_path / "nb_multi"))
        assert any(n.op == "NaiveBayes" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]

    def test_naive_bayes_bernoulli(self, binary_feat_df, tmp_path):
        model = C.NaiveBayes(modelType="bernoulli").fit(binary_feat_df)
        model.save(str(tmp_path / "nb_bern"))
        m = from_spark(str(tmp_path / "nb_bern"))
        assert any(n.op == "NaiveBayes" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]

    def test_naive_bayes_gaussian(self, binary_df, tmp_path):
        model = C.NaiveBayes(modelType="gaussian").fit(binary_df)
        model.save(str(tmp_path / "nb_gauss"))
        m = from_spark(str(tmp_path / "nb_gauss"))
        assert any(n.op == "NaiveBayes" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]

    def test_mlp_classifier(self, binary_df, tmp_path):
        model = C.MultilayerPerceptronClassifier(layers=[3, 4, 2], maxIter=10, seed=42).fit(binary_df)
        model.save(str(tmp_path / "mlp"))
        m = from_spark(str(tmp_path / "mlp"))
        assert any(n.op == "NeuralNetwork" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        pred_out = next(o for o in m.outputs if o.name == "prediction")
        prob_out = next(o for o in m.outputs if o.name == "probability")
        assert pred_out.type.dtype == omle.DataType.INT64   and pred_out.type.shape == [-1]
        assert prob_out.type.dtype == omle.DataType.FLOAT64 and prob_out.type.shape == [-1, 2]
        nd = next(n for n in m.nodes if n.op == "NeuralNetwork")
        nd_pred = next(o for o in nd.outputs if o.name == "prediction")
        nd_prob = next(o for o in nd.outputs if o.name == "probability")
        assert nd_pred.type.dtype == omle.DataType.INT64   and nd_pred.type.shape == [-1]
        assert nd_prob.type.dtype == omle.DataType.FLOAT64 and nd_prob.type.shape == [-1, 2]

    def test_kmeans(self, binary_df, tmp_path):
        model = CL.KMeans(k=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "kmeans"))
        m = from_spark(str(tmp_path / "kmeans"))
        assert any(n.op == "Clustering" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert {o.name for o in m.outputs} == {"prediction"}
        assert m.outputs[0].type.dtype == omle.DataType.INT32
        assert m.outputs[0].type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Clustering")
        assert nd.outputs[0].type.dtype == omle.DataType.INT32
        assert nd.outputs[0].type.shape == [-1]
        assert m.model_schema.targets == []                     # unsupervised

    def test_gaussian_mixture(self, binary_df, tmp_path):
        model = CL.GaussianMixture(k=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "gmm"))
        m = from_spark(str(tmp_path / "gmm"))
        assert any(n.op == "Clustering" for n in m.nodes)
        assert m.inputs[0].name == "features"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 3]
        assert {o.name for o in m.outputs} == {"prediction"}
        assert m.outputs[0].type.dtype == omle.DataType.INT32
        assert m.outputs[0].type.shape == [-1]
        nd = next(n for n in m.nodes if n.op == "Clustering")
        assert nd.outputs[0].type.dtype == omle.DataType.INT32
        assert nd.outputs[0].type.shape == [-1]
        assert m.model_schema.targets == []

    def test_cross_validator(self, binary_df, tmp_path):
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
        lr   = C.LogisticRegression(maxIter=5)
        grid = ParamGridBuilder().addGrid(lr.maxIter, [5]).build()
        model = CrossValidator(
            estimator=lr, estimatorParamMaps=grid,
            evaluator=BinaryClassificationEvaluator(), numFolds=2, seed=42,
        ).fit(binary_df)
        model.save(str(tmp_path / "cv"))
        m = from_spark(str(tmp_path / "cv"))
        assert any(n.op == "Linear" for n in m.nodes)

    def test_train_validation_split(self, binary_df, tmp_path):
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
        lr   = C.LogisticRegression(maxIter=5)
        grid = ParamGridBuilder().addGrid(lr.maxIter, [5]).build()
        model = TrainValidationSplit(
            estimator=lr, estimatorParamMaps=grid,
            evaluator=BinaryClassificationEvaluator(), trainRatio=0.8, seed=42,
        ).fit(binary_df)
        model.save(str(tmp_path / "tvs"))
        m = from_spark(str(tmp_path / "tvs"))
        assert any(n.op == "Linear" for n in m.nodes)


# ── Text transformers ──────────────────────────────────────────────────────────

class TestTextTransformers:

    def test_tokenizer(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "tok"))
        m = from_spark(str(tmp_path / "tok"))
        assert any(n.op == "Tokenizer" for n in m.nodes)

    def test_regex_tokenizer(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.RegexTokenizer(inputCol="text", outputCol="tokens", pattern=r"\s+"),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "re_tok"))
        m = from_spark(str(tmp_path / "re_tok"))
        assert any(n.op == "RegexTokenizer" for n in m.nodes)

    def test_stop_words_remover(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.StopWordsRemover(inputCol="tokens", outputCol="filtered"),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "swr"))
        m = from_spark(str(tmp_path / "swr"))
        assert any(n.op == "StopWordsRemover" for n in m.nodes)

    def test_ngram(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.NGram(n=2, inputCol="tokens", outputCol="ngrams"),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "ngram"))
        m = from_spark(str(tmp_path / "ngram"))
        node = next(n for n in m.nodes if n.op == "NGram")
        n_min = next(a.i for a in node.attributes if a.name == "n_min")
        n_max = next(a.i for a in node.attributes if a.name == "n_max")
        assert n_min == 2 and n_max == 2

    def test_count_vectorizer(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.CountVectorizer(inputCol="tokens", outputCol="vec"),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "cv"))
        m = from_spark(str(tmp_path / "cv"))
        assert any(n.op == "CountVectorizer" for n in m.nodes)

    def test_hashing_tf(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.HashingTF(inputCol="tokens", outputCol="tf_vec", numFeatures=100),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "htf"))
        m = from_spark(str(tmp_path / "htf"))
        assert any(n.op == "HashingVectorizer" for n in m.nodes)

    def test_idf(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.HashingTF(inputCol="tokens", outputCol="tf_vec", numFeatures=100),
            F.IDF(inputCol="tf_vec", outputCol="tfidf_vec"),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "idf"))
        m = from_spark(str(tmp_path / "idf"))
        assert any(n.op == "TfIdfTransformer" for n in m.nodes)

    def test_word2vec(self, text_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Tokenizer(inputCol="text", outputCol="tokens"),
            F.Word2Vec(inputCol="tokens", outputCol="w2v", vectorSize=4, minCount=1, seed=42),
        ]).fit(text_df)
        fitted.save(str(tmp_path / "w2v"))
        m = from_spark(str(tmp_path / "w2v"))
        assert any(n.op == "Word2Vec" for n in m.nodes)


# ── Integration / round-trip tests ────────────────────────────────────────────

class TestIntegration:

    def test_multi_stage_pipeline(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.StandardScaler(inputCol="features", outputCol="scaled", withMean=True),
            C.LogisticRegression(featuresCol="scaled", maxIter=5),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "multi"))
        m = from_spark(str(tmp_path / "multi"))
        ops = [n.op for n in m.nodes]
        assert "StandardScaler" in ops
        assert "Linear" in ops

    def test_minmax_rf_pipeline(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.MinMaxScaler(inputCol="features", outputCol="scaled"),
            C.RandomForestClassifier(featuresCol="scaled", numTrees=2, maxDepth=2, seed=1),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "mm_rf"))
        m = from_spark(str(tmp_path / "mm_rf"))
        ops = [n.op for n in m.nodes]
        assert "MinMaxScaler" in ops
        assert "TreeEnsemble" in ops

    def test_rf_pipeline_json_roundtrip(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "rf"))
        model = from_spark(str(tmp_path / "rf"))

        json_path = str(tmp_path / "rf.json")
        omle.save_json(model, json_path)
        loaded = omle.load_json(json_path)

        assert any(n.op == "TreeEnsemble" for n in loaded.nodes)
        orig_trees  = next(n for n in model.nodes  if n.op == "TreeEnsemble").tree_ensemble.trees
        loaded_trees = next(n for n in loaded.nodes if n.op == "TreeEnsemble").tree_ensemble.trees
        assert len(orig_trees) == len(loaded_trees)

    def test_pca_lr_pipeline(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.PCA(k=2, inputCol="features", outputCol="pca_out"),
            C.LogisticRegression(featuresCol="pca_out", maxIter=5),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "pca_lr"))
        m = from_spark(str(tmp_path / "pca_lr"))
        ops = [n.op for n in m.nodes]
        assert "PCA" in ops
        assert "Linear" in ops

    def test_mixed_type_indexer_assembler_lr(self, spark_mixed_df, tmp_path):
        """StringIndexer + VectorAssembler (float index + numerics) + LR on raw mixed-type data."""
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.StringIndexer(inputCol="color", outputCol="color_idx"),
            F.VectorAssembler(inputCols=["color_idx", "size", "weight"], outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=10),
        ]).fit(spark_mixed_df)
        fitted.save(str(tmp_path / "mixed_si_lr"))
        m = from_spark(str(tmp_path / "mixed_si_lr"))
        ops = [n.op for n in m.nodes]
        assert "LabelEncoder" in ops
        assert "Concat" in ops
        assert "Linear" in ops

    def test_mixed_type_ohe_assembler_rf(self, spark_mixed_df, tmp_path):
        """Full categorical pipeline: StringIndexer → OHE → VectorAssembler → RF."""
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.StringIndexer(inputCol="color", outputCol="color_idx"),
            F.OneHotEncoder(inputCols=["color_idx"], outputCols=["color_vec"]),
            F.VectorAssembler(inputCols=["color_vec", "size", "weight", "is_premium"],
                              outputCol="features"),
            C.RandomForestClassifier(featuresCol="features", numTrees=5, maxDepth=2, seed=42),
        ]).fit(spark_mixed_df)
        fitted.save(str(tmp_path / "mixed_ohe_rf"))
        m = from_spark(str(tmp_path / "mixed_ohe_rf"))
        ops = [n.op for n in m.nodes]
        assert "LabelEncoder" in ops
        assert "TreeEnsemble" in ops

    def test_mixed_type_ohe_assembler_gbt(self, spark_mixed_df, tmp_path):
        """StringIndexer → OHE → VectorAssembler → GBT pipeline on raw mixed-type data."""
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.StringIndexer(inputCol="color", outputCol="color_idx"),
            F.OneHotEncoder(inputCols=["color_idx"], outputCols=["color_vec"]),
            F.VectorAssembler(inputCols=["color_vec", "size", "weight"], outputCol="features"),
            C.GBTClassifier(featuresCol="features", maxIter=5, maxDepth=2, seed=42),
        ]).fit(spark_mixed_df)
        fitted.save(str(tmp_path / "mixed_ohe_gbt"))
        m = from_spark(str(tmp_path / "mixed_ohe_gbt"))
        ops = [n.op for n in m.nodes]
        assert "LabelEncoder" in ops
        assert "TreeEnsemble" in ops

    def test_mixed_numeric_assembler_lr_json_roundtrip(self, spark_mixed_df, tmp_path):
        """VectorAssembler from scalar numeric columns + LR: JSON round-trip."""
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.VectorAssembler(inputCols=["size", "weight", "is_premium"], outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=10),
        ]).fit(spark_mixed_df)
        fitted.save(str(tmp_path / "num_asm_lr"))
        m = from_spark(str(tmp_path / "num_asm_lr"), feature_names=["size", "weight", "is_premium"])
        input_names = {inp.name for inp in m.inputs}
        assert {"size", "weight", "is_premium"} <= input_names
        json_path = str(tmp_path / "num_asm_lr.json")
        omle.save_json(m, json_path)
        loaded = omle.load_json(json_path)
        assert any(n.op == "Linear" for n in loaded.nodes)

    def test_zip_archive_input(self, binary_df, tmp_path):
        import os
        import zipfile

        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            C.RandomForestClassifier(numTrees=2, maxDepth=2, seed=42),
        ]).fit(binary_df)
        model_dir = str(tmp_path / "rf_for_zip")
        fitted.save(model_dir)

        zip_path = str(tmp_path / "rf.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(model_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    zf.write(abs_path, os.path.relpath(abs_path, tmp_path))

        # omle-convert CLI uses _open_spark_input which extracts the zip
        from omle_convert.cli import _open_spark_input
        with _open_spark_input(zip_path) as extracted:
            m = from_spark(extracted)
        assert any(n.op == "TreeEnsemble" for n in m.nodes)


# ── Runtime numerical verification ────────────────────────────────────────────

class TestRuntimeVerification:
    """End-to-end numerical checks: omle-convert → omle_runtime vs Spark predictions."""

    @staticmethod
    def _load_runtime(ir):
        try:
            import omle_runtime as omr

            from omle.proto.convert import ir_to_proto
        except ImportError:
            pytest.skip("omle_runtime not available")
        return omr.load_bytes(ir_to_proto(ir).SerializeToString())

    @staticmethod
    def _to_numpy(df, col="features"):
        return np.array([row[col].toArray() for row in df.select(col).collect()], dtype=np.float64)

    @staticmethod
    def _spark_proba(preds_df):
        return np.array([row["probability"].toArray() for row in preds_df.select("probability").collect()])

    @staticmethod
    def _spark_pred(preds_df):
        return np.array([row["prediction"] for row in preds_df.select("prediction").collect()])

    def test_lr_binary(self, binary_df, tmp_path):
        model = C.LogisticRegression(maxIter=20).fit(binary_df)
        model.save(str(tmp_path / "lr_bin"))
        X = self._to_numpy(binary_df)
        spark_out = model.transform(binary_df)
        ir = from_spark(str(tmp_path / "lr_bin"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_lr_multiclass(self, multiclass_df, tmp_path):
        model = C.LogisticRegression(maxIter=20).fit(multiclass_df)
        model.save(str(tmp_path / "lr_mc"))
        X = self._to_numpy(multiclass_df)
        spark_out = model.transform(multiclass_df)
        ir = from_spark(str(tmp_path / "lr_mc"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_linear_regression(self, regression_df, tmp_path):
        model = R.LinearRegression(maxIter=20).fit(regression_df)
        model.save(str(tmp_path / "lin_reg"))
        X = self._to_numpy(regression_df)
        spark_pred = self._spark_pred(model.transform(regression_df))
        ir = from_spark(str(tmp_path / "lin_reg"))
        m = self._load_runtime(ir)
        np.testing.assert_allclose(m.predict(X).ravel(), spark_pred, rtol=1e-4, atol=1e-4)

    def test_dt_binary(self, binary_df, tmp_path):
        model = C.DecisionTreeClassifier(maxDepth=3, seed=42).fit(binary_df)
        model.save(str(tmp_path / "dt_bin"))
        X = self._to_numpy(binary_df)
        spark_out = model.transform(binary_df)
        ir = from_spark(str(tmp_path / "dt_bin"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_dt_multiclass(self, multiclass_df, tmp_path):
        model = C.DecisionTreeClassifier(maxDepth=3, seed=42).fit(multiclass_df)
        model.save(str(tmp_path / "dt_mc"))
        X = self._to_numpy(multiclass_df)
        spark_out = model.transform(multiclass_df)
        ir = from_spark(str(tmp_path / "dt_mc"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_dt_regressor(self, regression_df, tmp_path):
        model = R.DecisionTreeRegressor(maxDepth=3, seed=42).fit(regression_df)
        model.save(str(tmp_path / "dt_reg"))
        X = self._to_numpy(regression_df)
        spark_pred = self._spark_pred(model.transform(regression_df))
        ir = from_spark(str(tmp_path / "dt_reg"))
        m = self._load_runtime(ir)
        np.testing.assert_allclose(m.predict(X).ravel(), spark_pred, rtol=1e-4, atol=1e-4)

    def test_rf_binary(self, binary_df, tmp_path):
        model = C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "rf_bin"))
        X = self._to_numpy(binary_df)
        spark_out = model.transform(binary_df)
        ir = from_spark(str(tmp_path / "rf_bin"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_rf_multiclass(self, multiclass_df, tmp_path):
        model = C.RandomForestClassifier(numTrees=3, maxDepth=2, seed=42).fit(multiclass_df)
        model.save(str(tmp_path / "rf_mc"))
        X = self._to_numpy(multiclass_df)
        spark_out = model.transform(multiclass_df)
        ir = from_spark(str(tmp_path / "rf_mc"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_rf_regressor(self, regression_df, tmp_path):
        model = R.RandomForestRegressor(numTrees=3, maxDepth=2, seed=42).fit(regression_df)
        model.save(str(tmp_path / "rf_reg"))
        X = self._to_numpy(regression_df)
        spark_pred = self._spark_pred(model.transform(regression_df))
        ir = from_spark(str(tmp_path / "rf_reg"))
        m = self._load_runtime(ir)
        np.testing.assert_allclose(m.predict(X).ravel(), spark_pred, rtol=1e-4, atol=1e-4)

    def test_gbt_binary(self, binary_df, tmp_path):
        model = C.GBTClassifier(maxIter=5, maxDepth=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "gbt_bin"))
        X = self._to_numpy(binary_df)
        spark_out = model.transform(binary_df)
        ir = from_spark(str(tmp_path / "gbt_bin"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_gbt_regressor(self, regression_df, tmp_path):
        model = R.GBTRegressor(maxIter=5, maxDepth=2, seed=42).fit(regression_df)
        model.save(str(tmp_path / "gbt_reg"))
        X = self._to_numpy(regression_df)
        spark_pred = self._spark_pred(model.transform(regression_df))
        ir = from_spark(str(tmp_path / "gbt_reg"))
        m = self._load_runtime(ir)
        np.testing.assert_allclose(m.predict(X).ravel(), spark_pred, rtol=1e-4, atol=1e-4)

    def test_naive_bayes_multinomial(self, nonneg_df, tmp_path):
        model = C.NaiveBayes(modelType="multinomial").fit(nonneg_df)
        model.save(str(tmp_path / "nb_multi"))
        X = self._to_numpy(nonneg_df)
        spark_out = model.transform(nonneg_df)
        ir = from_spark(str(tmp_path / "nb_multi"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_naive_bayes_bernoulli(self, binary_feat_df, tmp_path):
        model = C.NaiveBayes(modelType="bernoulli").fit(binary_feat_df)
        model.save(str(tmp_path / "nb_bern"))
        X = self._to_numpy(binary_feat_df)
        spark_out = model.transform(binary_feat_df)
        ir = from_spark(str(tmp_path / "nb_bern"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_naive_bayes_gaussian(self, binary_df, tmp_path):
        model = C.NaiveBayes(modelType="gaussian").fit(binary_df)
        model.save(str(tmp_path / "nb_gauss"))
        X = self._to_numpy(binary_df)
        spark_out = model.transform(binary_df)
        ir = from_spark(str(tmp_path / "nb_gauss"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_linear_svc(self, binary_df, tmp_path):
        model = C.LinearSVC(maxIter=5).fit(binary_df)
        model.save(str(tmp_path / "lsvc"))
        X = self._to_numpy(binary_df)
        spark_pred = self._spark_pred(model.transform(binary_df))
        ir = from_spark(str(tmp_path / "lsvc"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), spark_pred)

    def test_mlp_binary(self, binary_df, tmp_path):
        model = C.MultilayerPerceptronClassifier(layers=[3, 4, 2], maxIter=10, seed=42).fit(binary_df)
        model.save(str(tmp_path / "mlp"))
        X = self._to_numpy(binary_df)
        spark_out = model.transform(binary_df)
        ir = from_spark(str(tmp_path / "mlp"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_stdscaler_lr(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.StandardScaler(inputCol="features", outputCol="scaled", withMean=True),
            C.LogisticRegression(featuresCol="scaled", maxIter=20),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "std_lr"))
        X = self._to_numpy(binary_df)
        spark_out = fitted.transform(binary_df)
        ir = from_spark(str(tmp_path / "std_lr"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_minmax_rf(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.MinMaxScaler(inputCol="features", outputCol="scaled"),
            C.RandomForestClassifier(featuresCol="scaled", numTrees=3, maxDepth=2, seed=42),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "mm_rf"))
        X = self._to_numpy(binary_df)
        spark_out = fitted.transform(binary_df)
        ir = from_spark(str(tmp_path / "mm_rf"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_maxabs_lr(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.MaxAbsScaler(inputCol="features", outputCol="scaled"),
            C.LogisticRegression(featuresCol="scaled", maxIter=20),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "ma_lr"))
        X = self._to_numpy(binary_df)
        spark_out = fitted.transform(binary_df)
        ir = from_spark(str(tmp_path / "ma_lr"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_normalizer_lr(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.Normalizer(inputCol="features", outputCol="normed", p=2.0),
            C.LogisticRegression(featuresCol="normed", maxIter=20),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "norm_lr"))
        X = self._to_numpy(binary_df)
        spark_out = fitted.transform(binary_df)
        ir = from_spark(str(tmp_path / "norm_lr"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_pca_dt(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.PCA(k=2, inputCol="features", outputCol="pca_out"),
            C.DecisionTreeClassifier(featuresCol="pca_out", maxDepth=3, seed=42),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "pca_dt"))
        X = self._to_numpy(binary_df)
        spark_out = fitted.transform(binary_df)
        ir = from_spark(str(tmp_path / "pca_dt"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_vector_slicer_lr(self, binary_df, tmp_path):
        from pyspark.ml import Pipeline
        fitted = Pipeline(stages=[
            F.VectorSlicer(inputCol="features", outputCol="sliced", indices=[0, 2]),
            C.LogisticRegression(featuresCol="sliced", maxIter=20),
        ]).fit(binary_df)
        fitted.save(str(tmp_path / "vs_lr"))
        X = self._to_numpy(binary_df)
        spark_out = fitted.transform(binary_df)
        ir = from_spark(str(tmp_path / "vs_lr"), n_features=3)
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_imputer_lr(self, spark, tmp_path):
        from pyspark.ml import Pipeline
        # Imputer requires scalar numeric columns (not Vector); use VectorAssembler after
        data = [
            (1.0, 2.0, 3.0, 0.0),
            (float("nan"), 5.0, 6.0, 1.0),
            (2.0, float("nan"), 4.0, 0.0),
            (5.0, 6.0, 7.0, 1.0),
            (3.0, 4.0, 5.0, 0.0),
            (6.0, 7.0, float("nan"), 1.0),
        ]
        imp_df = spark.createDataFrame(data, ["f0", "f1", "f2", "label"])
        fitted = Pipeline(stages=[
            F.Imputer(inputCols=["f0", "f1", "f2"], outputCols=["imp0", "imp1", "imp2"]),
            F.VectorAssembler(inputCols=["imp0", "imp1", "imp2"], outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=20),
        ]).fit(imp_df)
        fitted.save(str(tmp_path / "imp_lr"))
        X = np.array([r[:3] for r in imp_df.select("f0","f1","f2").collect()], dtype=np.float64)
        spark_out = fitted.transform(imp_df)
        ir = from_spark(str(tmp_path / "imp_lr"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_glm_identity_regression(self, regression_df, tmp_path):
        model = R.GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=20).fit(regression_df)
        model.save(str(tmp_path / "glm_id"))
        X = self._to_numpy(regression_df)
        spark_pred = self._spark_pred(model.transform(regression_df))
        ir = from_spark(str(tmp_path / "glm_id"))
        m = self._load_runtime(ir)
        np.testing.assert_allclose(m.predict(X).ravel(), spark_pred, rtol=1e-4, atol=1e-4)

    def test_glm_log_regression(self, regression_df, tmp_path):
        model = R.GeneralizedLinearRegression(family="poisson", link="log", maxIter=20).fit(regression_df)
        model.save(str(tmp_path / "glm_log"))
        X = self._to_numpy(regression_df)
        spark_pred = self._spark_pred(model.transform(regression_df))
        ir = from_spark(str(tmp_path / "glm_log"))
        m = self._load_runtime(ir)
        np.testing.assert_allclose(m.predict(X).ravel(), spark_pred, rtol=1e-4, atol=1e-4)

    def test_kmeans_prediction(self, binary_df, tmp_path):
        model = CL.KMeans(k=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "km_rt"))
        X = self._to_numpy(binary_df)
        spark_pred = self._spark_pred(model.transform(binary_df))
        ir = from_spark(str(tmp_path / "km_rt"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel().astype(int), spark_pred.astype(int))

    def test_gaussian_mixture_prediction(self, binary_df, tmp_path):
        model = CL.GaussianMixture(k=2, seed=42).fit(binary_df)
        model.save(str(tmp_path / "gmm_rt"))
        X = self._to_numpy(binary_df)
        spark_pred = self._spark_pred(model.transform(binary_df))
        ir = from_spark(str(tmp_path / "gmm_rt"))
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel().astype(int), spark_pred.astype(int))

    def test_pipeline_numeric_assembler_lr(self, spark_mixed_df, tmp_path):
        """VectorAssembler from scalar numeric columns + LR: prediction and probability match."""
        from pyspark.ml import Pipeline
        feature_cols = ["size", "weight", "is_premium"]
        fitted = Pipeline(stages=[
            F.VectorAssembler(inputCols=feature_cols, outputCol="features"),
            C.LogisticRegression(featuresCol="features", maxIter=20),
        ]).fit(spark_mixed_df)
        fitted.save(str(tmp_path / "num_asm_lr_rt"))
        X = np.array(
            [[r.size, r.weight, r.is_premium]
             for r in spark_mixed_df.select("size", "weight", "is_premium").collect()],
            dtype=np.float64,
        )
        spark_out = fitted.transform(spark_mixed_df)
        ir = from_spark(str(tmp_path / "num_asm_lr_rt"), feature_names=feature_cols)
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_numeric_assembler_rf(self, spark_mixed_df, tmp_path):
        """VectorAssembler from scalar numeric columns + RF: prediction and probability match."""
        from pyspark.ml import Pipeline
        feature_cols = ["size", "weight", "is_premium"]
        fitted = Pipeline(stages=[
            F.VectorAssembler(inputCols=feature_cols, outputCol="features"),
            C.RandomForestClassifier(featuresCol="features", numTrees=10, maxDepth=3, seed=42),
        ]).fit(spark_mixed_df)
        fitted.save(str(tmp_path / "num_asm_rf_rt"))
        X = np.array(
            [[r.size, r.weight, r.is_premium]
             for r in spark_mixed_df.select("size", "weight", "is_premium").collect()],
            dtype=np.float64,
        )
        spark_out = fitted.transform(spark_mixed_df)
        ir = from_spark(str(tmp_path / "num_asm_rf_rt"), feature_names=feature_cols)
        m = self._load_runtime(ir)
        np.testing.assert_array_equal(m.predict(X).ravel(), self._spark_pred(spark_out))
        np.testing.assert_allclose(m.predict_proba(X), self._spark_proba(spark_out), rtol=1e-4, atol=1e-4)

    def test_pipeline_numeric_assembler_gbt_regression(self, spark_mixed_df, tmp_path):
        """VectorAssembler from scalar numeric columns + GBT regressor: runtime prediction match."""
        from pyspark.ml import Pipeline
        feature_cols = ["size", "weight", "is_premium"]
        fitted = Pipeline(stages=[
            F.VectorAssembler(inputCols=feature_cols, outputCol="features"),
            R.GBTRegressor(featuresCol="features", maxIter=5, maxDepth=2, seed=42),
        ]).fit(spark_mixed_df)
        fitted.save(str(tmp_path / "num_asm_gbt_reg_rt"))
        X = np.array(
            [[r.size, r.weight, r.is_premium]
             for r in spark_mixed_df.select("size", "weight", "is_premium").collect()],
            dtype=np.float64,
        )
        spark_pred = self._spark_pred(fitted.transform(spark_mixed_df))
        ir = from_spark(str(tmp_path / "num_asm_gbt_reg_rt"), feature_names=feature_cols)
        m = self._load_runtime(ir)
        np.testing.assert_allclose(m.predict(X).ravel(), spark_pred, rtol=1e-4, atol=1e-4)
