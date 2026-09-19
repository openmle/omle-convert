"""Tests for the XGBoost → OMLE converter."""

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

import omle
from omle.ir.enums import (
    OutputRole,
    PostTransform,
    TargetKind,
    TreeAggregation,
    TreeNodeKind,
    TreeSplitOp,
)
from omle_convert.xgboost import from_xgboost, from_xgboost_json


def _tv_float(tv, model):
    """Resolve TensorValue to flat number list regardless of inline/external storage."""
    if tv is None:
        return []
    if tv.tensor is not None:
        t = tv.tensor
        return list(t.float32_data or t.float64_data or [])
    if tv.tensor_ref is not None:
        idx = {e.id: e for e in (model.tensor_entries or [])}
        entry = idx.get(tv.tensor_ref.id)
        if entry and entry.dense:
            return list(entry.dense.float32_data or entry.dense.float64_data or [])
    return []


def _assert_std_io(m, *, task, n_features=6, n_classes=2,
                   input_dtype=None, target_name="y"):
    """Assert standard IO invariants for models with a single 'X' matrix input."""
    if input_dtype is None:
        input_dtype = omle.DataType.FLOAT32
    assert len(m.inputs) == 1
    assert m.inputs[0].name == "X"
    assert m.inputs[0].type.dtype == input_dtype
    assert m.inputs[0].type.shape == [-1, n_features]
    if task == "regression":
        assert len(m.outputs) == 1
        out = m.outputs[0]
        assert out.name == "y_pred"
        assert out.role == OutputRole.PREDICTION
        assert out.type.dtype == omle.DataType.FLOAT32
        assert out.type.shape == [-1]
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION
        assert m.model_schema.targets[0].name == target_name
    elif task in ("binary", "multiclass"):
        pred = next(o for o in m.outputs if o.name == "y_pred")
        prob = next(o for o in m.outputs if o.name == "y_prob")
        assert pred.role == OutputRole.PREDICTION
        assert pred.type.dtype == omle.DataType.INT64
        assert pred.type.shape == [-1]
        assert prob.role == OutputRole.PROBABILITY
        assert prob.type.dtype == omle.DataType.FLOAT32
        assert prob.type.shape == [-1, n_classes]
        assert m.model_schema.targets[0].kind == (
            TargetKind.BINARY if task == "binary" else TargetKind.MULTICLASS
        )
        assert m.model_schema.targets[0].name == target_name


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_ESTIMATORS = 10


@pytest.fixture(scope="module")
def xgb_regressor(regression_data):
    X, y = regression_data
    model = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=3,
                              random_state=0, verbosity=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def xgb_binary(binary_data):
    X, y = binary_data
    model = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                               random_state=0, verbosity=0, eval_metric="logloss")
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def xgb_multiclass(multiclass_data):
    X, y = multiclass_data
    model = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                               num_class=3, random_state=0, verbosity=0,
                               eval_metric="mlogloss")
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def xgb_named_features(binary_data):
    import pandas as pd
    X_np, y = binary_data
    cols = [f"feat_{i}" for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=cols)
    model = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=2,
                               random_state=0, verbosity=0, eval_metric="logloss")
    model.fit(X, y)
    return model, cols


@pytest.fixture(scope="module")
def xgb_regressor_f64(regression_data_f64):
    X, y = regression_data_f64
    model = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=3,
                              random_state=0, verbosity=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def xgb_binary_f64(binary_data_f64):
    X, y = binary_data_f64
    model = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                               random_state=0, verbosity=0, eval_metric="logloss")
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def xgb_multiclass_f64(multiclass_data_f64):
    X, y = multiclass_data_f64
    model = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                               num_class=3, random_state=0, verbosity=0,
                               eval_metric="mlogloss")
    model.fit(X, y)
    return model


# ── Regressor ─────────────────────────────────────────────────────────────────

class TestXGBRegressor:
    def test_returns_omle_model(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_metadata(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        # Bare XGBRegressor: only xgboost in source_frameworks (not sklearn)
        assert m.metadata.source_frameworks[0].name == "xgboost"
        assert len(m.metadata.source_frameworks) == 1
        assert m.metadata.format_version == "0.1.0"

    def test_single_output_prediction(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.outputs) == 1
        assert m.outputs[0].role == OutputRole.PREDICTION

    def test_correct_tree_count(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_aggregation_sum(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_no_post_transform(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_no_tree_group(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.tree_group == []

    def test_schema_regression_target(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_output_dtype_float32(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT32

    def test_target_dtype_float32(self, xgb_regressor):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.model_schema.targets[0].type.dtype == omle.DataType.FLOAT32

    def test_json_roundtrip(self, xgb_regressor, tmp_path):
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        p = tmp_path / "reg.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        _assert_std_io(loaded, task="regression")
        assert len(loaded.nodes[0].tree_ensemble.trees) == N_ESTIMATORS


class TestXGBRegressorWithX:
    """Same regression model but converted with X — CLI is no-X, runtime is with-X."""

    def test_converts(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_tree_count(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_no_post_transform(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_input_spec_dtype(self, xgb_regressor, regression_data):
        X, _ = regression_data  # float32
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_feature_dtype(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32


# ── Binary classifier ─────────────────────────────────────────────────────────

class TestXGBBinary:
    def test_two_outputs(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        roles = {o.role for o in m.outputs}
        assert OutputRole.PREDICTION in roles
        assert OutputRole.PROBABILITY in roles

    def test_sigmoid_post_transform(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_correct_tree_count(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_schema_binary_target(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        assert m.model_schema.targets[0].kind == TargetKind.BINARY

    def test_class_labels_inferred(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        labels = [s.value for s in m.model_schema.targets[0].class_labels]
        assert len(labels) == 2

    def test_y_prob_dtype_float32(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        prob_out = next(o for o in m.outputs if o.role == OutputRole.PROBABILITY)
        assert prob_out.type.dtype == omle.DataType.FLOAT32


class TestXGBBinaryWithX:
    """Same binary model converted with X provided."""

    def test_converts(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = from_xgboost(xgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_sigmoid(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = from_xgboost(xgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_tree_count(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = from_xgboost(xgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_input_spec_dtype(self, xgb_binary, binary_data):
        X, _ = binary_data  # float32
        m = from_xgboost(xgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32


# ── Multiclass classifier ──────────────────────────────────────────────────────

class TestXGBMulticlass:
    def test_total_trees(self, xgb_multiclass):
        m = from_xgboost(xgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        # n_estimators * n_classes trees
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3

    def test_softmax_post_transform(self, xgb_multiclass):
        m = from_xgboost(xgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_tree_group_length(self, xgb_multiclass):
        m = from_xgboost(xgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        assert len(tg) == N_ESTIMATORS * 3

    def test_tree_group_round_robin(self, xgb_multiclass):
        m = from_xgboost(xgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        for i, g in enumerate(tg):
            assert g == i % 3

    def test_schema_multiclass_target(self, xgb_multiclass):
        m = from_xgboost(xgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS


class TestXGBMulticlassWithX:
    """Same multiclass model converted with X provided."""

    def test_converts(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_xgboost(xgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_softmax(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_xgboost(xgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_tree_count(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_xgboost(xgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3

    def test_input_spec_dtype(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data  # float32
        m = from_xgboost(xgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32


# ── Feature names ─────────────────────────────────────────────────────────────

class TestXGBFeatureNames:
    def test_named_features_in_schema(self, xgb_named_features):
        model, cols = xgb_named_features
        m = from_xgboost(model)
        _assert_std_io(m, task="binary")
        expanded = [n for f in m.model_schema.features for n in f.expand_names()]
        assert expanded == cols

    def test_explicit_feature_names(self, xgb_binary):
        names = [f"col_{i}" for i in range(6)]
        m = from_xgboost(xgb_binary, feature_names=names)
        _assert_std_io(m, task="binary")
        expanded = [n for f in m.model_schema.features for n in f.expand_names()]
        assert expanded == names

    def test_custom_target_name(self, xgb_regressor):
        m = from_xgboost(xgb_regressor, target_name="price")
        _assert_std_io(m, task="regression", target_name="price")
        assert m.model_schema.targets[0].name == "price"

    def test_custom_class_labels(self, xgb_binary):
        m = from_xgboost(xgb_binary, class_labels=["no", "yes"])
        _assert_std_io(m, task="binary")
        labels = [s.value for s in m.model_schema.targets[0].class_labels]
        assert labels == ["no", "yes"]


# ── Tree structure ────────────────────────────────────────────────────────────

class TestXGBTreeStructure:
    def test_tree_flat_arrays_consistent(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n = tree.num_nodes
            assert len(tree.node_kind) == n
            assert len(tree.split_feature) == n
            assert len(_tv_float(tree.split_threshold, m)) == n
            assert len(tree.split_op) == n
            assert len(tree.children_offset) == n
            assert len(tree.children_count) == n
            assert len(tree.default_child) == n
            assert len(_tv_float(tree.leaf_value, m)) == n

    def test_leaf_nodes_have_zero_children(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.LEAF:
                    assert tree.children_count[i] == 0

    def test_branch_nodes_have_two_children(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.BRANCH:
                    assert tree.children_count[i] == 2

    def test_branch_split_op_less_than(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.BRANCH:
                    assert tree.split_op[i] == TreeSplitOp.LESS_THAN

    def test_children_index_within_range(self, xgb_binary):
        m = from_xgboost(xgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n = tree.num_nodes
            for c in tree.children_index:
                assert 0 <= c < n

    def test_native_booster(self, xgb_binary):
        booster = xgb_binary.get_booster()
        m = from_xgboost(booster, feature_names=[f"f{i}" for i in range(6)])
        _assert_std_io(m, task="binary")
        assert isinstance(m, omle.OMLEModel)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS


# ── Mixed-dtype DataFrame ─────────────────────────────────────────────────────

def _make_xgb_mixed_pipeline(estimator):
    import inspect

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    ct_kwargs = {"remainder": "passthrough"}
    if "force_int_remainder_cols" in inspect.signature(ColumnTransformer).parameters:
        ct_kwargs["force_int_remainder_cols"] = False
    ct = ColumnTransformer([("ord", OrdinalEncoder(), ["gender", "segment"])], **ct_kwargs)
    return Pipeline([("preprocessor", ct), ("model", estimator)])


@pytest.fixture(scope="module")
def xgb_mixed_regression(mixed_df):
    df, y_reg, _, _ = mixed_df
    pipeline = _make_xgb_mixed_pipeline(
        xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=0, verbosity=0)
    )
    pipeline.fit(df, y_reg)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="module")
def xgb_mixed_binary(mixed_df):
    df, _, y_bin, _ = mixed_df
    pipeline = _make_xgb_mixed_pipeline(
        xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0, verbosity=0,
                           eval_metric="logloss")
    )
    pipeline.fit(df, y_bin)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="module")
def xgb_mixed_multiclass(mixed_df):
    df, _, _, y_mc = mixed_df
    pipeline = _make_xgb_mixed_pipeline(
        xgb.XGBClassifier(n_estimators=10, max_depth=3, num_class=4,
                           random_state=0, verbosity=0, eval_metric="mlogloss")
    )
    pipeline.fit(df, y_mc)
    return pipeline, df.columns.tolist()


class TestXGBMixedData:
    def test_regression_converts(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        assert isinstance(m, omle.OMLEModel)
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_source_frameworks_pipeline(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        names = [f.name for f in m.metadata.source_frameworks]
        roles = [f.role for f in m.metadata.source_frameworks]
        assert "xgboost" in names
        assert "sklearn" in names
        sk_idx = names.index("sklearn")
        assert roles[sk_idx] == "preprocessing"

    def test_regression_per_column_inputs(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        # InputSpecs reflect original df column order
        assert [s.name for s in m.inputs] == cols
        # TreeEnsemble receives the encoded matrix directly (no TakeSlots)
        ensemble_node = m.nodes[-1]
        ensemble_input_names = {ni.name for ni in ensemble_node.inputs}
        assert "ord_encoded" in ensemble_input_names
        assert "gender_encoded" not in ensemble_input_names
        assert "segment_encoded" not in ensemble_input_names

    def test_regression_column_types(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        # InputSpec reflects actual training-data column dtype
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["income"].type.dtype == omle.DataType.FLOAT64
        # Feature.type reflects XGBoost's internal float32 precision for float columns;
        # non-float columns keep their own dtype
        feat_by_name = {f.name: f for f in m.model_schema.features}
        assert feat_by_name["age"].type.dtype    == omle.DataType.INT64
        assert feat_by_name["income"].type.dtype == omle.DataType.FLOAT32
        assert feat_by_name["active"].measure_level == omle.MeasureLevel.FLAG
        assert feat_by_name["gender"].type.dtype  == omle.DataType.STRING
        assert feat_by_name["gender"].measure_level == omle.MeasureLevel.NOMINAL
        assert feat_by_name["segment"].type.dtype == omle.DataType.STRING

    def test_regression_feature_source_is_col_name(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        for feat in m.model_schema.features:
            assert feat.source == feat.name

    def test_regression_scalar_shape(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        for spec in m.inputs:
            assert spec.type.shape == [-1], f"{spec.name} shape != [-1]"
        for feat in m.model_schema.features:
            assert feat.type.shape == [-1], f"{feat.name} shape != [-1]"

    def test_regression_string_categories_in_domain(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        feat_by_name = {f.name: f for f in m.model_schema.features}
        gender_cats = [dv.value.string_value for dv in feat_by_name["gender"].domain.discrete.values]
        assert set(gender_cats) == {"M", "F"}
        assert feat_by_name["gender"].domain.discrete.ordered is False
        segment_cats = [dv.value.string_value for dv in feat_by_name["segment"].domain.discrete.values]
        assert set(segment_cats) == {"low", "mid", "high", "premium"}
        assert feat_by_name["age"].domain is None
        assert feat_by_name["income"].domain is None

    def test_regression_label_encode_nodes(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        # OrdinalEncoder + TreeEnsemble = 2 nodes (no TakeSlots)
        assert len(m.nodes) == 2
        oe_nodes = [n for n in m.nodes if n.op == "OrdinalEncoder"]
        assert len(oe_nodes) == 1
        oe = oe_nodes[0]
        assert oe.domain == "omle.feature"
        assert oe.name == "ord_ordinal_encoder"
        input_names = [ni.name for ni in oe.inputs]
        assert input_names == ["gender", "segment"]
        te_map = {te.id: te for te in m.tensor_entries}
        # single flat categories tensor + boundary offsets (may be inline or tensor_ref)
        def _str_data(attr):
            if attr.tensor_ref is not None and attr.tensor_ref.id:
                return list(te_map[attr.tensor_ref.id].dense.string_data)
            return list(attr.tensor.string_data)
        def _int_data(attr):
            if attr.tensor_ref is not None and attr.tensor_ref.id:
                return list(te_map[attr.tensor_ref.id].dense.int64_data)
            return list(attr.tensor.int64_data)
        cats_attr = next(a for a in oe.attributes if a.name == "categories")
        all_cats = _str_data(cats_attr)
        assert all_cats[:2] == ["F", "M"]                               # gender (alphabetical)
        assert set(all_cats[2:]) == {"high", "low", "mid", "premium"}   # segment
        off_attr = next(a for a in oe.attributes if a.name == "category_offsets")
        offsets = _int_data(off_attr)
        assert offsets == [0, 2, 6]
        # single matrix output (INT64, shape [-1, 2])
        assert len(oe.outputs) == 1
        assert oe.outputs[0].name == "ord_encoded"
        assert oe.outputs[0].type.dtype == omle.DataType.INT64
        # no TakeSlots — TreeEnsemble consumes the encoded matrix directly
        assert [n for n in m.nodes if n.op == "TakeSlots"] == []
        ensemble_input_names = {ni.name for ni in m.nodes[-1].inputs}
        assert "ord_encoded" in ensemble_input_names
        assert m.nodes[-1].op == "TreeEnsemble"

    def test_regression_source_frameworks(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        fw_by_name = {sf.name: sf for sf in m.metadata.source_frameworks}
        assert "xgboost" in fw_by_name
        assert "sklearn" in fw_by_name
        assert fw_by_name["sklearn"].role == "preprocessing"

    def test_binary_converts(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        assert m.model_schema.targets[0].kind == TargetKind.BINARY
        assert [s.name for s in m.inputs] == cols

    def test_binary_sigmoid(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_multiclass_converts(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS
        assert [s.name for s in m.inputs] == cols

    def test_multiclass_tree_group(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        tg = m.nodes[-1].tree_ensemble.tree_group
        assert len(tg) == 10 * 4
        for i, g in enumerate(tg):
            assert g == i % 4

    def test_int_target_type(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        target = m.model_schema.targets[0]
        assert target.type.dtype == omle.DataType.INT64
        assert target.type.shape == [-1]
        assert all(s.int_value is not None for s in target.class_labels)

    def test_int_multiclass_target_type(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols])
        target = m.model_schema.targets[0]
        assert target.type.dtype == omle.DataType.INT64
        assert target.type.shape == [-1]


class TestXGBMixedDataStringTarget:
    # XGBoost 2.x requires integer training labels; string class_labels are
    # supplied explicitly to the converter to annotate the model schema.

    def test_binary_str_target_type(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols], class_labels=["no", "yes"])
        target = m.model_schema.targets[0]
        assert target.kind == TargetKind.BINARY
        assert target.type.dtype == omle.DataType.STRING
        assert target.type.shape == [-1]
        assert {s.string_value for s in target.class_labels} == {"no", "yes"}

    def test_multiclass_str_target_type(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_xgboost(pipeline, X=df[cols],
                         class_labels=["low", "mid", "high", "premium"])
        target = m.model_schema.targets[0]
        assert target.kind == TargetKind.MULTICLASS
        assert target.type.dtype == omle.DataType.STRING
        assert target.type.shape == [-1]
        assert {s.string_value for s in target.class_labels} == {
            "low", "mid", "high", "premium"
        }


class TestXGBMixedDataNoX:
    """Same pipeline scenarios as TestXGBMixedData but without X — CLI / model-file-only path."""

    def test_regression_converts(self, xgb_mixed_regression):
        pipeline, _ = xgb_mixed_regression
        m = from_xgboost(pipeline)
        assert isinstance(m, omle.OMLEModel)
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_regression_tree_count(self, xgb_mixed_regression):
        pipeline, _ = xgb_mixed_regression
        m = from_xgboost(pipeline)
        assert len(m.nodes[-1].tree_ensemble.trees) == N_ESTIMATORS

    def test_regression_ordinal_encoder_nodes(self, xgb_mixed_regression):
        pipeline, _ = xgb_mixed_regression
        m = from_xgboost(pipeline)
        oe_nodes = [n for n in m.nodes if n.op == "OrdinalEncoder"]
        assert len(oe_nodes) == 1
        assert len(oe_nodes[0].inputs) == 2
        assert len(oe_nodes[0].outputs) == 1
        assert [n for n in m.nodes if n.op == "TakeSlots"] == []

    def test_regression_inputs_from_ct(self, xgb_mixed_regression):
        pipeline, _ = xgb_mixed_regression
        m = from_xgboost(pipeline)
        # Without X, per-column InputSpecs are inferred from the ColumnTransformer:
        # string-categorical columns get STRING, numeric passthrough columns get FLOAT32.
        input_map = {inp.name: inp for inp in m.inputs}
        assert "gender" in input_map and "segment" in input_map
        assert input_map["gender"].type.dtype == omle.DataType.STRING
        assert input_map["segment"].type.dtype == omle.DataType.STRING
        for inp in m.inputs:
            if inp.name not in ("gender", "segment"):
                assert inp.type.dtype == omle.DataType.FLOAT32

    def test_binary_converts(self, xgb_mixed_binary):
        pipeline, _ = xgb_mixed_binary
        m = from_xgboost(pipeline)
        assert m.model_schema.targets[0].kind == TargetKind.BINARY
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_multiclass_converts(self, xgb_mixed_multiclass):
        pipeline, _ = xgb_mixed_multiclass
        m = from_xgboost(pipeline)
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SOFTMAX


# ── Auxiliary data (verification / warmup / sample inputs) ───────────────────

class TestXGBAuxiliaryData:
    def test_no_auxiliary_by_default(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.verification is None
        assert m.warmup is None
        assert m.sample_inputs is None
        # tensor_entries may hold body-param tensors when OMLE_INLINE_TENSOR_LIMIT is low.

    def test_tensor_entries_populated(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.tensor_entries) > 0

    def test_verification_populated(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        assert len(m.verification.cases) == 1
        assert m.verification.tolerance is not None
        assert m.verification.tolerance.atol is not None
        assert m.verification.tolerance.rtol is not None

    def test_warmup_populated(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup is not None
        assert len(m.warmup.cases) == 1
        assert m.warmup.cases[0].repeat == 3

    def test_sample_inputs_populated(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.sample_inputs is not None
        assert len(m.sample_inputs.cases) == 3

    def test_n_verify_zero_skips_verification(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X, n_verify=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is None
        assert m.warmup is not None

    def test_n_warmup_zero_skips_warmup(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X, n_warmup=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup is None
        assert m.sample_inputs is not None

    def test_n_sample_zero_skips_sample_inputs(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X, n_sample=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.sample_inputs is None

    def test_custom_n_sample(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X, n_sample=3)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.sample_inputs.cases) == 3

    def test_small_x_test_clamped(self, xgb_regressor, regression_data):
        X, _ = regression_data
        tiny = X[:2]
        m = from_xgboost(xgb_regressor, X=tiny, n_warmup=50)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        # clamped to 2
        assert m.warmup.cases[0].repeat == 3
        warmup_inputs = m.warmup.cases[0].inputs
        entry_ids = {te.id for te in m.tensor_entries}
        assert all(r.id in entry_ids for r in warmup_inputs)

    def test_verify_inputs_reference_tensor_entries(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = from_xgboost(xgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        entry_ids = {te.id for te in m.tensor_entries}
        case = m.verification.cases[0]
        assert all(r.id in entry_ids for r in case.inputs)
        assert all(r.id in entry_ids for r in case.expected_outputs)

    def test_verify_expected_outputs_present(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = from_xgboost(xgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_verify_regression_output_name(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_pred" in out_names

    def test_verify_multiclass(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_xgboost(xgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_tensor_entry_data_populated(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X, n_verify=1, n_warmup=0, n_sample=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        for te in m.tensor_entries:
            assert te.dense is not None
            has_data = (te.dense.float32_data or te.dense.float64_data
                        or te.dense.int64_data or te.dense.string_data)
            assert has_data

    def test_json_roundtrip_preserves_auxiliary(self, xgb_binary, binary_data, tmp_path):
        X, _ = binary_data
        m = from_xgboost(xgb_binary, X=X, n_verify=2, n_warmup=4, n_sample=2)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        p = tmp_path / "with_aux.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        _assert_std_io(loaded, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert loaded.verification is not None
        assert len(loaded.verification.cases) == 1
        assert loaded.warmup is not None
        assert loaded.sample_inputs is not None
        assert len(loaded.sample_inputs.cases) == 2
        assert len(loaded.tensor_entries) == len(m.tensor_entries)

    def test_warmup_repeat_param(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X, n_warmup_repeat=7)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup.cases[0].repeat == 7

    def test_tolerance_values(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = from_xgboost(xgb_regressor, X=X, verify_atol=1e-3, verify_rtol=2e-3)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        tol = m.verification.tolerance
        assert abs(tol.atol.float_value - 1e-3) < 1e-10
        assert abs(tol.rtol.float_value - 2e-3) < 1e-10


# ── from_xgboost_json ─────────────────────────────────────────────────────────

class TestXGBFromJson:
    def test_from_json_regressor(self, xgb_regressor, tmp_path):
        p = tmp_path / "reg.json"
        xgb_regressor.get_booster().save_model(str(p))
        m = from_xgboost_json(p)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_from_json_binary(self, xgb_binary, tmp_path):
        p = tmp_path / "clf.json"
        xgb_binary.get_booster().save_model(str(p))
        m = from_xgboost_json(p)
        _assert_std_io(m, task="binary")
        assert isinstance(m, omle.OMLEModel)
        from omle.ir.enums import PostTransform
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_from_json_multiclass(self, xgb_multiclass, tmp_path):
        p = tmp_path / "mc.json"
        xgb_multiclass.get_booster().save_model(str(p))
        m = from_xgboost_json(p)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3

    def test_from_json_kwargs_forwarded(self, xgb_regressor, tmp_path):
        p = tmp_path / "reg.json"
        xgb_regressor.get_booster().save_model(str(p))
        m = from_xgboost_json(p, target_name="price", model_name="my_model")
        _assert_std_io(m, task="regression", target_name="price")
        assert m.model_schema.targets[0].name == "price"
        assert m.metadata.name == "my_model"


# ── Float64 input data ────────────────────────────────────────────────────────

class TestXGBFloat64:
    def test_regression_converts(self, xgb_regressor_f64):
        m = from_xgboost(xgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_regression_tree_count(self, xgb_regressor_f64):
        m = from_xgboost(xgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_regression_aggregation(self, xgb_regressor_f64):
        m = from_xgboost(xgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_regression_no_post_transform(self, xgb_regressor_f64):
        m = from_xgboost(xgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_binary_converts(self, xgb_binary_f64):
        m = from_xgboost(xgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert isinstance(m, omle.OMLEModel)

    def test_binary_sigmoid(self, xgb_binary_f64):
        m = from_xgboost(xgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_binary_tree_count(self, xgb_binary_f64):
        m = from_xgboost(xgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_multiclass_converts(self, xgb_multiclass_f64):
        m = from_xgboost(xgb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert isinstance(m, omle.OMLEModel)

    def test_multiclass_softmax(self, xgb_multiclass_f64):
        m = from_xgboost(xgb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_multiclass_tree_group(self, xgb_multiclass_f64):
        m = from_xgboost(xgb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        for i, g in enumerate(tg):
            assert g == i % 3

    def test_n_features(self, xgb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = from_xgboost(xgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert m.inputs[0].type.shape[1] == X.shape[1]


# ── omle_runtime numerical verification ─────────────────────────────────────

omr = pytest.importorskip("omle_runtime", reason="omle_runtime not installed")


def _load_runtime(ir_model) -> "omr.Model":
    from omle.proto.convert import ir_to_proto
    data = ir_to_proto(ir_model).SerializeToString()
    return omr.load_bytes(data)


class TestXGBRuntimeRegressor:
    def test_loads_without_error(self, xgb_regressor):
        m = _load_runtime(from_xgboost(xgb_regressor))
        assert m is not None

    def test_n_features_in(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_xgboost(xgb_regressor))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_dtype(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_xgboost(xgb_regressor))
        assert m.predict(X).dtype == np.float32

    def test_predict_matches_native(self, xgb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_xgboost(xgb_regressor))
        omle_pred = m.predict(X)
        native_pred = xgb_regressor.predict(X)
        np.testing.assert_allclose(omle_pred, native_pred, rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_runtime_consistent(self, xgb_regressor, regression_data, tmp_path):
        X, _ = regression_data
        ir = from_xgboost(xgb_regressor)
        _assert_std_io(ir, task="regression")
        p = tmp_path / "reg.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestXGBRuntimeBinary:
    def test_loads_without_error(self, xgb_binary):
        m = _load_runtime(from_xgboost(xgb_binary))
        assert m is not None

    def test_n_features_in(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_xgboost(xgb_binary))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_proba_dtype(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_xgboost(xgb_binary))
        assert m.predict_proba(X).dtype == np.float32

    def test_predict_proba_matches_native(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_xgboost(xgb_binary))
        np.testing.assert_allclose(m.predict_proba(X), xgb_binary.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, xgb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_xgboost(xgb_binary))
        native_class = xgb_binary.predict(X)
        assert np.array_equal(m.predict(X), native_class)

    def test_json_roundtrip_runtime_consistent(self, xgb_binary, binary_data, tmp_path):
        X, _ = binary_data
        ir = from_xgboost(xgb_binary)
        _assert_std_io(ir, task="binary")
        p = tmp_path / "bin.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestXGBRuntimeMulticlass:
    def test_loads_without_error(self, xgb_multiclass):
        m = _load_runtime(from_xgboost(xgb_multiclass))
        assert m is not None

    def test_n_features_in(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_xgboost(xgb_multiclass))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_proba_dtype(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_xgboost(xgb_multiclass))
        assert m.predict_proba(X).dtype == np.float32

    def test_predict_proba_matches_native(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_xgboost(xgb_multiclass))
        omle_proba = m.predict_proba(X)
        native_proba = xgb_multiclass.predict_proba(X)
        np.testing.assert_allclose(omle_proba, native_proba, rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_xgboost(xgb_multiclass))
        native_class = xgb_multiclass.predict(X)
        assert np.array_equal(m.predict(X), native_class)

    def test_json_roundtrip_runtime_consistent(self, xgb_multiclass, multiclass_data, tmp_path):
        X, _ = multiclass_data
        ir = from_xgboost(xgb_multiclass)
        _assert_std_io(ir, task="multiclass", n_classes=3)
        p = tmp_path / "mc.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict_proba(X)
        r2 = _load_runtime(ir2).predict_proba(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ── Mixed-pipeline runtime numerical verification ─────────────────────────────
class TestXGBRuntimeMixedRegression:
    def test_loads_without_error(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        assert m is not None

    def test_n_features_in(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        assert m.n_features_in_ == len(cols)

    def test_predict_matches_native(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, y_reg, _, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        omle_pred = m.predict(df[cols])
        native_pred = pipeline.predict(df[cols])
        np.testing.assert_allclose(omle_pred, native_pred, rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_runtime_consistent(self, xgb_mixed_regression, mixed_df, tmp_path):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        ir = from_xgboost(pipeline, X=df[cols])
        p = tmp_path / "mixed_reg.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(df[cols])
        r2 = _load_runtime(ir2).predict(df[cols])
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestXGBRuntimeMixedBinary:
    def test_loads_without_error(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        assert m is not None

    def test_n_features_in(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        assert m.n_features_in_ == len(cols)

    def test_predict_proba_matches_native(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, y_bin, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        np.testing.assert_allclose(m.predict_proba(df[cols]), pipeline.predict_proba(df[cols]), rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, y_bin, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        native_class = pipeline.predict(df[cols])
        assert np.array_equal(m.predict(df[cols]), native_class)

    def test_json_roundtrip_runtime_consistent(self, xgb_mixed_binary, mixed_df, tmp_path):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        ir = from_xgboost(pipeline, X=df[cols])
        p = tmp_path / "mixed_bin.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(df[cols])
        r2 = _load_runtime(ir2).predict(df[cols])
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestXGBRuntimeMixedMulticlass:
    def test_loads_without_error(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        assert m is not None

    def test_n_features_in(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        assert m.n_features_in_ == len(cols)

    def test_predict_proba_matches_native(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, y_mc = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        omle_proba = m.predict_proba(df[cols])
        native_proba = pipeline.predict_proba(df[cols])
        np.testing.assert_allclose(omle_proba, native_proba, rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, y_mc = mixed_df
        m = _load_runtime(from_xgboost(pipeline, X=df[cols]))
        native_class = pipeline.predict(df[cols])
        assert np.array_equal(m.predict(df[cols]), native_class)

    def test_json_roundtrip_runtime_consistent(self, xgb_mixed_multiclass, mixed_df, tmp_path):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        ir = from_xgboost(pipeline, X=df[cols])
        p = tmp_path / "mixed_mc.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict_proba(df[cols])
        r2 = _load_runtime(ir2).predict_proba(df[cols])
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ── Float64 runtime numerical verification ────────────────────────────────────

class TestXGBRuntimeFloat64:
    def test_regression_predict_matches_native(self, xgb_regressor_f64, regression_data_f64):
        X, _ = regression_data_f64
        m = _load_runtime(from_xgboost(xgb_regressor_f64))
        np.testing.assert_allclose(m.predict(X), xgb_regressor_f64.predict(X), rtol=1e-4, atol=1e-4)

    def test_binary_predict_proba_matches_native(self, xgb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = _load_runtime(from_xgboost(xgb_binary_f64))
        np.testing.assert_allclose(
            m.predict_proba(X), xgb_binary_f64.predict_proba(X),
            rtol=1e-4, atol=1e-4,
        )

    def test_multiclass_predict_proba_matches_native(self, xgb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        m = _load_runtime(from_xgboost(xgb_multiclass_f64))
        np.testing.assert_allclose(
            m.predict_proba(X), xgb_multiclass_f64.predict_proba(X),
            rtol=1e-4, atol=1e-4,
        )


# ── CategoricalDtype string column (direct, no Pipeline) ──────────────────────

@pytest.fixture(scope="module")
def xgb_cat_mixed(cat_mixed_df):
    # enable_categorical=True is required for pandas CategoricalDtype columns
    df, _, y_binary, _ = cat_mixed_df
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0,
                               verbosity=0, eval_metric="logloss",
                               enable_categorical=True)
    model.fit(df, y_binary)
    return model


class TestXGBCatStringDataFrame:
    """XGBoost trained directly on cat_mixed_df (mixed_df with string cols as CategoricalDtype).

    Raw object/string columns are rejected; pandas CategoricalDtype + enable_categorical=True
    is the supported path.  The converter emits a LabelEncoder node per categorical string
    column, mapping string categories to integer codes in pandas sorted-category order to
    match XGBoost's internal integer-code representation.
    """

    def test_raw_string_column_raises(self, cat_mixed_df):
        df, _, y_binary, _ = cat_mixed_df
        df_raw = df.copy()
        df_raw["gender"] = df_raw["gender"].astype(str)  # object dtype
        model = xgb.XGBClassifier(n_estimators=5, verbosity=0, eval_metric="logloss")
        with pytest.raises(Exception):
            model.fit(df_raw, y_binary)

    def test_converts(self, xgb_cat_mixed, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        assert m is not None

    def test_categorical_input_specs_are_string(self, xgb_cat_mixed, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["gender"].type.dtype  == omle.DataType.STRING
        assert input_by_name["segment"].type.dtype == omle.DataType.STRING

    def test_numeric_input_spec_dtypes(self, xgb_cat_mixed, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["income"].type.dtype == omle.DataType.FLOAT64
        assert input_by_name["stock"].type.dtype  == omle.DataType.FLOAT32

    def test_categorical_features_are_string_nominal(self, xgb_cat_mixed, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        assert feat_by_name["gender"].type.dtype   == omle.DataType.STRING
        assert feat_by_name["gender"].measure_level == omle.MeasureLevel.NOMINAL
        assert feat_by_name["segment"].type.dtype  == omle.DataType.STRING
        assert feat_by_name["segment"].measure_level == omle.MeasureLevel.NOMINAL

    def test_categorical_domain_values(self, xgb_cat_mixed, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        gender_cats = sorted(dv.value.string_value
                             for dv in feat_by_name["gender"].domain.discrete.values)
        assert gender_cats == ["F", "M"]
        segment_cats = sorted(dv.value.string_value
                              for dv in feat_by_name["segment"].domain.discrete.values)
        assert segment_cats == ["high", "low", "mid", "premium"]

    def test_float_feature_dtype_is_float32(self, xgb_cat_mixed, cat_mixed_df):
        # float64 column → Feature.type is FLOAT32 (XGBoost internal precision)
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        assert feat_by_name["income"].type.dtype == omle.DataType.FLOAT32

    def test_label_encoder_nodes_present(self, xgb_cat_mixed, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        le_nodes = [n for n in m.nodes if n.op == "LabelEncoder"]
        assert len(le_nodes) == 1
        le = le_nodes[0]
        le_input_names = {ni.name for ni in le.inputs}
        assert "gender"  in le_input_names
        assert "segment" in le_input_names
        assert len(le.outputs) == 1
        assert le.outputs[0].name == "label_encoded"
        # no TakeSlots — TreeEnsemble receives the encoded matrix directly
        assert [n for n in m.nodes if n.op == "TakeSlots"] == []

    def test_ensemble_uses_encoded_columns(self, xgb_cat_mixed, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_xgboost(xgb_cat_mixed, X=df)
        ensemble_node = m.nodes[-1]
        input_names = {ni.name for ni in ensemble_node.inputs}
        assert "label_encoded" in input_names
        assert "gender"  not in input_names
        assert "segment" not in input_names


@pytest.fixture(scope="module")
def xgb_cat_gender_target(cat_mixed_df):
    """XGBoost model where the target is correlated with gender/segment → trees split on categoricals."""
    df, _, _, _ = cat_mixed_df
    # target based on categorical column → forces IN_SET splits
    y = (df["gender"].astype(str) == "M").astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=4, random_state=0,
                               verbosity=0, eval_metric="logloss",
                               enable_categorical=True)
    model.fit(df, y)
    return model, df, y


class TestXGBRuntimeCatStringDataFrame:
    """Runtime correctness for XGBoost models with IN_SET categorical splits."""

    def test_has_in_set_splits(self, xgb_cat_gender_target, cat_mixed_df):
        model, df, _ = xgb_cat_gender_target
        m = from_xgboost(model, X=df)
        ensemble_node = m.nodes[-1]
        in_set_count = sum(
            1 for tree in ensemble_node.tree_ensemble.trees
            for op in tree.split_op
            if op.name == "IN_SET"
        )
        assert in_set_count > 0, "Expected IN_SET categorical splits in the trees"

    def test_predict_proba_matches_native(self, xgb_cat_gender_target, cat_mixed_df):
        model, df, _ = xgb_cat_gender_target
        m = _load_runtime(from_xgboost(model, X=df))
        np.testing.assert_allclose(m.predict_proba(df), model.predict_proba(df), rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, xgb_cat_gender_target, cat_mixed_df):
        model, df, _ = xgb_cat_gender_target
        m = _load_runtime(from_xgboost(model, X=df))
        native_class = model.predict(df)
        assert np.array_equal(m.predict(df), native_class)

    def test_json_roundtrip_runtime_consistent(self, xgb_cat_gender_target, cat_mixed_df, tmp_path):
        model, df, _ = xgb_cat_gender_target
        ir = from_xgboost(model, X=df)
        p = tmp_path / "cat_gender.json"
        import omle as _omle
        _omle.save_json(ir, p)
        ir2 = _omle.load_json(p)
        r1 = _load_runtime(ir).predict_proba(df)
        r2 = _load_runtime(ir2).predict_proba(df)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestXGBCatStringDataFrameNoX:
    """Same model as TestXGBCatStringDataFrame, converted without X (CLI path).

    XGBoost stores feature_types ('c' markers for categoricals) but NOT the category
    label strings — so a valid LabelEncoder cannot be emitted.  The converter falls
    back to a single matrix input and omits LabelEncoder nodes.  Pass X to get the
    full per-column schema with working LabelEncoder nodes.
    """

    def test_converts(self, xgb_cat_mixed):
        m = from_xgboost(xgb_cat_mixed)
        assert m is not None

    def test_per_column_inputs(self, xgb_cat_mixed):
        m = from_xgboost(xgb_cat_mixed)
        # feature_types='c' proves a DataFrame was used; emit per-column InputSpecs
        assert len(m.inputs) > 1
        input_names = {s.name for s in m.inputs}
        assert "gender"  in input_names
        assert "segment" in input_names

    def test_categorical_columns_are_int64(self, xgb_cat_mixed):
        m = from_xgboost(xgb_cat_mixed)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["gender"].type.dtype  == omle.DataType.INT64
        assert input_by_name["segment"].type.dtype == omle.DataType.INT64

    def test_numeric_columns_are_float32(self, xgb_cat_mixed):
        m = from_xgboost(xgb_cat_mixed)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["age"].type.dtype == omle.DataType.FLOAT32

    def test_no_label_encoder_nodes_without_x(self, xgb_cat_mixed):
        m = from_xgboost(xgb_cat_mixed)
        # XGBoost does not store category label strings — INT64 inputs, no LabelEncoder
        assert [n for n in m.nodes if n.op == "LabelEncoder"] == []

    def test_feature_names_from_model(self, xgb_cat_mixed):
        m = from_xgboost(xgb_cat_mixed)
        feat_names = {f.name for f in m.model_schema.features}
        assert "gender"  in feat_names
        assert "segment" in feat_names


# ── Mixed numeric DataFrame (direct, no Pipeline) ─────────────────────────────

@pytest.fixture(scope="module")
def xgb_numeric_mixed(numeric_mixed_df):
    df, y = numeric_mixed_df
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0,
                               verbosity=0, eval_metric="logloss")
    model.fit(df, y)
    return model


class TestXGBNumericMixedDataFrame:
    """XGBoost trained directly on a mixed-dtype numeric DataFrame (no Pipeline).

    InputSpec.type.dtype reflects the actual column dtype.
    Feature.type.dtype reflects XGBoost's internal float32 for float columns;
    non-float columns (int, bool) keep their own dtype.
    """

    def test_converts(self, xgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_xgboost(xgb_numeric_mixed, X=df)
        assert m is not None

    def test_per_column_input_specs(self, xgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_xgboost(xgb_numeric_mixed, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert set(input_by_name) == {"a_int", "b_float32", "c_float64", "d_bool"}

    def test_input_spec_dtypes_match_columns(self, xgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_xgboost(xgb_numeric_mixed, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["a_int"].type.dtype    == omle.DataType.INT64
        assert input_by_name["b_float32"].type.dtype == omle.DataType.FLOAT32
        assert input_by_name["c_float64"].type.dtype == omle.DataType.FLOAT64
        assert input_by_name["d_bool"].type.dtype    == omle.DataType.BOOL

    def test_feature_dtypes_reflect_xgb_precision(self, xgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_xgboost(xgb_numeric_mixed, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        # non-float columns: Feature dtype = InputSpec dtype (no cast)
        assert feat_by_name["a_int"].type.dtype    == omle.DataType.INT64
        assert feat_by_name["d_bool"].type.dtype   == omle.DataType.BOOL
        # float columns: Feature dtype = FLOAT32 regardless of input precision
        assert feat_by_name["b_float32"].type.dtype == omle.DataType.FLOAT32
        assert feat_by_name["c_float64"].type.dtype == omle.DataType.FLOAT32

    def test_no_label_encode_nodes(self, xgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_xgboost(xgb_numeric_mixed, X=df)
        label_encode_nodes = [n for n in m.nodes if n.op == "LabelEncoder"]
        assert label_encode_nodes == []


class TestXGBNumericMixedDataFrameNoX:
    """Same model as TestXGBNumericMixedDataFrame, converted without X (CLI path)."""

    def test_converts(self, xgb_numeric_mixed):
        m = from_xgboost(xgb_numeric_mixed)
        assert m is not None

    def test_single_matrix_input(self, xgb_numeric_mixed):
        m = from_xgboost(xgb_numeric_mixed)
        # Without X, no per-column InputSpecs — single matrix input at FLOAT32
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_feature_names_from_model(self, xgb_numeric_mixed):
        m = from_xgboost(xgb_numeric_mixed)
        # Feature names are preserved in model_schema (from booster's feature_names)
        feat_names = {f.name for f in m.model_schema.features}
        assert feat_names == {"a_int", "b_float32", "c_float64", "d_bool"}

    def test_no_label_encode_nodes(self, xgb_numeric_mixed):
        m = from_xgboost(xgb_numeric_mixed)
        assert [n for n in m.nodes if n.op == "LabelEncoder"] == []


# ── Homogeneous DataFrame (all columns same dtype) ────────────────────────────

@pytest.fixture(scope="module")
def xgb_homogeneous_f32(regression_data):
    import pandas as pd
    X, y = regression_data  # float32
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=0, verbosity=0)
    model.fit(df, y)
    return model, df


@pytest.fixture(scope="module")
def xgb_homogeneous_f64(regression_data_f64):
    import pandas as pd
    X, y = regression_data_f64  # float64
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=0, verbosity=0)
    model.fit(df, y)
    return model, df


class TestXGBHomogeneousDataFrame:
    """When all DataFrame columns share the same dtype, a single 'X' matrix InputSpec is used."""

    def test_converts_f32(self, xgb_homogeneous_f32):
        model, df = xgb_homogeneous_f32
        assert from_xgboost(model, X=df) is not None

    def test_converts_f64(self, xgb_homogeneous_f64):
        model, df = xgb_homogeneous_f64
        assert from_xgboost(model, X=df) is not None

    def test_single_input_spec_f32(self, xgb_homogeneous_f32):
        model, df = xgb_homogeneous_f32
        m = from_xgboost(model, X=df)
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert m.inputs[0].type.shape == [-1, len(df.columns)]

    def test_single_input_spec_f64(self, xgb_homogeneous_f64):
        model, df = xgb_homogeneous_f64
        m = from_xgboost(model, X=df)
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, len(df.columns)]

    def test_features_name_range_f32(self, xgb_homogeneous_f32):
        model, df = xgb_homogeneous_f32
        m = from_xgboost(model, X=df)
        assert len(m.model_schema.features) == 1
        feat = m.model_schema.features[0]
        assert feat.range is not None
        assert feat.range.start == 0
        assert feat.range.end == len(df.columns)
        assert feat.source == "X"
        assert feat.index == 0

    def test_features_name_range_f64(self, xgb_homogeneous_f64):
        model, df = xgb_homogeneous_f64
        m = from_xgboost(model, X=df)
        assert len(m.model_schema.features) == 1
        feat = m.model_schema.features[0]
        assert feat.range is not None
        assert feat.range.start == 0
        assert feat.range.end == len(df.columns)
        assert feat.source == "X"
        assert feat.index == 0

    def test_feature_dtype_is_xgb_internal_f32(self, xgb_homogeneous_f32):
        model, df = xgb_homogeneous_f32
        m = from_xgboost(model, X=df)
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32

    def test_feature_dtype_is_xgb_internal_f64(self, xgb_homogeneous_f64):
        model, df = xgb_homogeneous_f64
        m = from_xgboost(model, X=df)
        # XGBoost casts to float32 internally regardless of input dtype
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32

    def test_estimator_node_input_is_X(self, xgb_homogeneous_f32):
        model, df = xgb_homogeneous_f32
        m = from_xgboost(model, X=df)
        # Homogeneous DataFrame: single estimator node receiving X directly
        assert len(m.nodes) == 1
        assert m.nodes[0].inputs[0].name.value == "X"


# ── Numpy input dtype ─────────────────────────────────────────────────────────
#
# XGBoost always casts inputs to float32 internally regardless of the input
# numpy dtype.  The two dtypes tracked in the schema are:
#   InputSpec.type.dtype  — actual numpy array dtype (what the caller provides)
#   Feature.type.dtype    — XGBoost's internal float32 precision

class TestXGBNumpyInputDtype:
    def test_float32_input_spec_dtype(self, xgb_regressor, regression_data):
        # float32 numpy → InputSpec should be FLOAT32
        X, _ = regression_data  # float32
        m = from_xgboost(xgb_regressor, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_float32_feature_dtype_is_float32(self, xgb_regressor, regression_data):
        # XGBoost internal precision is always float32
        X, _ = regression_data  # float32
        m = from_xgboost(xgb_regressor, X=X)
        feat = m.model_schema.features[0]
        assert feat.type.dtype == omle.DataType.FLOAT32

    def test_float64_input_spec_dtype(self, xgb_regressor_f64, regression_data_f64):
        # float64 numpy → InputSpec should reflect actual input dtype (FLOAT64)
        X, _ = regression_data_f64  # float64
        m = from_xgboost(xgb_regressor_f64, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64

    def test_float64_feature_dtype_is_float32(self, xgb_regressor_f64, regression_data_f64):
        # Even with float64 input, XGBoost Feature.type is float32 (internal cast)
        X, _ = regression_data_f64  # float64
        m = from_xgboost(xgb_regressor_f64, X=X)
        feat = m.model_schema.features[0]
        assert feat.type.dtype == omle.DataType.FLOAT32

    def test_no_X_uses_feature_dtype(self, xgb_regressor):
        # No training data → InputSpec and Feature both use framework internal (FLOAT32)
        m = from_xgboost(xgb_regressor)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32

    def test_binary_float64_input_spec(self, xgb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = from_xgboost(xgb_binary_f64, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32

    def test_multiclass_float64_input_spec(self, xgb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        m = from_xgboost(xgb_multiclass_f64, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32


class TestXGBSklearnEquivalence:
    """from_xgboost and to_omle agree on graph shape and interface.

    They are not byte-identical: to_omle routes sklearn-compatible wrappers
    through from_sklearn, which records an extra source_framework entry.
    """

    def test_bare_regressor_nodes_equal(self, xgb_regressor):
        from omle_convert import to_omle
        m1 = from_xgboost(xgb_regressor)
        m2 = to_omle(xgb_regressor)
        assert [n.op for n in m1.nodes] == [n.op for n in m2.nodes]
        assert [n.name for n in m1.nodes] == [n.name for n in m2.nodes]

    def test_bare_regressor_inputs_equal(self, xgb_regressor):
        from omle_convert import to_omle
        m1 = from_xgboost(xgb_regressor)
        m2 = to_omle(xgb_regressor)
        assert [(i.name, i.type.dtype) for i in m1.inputs] == [(i.name, i.type.dtype) for i in m2.inputs]

    def test_bare_classifier_binary_equal(self, xgb_binary):
        from omle_convert import to_omle
        m1 = from_xgboost(xgb_binary)
        m2 = to_omle(xgb_binary)
        assert [n.op for n in m1.nodes] == [n.op for n in m2.nodes]
        assert [(i.name, i.type.dtype) for i in m1.inputs] == [(i.name, i.type.dtype) for i in m2.inputs]


# ── XGBRFClassifier ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def xgb_rf_binary(binary_data):
    X, y = binary_data
    model = xgb.XGBRFClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                  random_state=0, verbosity=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def xgb_rf_multiclass(multiclass_data):
    X, y = multiclass_data
    model = xgb.XGBRFClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                  random_state=0, verbosity=0)
    model.fit(X, y)
    return model


class TestXGBRFClassifier:
    def test_returns_omle_model(self, xgb_rf_binary):
        m = from_xgboost(xgb_rf_binary)
        assert isinstance(m, omle.OMLEModel)

    def test_binary_task_type(self, xgb_rf_binary):
        from omle.ir.enums import TaskType
        m = from_xgboost(xgb_rf_binary)
        assert m.nodes[-1].tree_ensemble.task_type == TaskType.BINARY

    def test_multiclass_task_type(self, xgb_rf_multiclass):
        from omle.ir.enums import TaskType
        m = from_xgboost(xgb_rf_multiclass)
        assert m.nodes[-1].tree_ensemble.task_type == TaskType.MULTICLASS

    def test_binary_n_trees(self, xgb_rf_binary):
        m = from_xgboost(xgb_rf_binary)
        assert len(m.nodes[-1].tree_ensemble.trees) == N_ESTIMATORS

    def test_source_framework_is_xgboost(self, xgb_rf_binary):
        m = from_xgboost(xgb_rf_binary)
        assert m.metadata.source_frameworks[0].name == "xgboost"

    def test_y_prob_dtype_float32(self, xgb_rf_binary):
        m = from_xgboost(xgb_rf_binary)
        prob_out = next(o for o in m.outputs if o.role == OutputRole.PROBABILITY)
        assert prob_out.type.dtype == omle.DataType.FLOAT32

    def test_node_name(self, xgb_rf_binary):
        m = from_xgboost(xgb_rf_binary)
        assert m.nodes[-1].name == "xgbrf_classifier"


class TestXGBRFRuntimeBinary:
    def test_loads_without_error(self, xgb_rf_binary):
        m = _load_runtime(from_xgboost(xgb_rf_binary))
        assert m is not None

    def test_predict_proba_matches_native(self, xgb_rf_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_xgboost(xgb_rf_binary))
        np.testing.assert_allclose(m.predict_proba(X), xgb_rf_binary.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, xgb_rf_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_xgboost(xgb_rf_binary))
        assert np.array_equal(m.predict(X), xgb_rf_binary.predict(X))


class TestXGBRFRuntimeMulticlass:
    def test_loads_without_error(self, xgb_rf_multiclass):
        m = _load_runtime(from_xgboost(xgb_rf_multiclass))
        assert m is not None

    def test_predict_proba_matches_native(self, xgb_rf_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_xgboost(xgb_rf_multiclass))
        np.testing.assert_allclose(m.predict_proba(X), xgb_rf_multiclass.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, xgb_rf_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_xgboost(xgb_rf_multiclass))
        assert np.array_equal(m.predict(X), xgb_rf_multiclass.predict(X))


# ── XGBRFRegressor ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def xgb_rf_regressor(regression_data):
    X, y = regression_data
    model = xgb.XGBRFRegressor(n_estimators=N_ESTIMATORS, max_depth=3,
                                 random_state=0, verbosity=0)
    model.fit(X, y)
    return model


class TestXGBRFRegressor:
    def test_returns_omle_model(self, xgb_rf_regressor):
        m = from_xgboost(xgb_rf_regressor)
        assert isinstance(m, omle.OMLEModel)

    def test_task_type(self, xgb_rf_regressor):
        from omle.ir.enums import TaskType
        m = from_xgboost(xgb_rf_regressor)
        assert m.nodes[-1].tree_ensemble.task_type == TaskType.REGRESSION

    def test_n_trees(self, xgb_rf_regressor):
        m = from_xgboost(xgb_rf_regressor)
        assert len(m.nodes[-1].tree_ensemble.trees) == N_ESTIMATORS

    def test_source_framework_is_xgboost(self, xgb_rf_regressor):
        m = from_xgboost(xgb_rf_regressor)
        assert m.metadata.source_frameworks[0].name == "xgboost"

    def test_output_dtype_float32(self, xgb_rf_regressor):
        m = from_xgboost(xgb_rf_regressor)
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT32

    def test_node_name(self, xgb_rf_regressor):
        m = from_xgboost(xgb_rf_regressor)
        assert m.nodes[-1].name == "xgbrf_regressor"


class TestXGBRFRuntimeRegressor:
    def test_loads_without_error(self, xgb_rf_regressor):
        m = _load_runtime(from_xgboost(xgb_rf_regressor))
        assert m is not None

    def test_predict_matches_native(self, xgb_rf_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_xgboost(xgb_rf_regressor))
        np.testing.assert_allclose(m.predict(X), xgb_rf_regressor.predict(X), rtol=1e-4, atol=1e-4)


# ── XGBRanker ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def xgb_ranker(regression_data):
    X, y = regression_data
    # XGBRanker requires integer labels (relevance scores) and group sizes
    y_rank = (y - y.min()).astype(int) % 5
    groups = np.array([X.shape[0] // 2, X.shape[0] - X.shape[0] // 2])
    model = xgb.XGBRanker(n_estimators=N_ESTIMATORS, max_depth=3,
                           random_state=0, verbosity=0)
    model.fit(X, y_rank, group=groups)
    return model


class TestXGBRanker:
    def test_returns_omle_model(self, xgb_ranker):
        m = from_xgboost(xgb_ranker)
        assert isinstance(m, omle.OMLEModel)

    def test_task_type_regression(self, xgb_ranker):
        from omle.ir.enums import TaskType
        m = from_xgboost(xgb_ranker)
        assert m.nodes[-1].tree_ensemble.task_type == TaskType.REGRESSION

    def test_n_trees(self, xgb_ranker):
        m = from_xgboost(xgb_ranker)
        assert len(m.nodes[-1].tree_ensemble.trees) == N_ESTIMATORS

    def test_source_framework_is_xgboost(self, xgb_ranker):
        m = from_xgboost(xgb_ranker)
        assert m.metadata.source_frameworks[0].name == "xgboost"

    def test_output_dtype_float32(self, xgb_ranker):
        m = from_xgboost(xgb_ranker)
        assert m.outputs[0].type.dtype == omle.DataType.FLOAT32

    def test_node_name(self, xgb_ranker):
        m = from_xgboost(xgb_ranker)
        assert m.nodes[-1].name == "xgb_ranker"

    def test_single_output(self, xgb_ranker):
        m = from_xgboost(xgb_ranker)
        assert len(m.outputs) == 1
        assert m.outputs[0].name == "y_pred"


class TestXGBRankerRuntime:
    def test_loads_without_error(self, xgb_ranker):
        m = _load_runtime(from_xgboost(xgb_ranker))
        assert m is not None

    def test_predict_matches_native(self, xgb_ranker, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_xgboost(xgb_ranker))
        np.testing.assert_allclose(m.predict(X), xgb_ranker.predict(X), rtol=1e-4, atol=1e-4)

    def test_predict_dtype(self, xgb_ranker, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_xgboost(xgb_ranker))
        assert m.predict(X).dtype == np.float32
