"""Tests for the LightGBM → OMLE converter."""

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")

import omle
from omle.ir.enums import (
    OutputRole,
    PostTransform,
    TargetKind,
    TreeAggregation,
    TreeNodeKind,
)
from omle_convert.lightgbm import from_lightgbm, from_lightgbm_text


def _tv_float64(tv, model):
    """Resolve TensorValue to flat float64 list regardless of inline/external storage."""
    if tv is None:
        return []
    if tv.tensor is not None:
        t = tv.tensor
        return list(t.float64_data or t.float32_data or [])
    if tv.tensor_ref is not None:
        idx = {e.id: e for e in (model.tensor_entries or [])}
        entry = idx.get(tv.tensor_ref.id)
        if entry and entry.dense:
            return list(entry.dense.float64_data or entry.dense.float32_data or [])
    return []


def _assert_std_io(m, *, task, n_features=6, n_classes=2,
                   input_dtype=None, target_name="y"):
    """Assert standard IO invariants for models with a single 'X' matrix input."""
    if input_dtype is None:
        input_dtype = omle.DataType.FLOAT64
    assert len(m.inputs) == 1
    assert m.inputs[0].name == "X"
    assert m.inputs[0].type.dtype == input_dtype
    assert m.inputs[0].type.shape == [-1, n_features]
    if task == "regression":
        assert len(m.outputs) == 1
        out = m.outputs[0]
        assert out.name == "y_pred"
        assert out.role == OutputRole.PREDICTION
        assert out.type.dtype == omle.DataType.FLOAT64
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
        assert prob.type.dtype == omle.DataType.FLOAT64
        assert prob.type.shape == [-1, n_classes]
        assert m.model_schema.targets[0].kind == (
            TargetKind.BINARY if task == "binary" else TargetKind.MULTICLASS
        )
        assert m.model_schema.targets[0].name == target_name


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_ESTIMATORS = 10


@pytest.fixture(scope="module")
def lgb_regressor(regression_data):
    X, y = regression_data
    model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, max_depth=3,
                               random_state=0, verbose=-1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def lgb_binary(binary_data):
    X, y = binary_data
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                random_state=0, verbose=-1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def lgb_multiclass(multiclass_data):
    X, y = multiclass_data
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                num_class=3, random_state=0, verbose=-1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def lgb_named_features(binary_data):
    import pandas as pd
    X_np, y = binary_data
    cols = [f"feat_{i}" for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=cols)
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=2,
                                random_state=0, verbose=-1)
    model.fit(X, y)
    return model, cols


@pytest.fixture(scope="module")
def lgb_regressor_f64(regression_data_f64):
    X, y = regression_data_f64
    model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, max_depth=3,
                               random_state=0, verbose=-1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def lgb_binary_f64(binary_data_f64):
    X, y = binary_data_f64
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                random_state=0, verbose=-1)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def lgb_multiclass_f64(multiclass_data_f64):
    X, y = multiclass_data_f64
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                num_class=3, random_state=0, verbose=-1)
    model.fit(X, y)
    return model


# ── Regressor ─────────────────────────────────────────────────────────────────

class TestLGBRegressor:
    def test_returns_omle_model(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_metadata(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        # Bare LGBMRegressor: only lightgbm in source_frameworks (not sklearn)
        assert m.metadata.source_frameworks[0].name == "lightgbm"
        assert len(m.metadata.source_frameworks) == 1
        assert m.metadata.format_version == "0.1.0"

    def test_single_output_prediction(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.outputs) == 1
        assert m.outputs[0].role == OutputRole.PREDICTION

    def test_correct_tree_count(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_aggregation_sum(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_no_post_transform(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_no_tree_group(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.tree_group == []

    def test_schema_regression_target(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_json_roundtrip(self, lgb_regressor, tmp_path):
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        p = tmp_path / "reg.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        _assert_std_io(loaded, task="regression")
        assert len(loaded.nodes[0].tree_ensemble.trees) == N_ESTIMATORS


class TestLGBRegressorWithX:
    """Same regression model converted with X — CLI is no-X, runtime is with-X."""

    def test_converts(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_tree_count(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_no_post_transform(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_input_spec_dtype(self, lgb_regressor, regression_data):
        X, _ = regression_data  # float32 — InputSpec reflects actual input dtype
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_feature_dtype(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        # Feature dtype is always FLOAT64 (LightGBM internal precision)
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT64


# ── Binary classifier ─────────────────────────────────────────────────────────

class TestLGBBinary:
    def test_two_outputs(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        roles = {o.role for o in m.outputs}
        assert OutputRole.PREDICTION in roles
        assert OutputRole.PROBABILITY in roles

    def test_sigmoid_post_transform(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_correct_tree_count(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_schema_binary_target(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        assert m.model_schema.targets[0].kind == TargetKind.BINARY

    def test_class_labels_inferred(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        labels = [s.value for s in m.model_schema.targets[0].class_labels]
        assert len(labels) == 2


class TestLGBBinaryWithX:
    """Same binary model converted with X provided."""

    def test_converts(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = from_lightgbm(lgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_sigmoid(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = from_lightgbm(lgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_tree_count(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = from_lightgbm(lgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_input_spec_dtype(self, lgb_binary, binary_data):
        X, _ = binary_data  # float32
        m = from_lightgbm(lgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32


# ── Multiclass classifier ─────────────────────────────────────────────────────

class TestLGBMulticlass:
    def test_total_trees(self, lgb_multiclass):
        m = from_lightgbm(lgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3

    def test_softmax_post_transform(self, lgb_multiclass):
        m = from_lightgbm(lgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_tree_group_length(self, lgb_multiclass):
        m = from_lightgbm(lgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        assert len(tg) == N_ESTIMATORS * 3

    def test_tree_group_round_robin(self, lgb_multiclass):
        m = from_lightgbm(lgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        for i, g in enumerate(tg):
            assert g == i % 3

    def test_schema_multiclass_target(self, lgb_multiclass):
        m = from_lightgbm(lgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS


class TestLGBMulticlassWithX:
    """Same multiclass model converted with X provided."""

    def test_converts(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_lightgbm(lgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_softmax(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_lightgbm(lgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_tree_count(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_lightgbm(lgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3

    def test_input_spec_dtype(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data  # float32
        m = from_lightgbm(lgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32


# ── Feature names ─────────────────────────────────────────────────────────────

class TestLGBFeatureNames:
    def test_named_features_in_schema(self, lgb_named_features):
        model, cols = lgb_named_features
        m = from_lightgbm(model)
        _assert_std_io(m, task="binary")
        expanded = [n for f in m.model_schema.features for n in f.expand_names()]
        assert expanded == cols

    def test_explicit_feature_names(self, lgb_binary):
        names = [f"col_{i}" for i in range(6)]
        m = from_lightgbm(lgb_binary, feature_names=names)
        _assert_std_io(m, task="binary")
        expanded = [n for f in m.model_schema.features for n in f.expand_names()]
        assert expanded == names

    def test_custom_target_name(self, lgb_regressor):
        m = from_lightgbm(lgb_regressor, target_name="price")
        _assert_std_io(m, task="regression", target_name="price")
        assert m.model_schema.targets[0].name == "price"

    def test_custom_class_labels(self, lgb_binary):
        m = from_lightgbm(lgb_binary, class_labels=["no", "yes"])
        _assert_std_io(m, task="binary")
        labels = [s.value for s in m.model_schema.targets[0].class_labels]
        assert labels == ["no", "yes"]


# ── Tree structure ────────────────────────────────────────────────────────────

class TestLGBTreeStructure:
    def test_tree_flat_arrays_consistent(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n = tree.num_nodes
            assert len(tree.node_kind) == n
            assert len(tree.split_feature) == n
            assert len(_tv_float64(tree.split_threshold, m)) == n
            assert len(tree.split_op) == n
            assert len(tree.children_offset) == n
            assert len(tree.children_count) == n
            assert len(tree.default_child) == n
            assert len(_tv_float64(tree.leaf_value, m)) == n

    def test_leaf_nodes_have_zero_children(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.LEAF:
                    assert tree.children_count[i] == 0

    def test_branch_nodes_have_two_children(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.BRANCH:
                    assert tree.children_count[i] == 2

    def test_children_index_within_range(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n = tree.num_nodes
            for c in tree.children_index:
                assert 0 <= c < n

    def test_leaves_sum_equals_n_internal_plus_one(self, lgb_binary):
        m = from_lightgbm(lgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n_leaves = sum(1 for k in tree.node_kind if k == TreeNodeKind.LEAF)
            n_branch = sum(1 for k in tree.node_kind if k == TreeNodeKind.BRANCH)
            # Full binary tree: n_leaves == n_branch + 1
            assert n_leaves == n_branch + 1

    def test_native_booster(self, lgb_binary):
        booster = lgb_binary.booster_
        m = from_lightgbm(booster, feature_names=[f"f{i}" for i in range(6)])
        _assert_std_io(m, task="binary")
        assert isinstance(m, omle.OMLEModel)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS


# ── Mixed-dtype DataFrame ─────────────────────────────────────────────────────

def _make_lgb_mixed_pipeline(estimator):
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
def lgb_mixed_regression(mixed_df):
    df, y_reg, _, _ = mixed_df
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMRegressor(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_reg)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="module")
def lgb_mixed_binary(mixed_df):
    df, _, y_bin, _ = mixed_df
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_bin)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="module")
def lgb_mixed_multiclass(mixed_df):
    df, _, _, y_mc = mixed_df
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMClassifier(n_estimators=10, max_depth=3, num_class=4,
                            random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_mc)
    return pipeline, df.columns.tolist()


class TestLGBMixedData:
    def test_regression_converts(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        pipeline.predict(df)
        assert isinstance(m, omle.OMLEModel)
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_source_frameworks_pipeline(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        names = [f.name for f in m.metadata.source_frameworks]
        roles = [f.role for f in m.metadata.source_frameworks]
        assert "lightgbm" in names
        assert "sklearn" in names
        sk_idx = names.index("sklearn")
        assert roles[sk_idx] == "preprocessing"

    def test_regression_per_column_inputs(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        # InputSpecs reflect original df column order
        assert [s.name for s in m.inputs] == cols
        # TreeEnsemble receives the encoded matrix directly (no TakeSlots)
        ensemble_node = m.nodes[-1]
        ensemble_input_names = {ni.name for ni in ensemble_node.inputs}
        assert "ord_encoded" in ensemble_input_names
        assert "gender_encoded" not in ensemble_input_names
        assert "segment_encoded" not in ensemble_input_names

    def test_regression_column_types(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        feat_by_name = {f.name: f for f in m.model_schema.features}
        assert feat_by_name["age"].type.dtype     == omle.DataType.INT64
        assert feat_by_name["income"].type.dtype  == omle.DataType.FLOAT64
        assert feat_by_name["active"].measure_level == omle.MeasureLevel.FLAG
        assert feat_by_name["gender"].type.dtype  == omle.DataType.STRING
        assert feat_by_name["gender"].measure_level == omle.MeasureLevel.NOMINAL
        assert feat_by_name["segment"].type.dtype == omle.DataType.STRING

    def test_regression_feature_source_is_col_name(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        for feat in m.model_schema.features:
            assert feat.source == feat.name

    def test_regression_scalar_shape(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        for spec in m.inputs:
            assert spec.type.shape == [-1], f"{spec.name} shape != [-1]"
        for feat in m.model_schema.features:
            assert feat.type.shape == [-1], f"{feat.name} shape != [-1]"

    def test_regression_string_categories_in_domain(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        feat_by_name = {f.name: f for f in m.model_schema.features}
        gender_cats = [dv.value.string_value for dv in feat_by_name["gender"].domain.discrete.values]
        assert set(gender_cats) == {"M", "F"}
        assert feat_by_name["gender"].domain.discrete.ordered is False
        segment_cats = [dv.value.string_value for dv in feat_by_name["segment"].domain.discrete.values]
        assert set(segment_cats) == {"low", "mid", "high", "premium"}
        assert feat_by_name["age"].domain is None
        assert feat_by_name["income"].domain is None

    def test_regression_label_encode_nodes(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
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
        assert all_cats[:2] == ["F", "M"]
        assert set(all_cats[2:]) == {"high", "low", "mid", "premium"}
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

    def test_binary_converts(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        assert m.model_schema.targets[0].kind == TargetKind.BINARY
        assert [s.name for s in m.inputs] == cols

    def test_binary_sigmoid(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_multiclass_converts(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS
        assert [s.name for s in m.inputs] == cols

    def test_multiclass_softmax(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_multiclass_tree_group(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        tg = m.nodes[-1].tree_ensemble.tree_group
        assert len(tg) == 10 * 4
        for i, g in enumerate(tg):
            assert g == i % 4

    def test_int_target_type(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        target = m.model_schema.targets[0]
        assert target.type.dtype == omle.DataType.INT64
        assert target.type.shape == [-1]
        assert all(s.int_value is not None for s in target.class_labels)

    def test_int_multiclass_target_type(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        target = m.model_schema.targets[0]
        assert target.type.dtype == omle.DataType.INT64
        assert target.type.shape == [-1]


@pytest.fixture(scope="module")
def lgb_mixed_binary_str(mixed_df):
    df, _, y_bin, _ = mixed_df
    y_str = np.where(y_bin == 0, "no", "yes")
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_str)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="module")
def lgb_mixed_multiclass_str(mixed_df):
    df, _, _, _ = mixed_df
    y_str = df["segment"].to_numpy()
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_str)
    return pipeline, df.columns.tolist()


class TestLGBMixedDataStringTarget:
    def test_binary_str_target_type(self, lgb_mixed_binary_str, mixed_df):
        pipeline, cols = lgb_mixed_binary_str
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        target = m.model_schema.targets[0]
        assert target.kind == TargetKind.BINARY
        assert target.type.dtype == omle.DataType.STRING
        assert target.type.shape == [-1]
        labels = {s.string_value for s in target.class_labels}
        assert labels == {"no", "yes"}

    def test_multiclass_str_target_type(self, lgb_mixed_multiclass_str, mixed_df):
        pipeline, cols = lgb_mixed_multiclass_str
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        target = m.model_schema.targets[0]
        assert target.kind == TargetKind.MULTICLASS
        assert target.type.dtype == omle.DataType.STRING
        assert target.type.shape == [-1]
        labels = {s.string_value for s in target.class_labels}
        assert labels == {"low", "mid", "high", "premium"}

    def test_binary_str_sigmoid(self, lgb_mixed_binary_str, mixed_df):
        pipeline, cols = lgb_mixed_binary_str
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_multiclass_str_softmax(self, lgb_mixed_multiclass_str, mixed_df):
        pipeline, cols = lgb_mixed_multiclass_str
        df, _, _, _ = mixed_df
        m = from_lightgbm(pipeline, X=df[cols])
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SOFTMAX


class TestLGBMixedDataNoX:
    """Same pipeline scenarios as TestLGBMixedData but without X — CLI / model-file-only path."""

    def test_regression_converts(self, lgb_mixed_regression):
        pipeline, _ = lgb_mixed_regression
        m = from_lightgbm(pipeline)
        assert isinstance(m, omle.OMLEModel)
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_regression_tree_count(self, lgb_mixed_regression):
        pipeline, _ = lgb_mixed_regression
        m = from_lightgbm(pipeline)
        assert len(m.nodes[-1].tree_ensemble.trees) == N_ESTIMATORS

    def test_regression_ordinal_encoder_nodes(self, lgb_mixed_regression):
        pipeline, _ = lgb_mixed_regression
        m = from_lightgbm(pipeline)
        oe_nodes = [n for n in m.nodes if n.op == "OrdinalEncoder"]
        assert len(oe_nodes) == 1
        assert len(oe_nodes[0].inputs) == 2
        assert len(oe_nodes[0].outputs) == 1
        assert [n for n in m.nodes if n.op == "TakeSlots"] == []

    def test_regression_inputs_from_ct(self, lgb_mixed_regression):
        pipeline, _ = lgb_mixed_regression
        m = from_lightgbm(pipeline)
        # Without X, per-column InputSpecs are inferred from the ColumnTransformer:
        # string-categorical columns get STRING, numeric passthrough columns get FLOAT64.
        input_map = {inp.name: inp for inp in m.inputs}
        assert "gender" in input_map and "segment" in input_map
        assert input_map["gender"].type.dtype == omle.DataType.STRING
        assert input_map["segment"].type.dtype == omle.DataType.STRING
        for inp in m.inputs:
            if inp.name not in ("gender", "segment"):
                assert inp.type.dtype == omle.DataType.FLOAT64

    def test_binary_converts(self, lgb_mixed_binary):
        pipeline, _ = lgb_mixed_binary
        m = from_lightgbm(pipeline)
        assert m.model_schema.targets[0].kind == TargetKind.BINARY
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_multiclass_converts(self, lgb_mixed_multiclass):
        pipeline, _ = lgb_mixed_multiclass
        m = from_lightgbm(pipeline)
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS
        assert m.nodes[-1].tree_ensemble.post_transform == PostTransform.SOFTMAX


# ── from_lightgbm_text ────────────────────────────────────────────────────────

class TestLGBFromText:
    def test_from_text_regressor(self, lgb_regressor, tmp_path):
        p = tmp_path / "reg.txt"
        lgb_regressor.booster_.save_model(str(p))
        m = from_lightgbm_text(p)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_from_text_binary(self, lgb_binary, tmp_path):
        p = tmp_path / "clf.txt"
        lgb_binary.booster_.save_model(str(p))
        m = from_lightgbm_text(p)
        _assert_std_io(m, task="binary")
        assert isinstance(m, omle.OMLEModel)
        from omle.ir.enums import PostTransform
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_from_text_multiclass(self, lgb_multiclass, tmp_path):
        p = tmp_path / "mc.txt"
        lgb_multiclass.booster_.save_model(str(p))
        m = from_lightgbm_text(p)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3

    def test_from_text_kwargs_forwarded(self, lgb_regressor, tmp_path):
        p = tmp_path / "reg.txt"
        lgb_regressor.booster_.save_model(str(p))
        m = from_lightgbm_text(p, target_name="price", model_name="my_model")
        _assert_std_io(m, task="regression", target_name="price")
        assert m.model_schema.targets[0].name == "price"
        assert m.metadata.name == "my_model"


# ── Auxiliary data (verification / warmup / sample inputs) ───────────────────

class TestLGBAuxiliaryData:
    def test_no_auxiliary_by_default(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.verification is None
        assert m.warmup is None
        assert m.sample_inputs is None
        # tensor_entries may hold body-param tensors when OMLE_INLINE_TENSOR_LIMIT is low.

    def test_tensor_entries_populated(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.tensor_entries) > 0

    def test_verification_populated(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        assert len(m.verification.cases) == 1
        assert m.verification.tolerance is not None

    def test_warmup_populated(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup is not None
        assert len(m.warmup.cases) == 1
        assert m.warmup.cases[0].repeat == 3

    def test_sample_inputs_populated(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.sample_inputs is not None
        assert len(m.sample_inputs.cases) == 3

    def test_n_verify_zero_skips_verification(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X, n_verify=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is None
        assert m.warmup is not None

    def test_n_warmup_zero_skips_warmup(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X, n_warmup=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup is None
        assert m.sample_inputs is not None

    def test_n_sample_zero_skips_sample_inputs(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X, n_sample=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.sample_inputs is None

    def test_custom_n_sample(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X, n_sample=3)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.sample_inputs.cases) == 3

    def test_verify_inputs_reference_tensor_entries(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = from_lightgbm(lgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        entry_ids = {te.id for te in m.tensor_entries}
        case = m.verification.cases[0]
        assert all(r.id in entry_ids for r in case.inputs)
        assert all(r.id in entry_ids for r in case.expected_outputs)

    def test_regression_verify_output_name(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_pred" in out_names

    def test_binary_verify_output_name(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = from_lightgbm(lgb_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_multiclass_verify_output_name(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_lightgbm(lgb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_warmup_repeat_param(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X, n_warmup_repeat=7)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup.cases[0].repeat == 7

    def test_tolerance_values(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = from_lightgbm(lgb_regressor, X=X, verify_atol=1e-3, verify_rtol=2e-3)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        tol = m.verification.tolerance
        assert abs(tol.atol.float_value - 1e-3) < 1e-10
        assert abs(tol.rtol.float_value - 2e-3) < 1e-10

    def test_json_roundtrip_preserves_auxiliary(self, lgb_binary, binary_data, tmp_path):
        X, _ = binary_data
        m = from_lightgbm(lgb_binary, X=X, n_verify=2, n_warmup=4, n_sample=2)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        p = tmp_path / "with_aux.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        _assert_std_io(loaded, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert loaded.verification is not None
        assert loaded.warmup is not None
        assert loaded.sample_inputs is not None
        assert len(loaded.sample_inputs.cases) == 2
        assert len(loaded.tensor_entries) == len(m.tensor_entries)


# ── Float64 input data ────────────────────────────────────────────────────────

class TestLGBFloat64:
    def test_regression_converts(self, lgb_regressor_f64):
        m = from_lightgbm(lgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_regression_tree_count(self, lgb_regressor_f64):
        m = from_lightgbm(lgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_regression_aggregation(self, lgb_regressor_f64):
        m = from_lightgbm(lgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_regression_no_post_transform(self, lgb_regressor_f64):
        m = from_lightgbm(lgb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_binary_converts(self, lgb_binary_f64):
        m = from_lightgbm(lgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert isinstance(m, omle.OMLEModel)

    def test_binary_sigmoid(self, lgb_binary_f64):
        m = from_lightgbm(lgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_binary_tree_count(self, lgb_binary_f64):
        m = from_lightgbm(lgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_multiclass_converts(self, lgb_multiclass_f64):
        m = from_lightgbm(lgb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert isinstance(m, omle.OMLEModel)

    def test_multiclass_softmax(self, lgb_multiclass_f64):
        m = from_lightgbm(lgb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_multiclass_tree_group(self, lgb_multiclass_f64):
        m = from_lightgbm(lgb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        for i, g in enumerate(tg):
            assert g == i % 3

    def test_n_features(self, lgb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = from_lightgbm(lgb_binary_f64)
        _assert_std_io(m, task="binary")
        assert m.inputs[0].type.shape[1] == X.shape[1]


# ── Runtime numerical verification ────────────────────────────────────────────

omr = pytest.importorskip("omle_runtime", reason="omle_runtime not installed")


def _load_runtime(ir_model) -> "omr.Model":
    from omle.proto.convert import ir_to_proto
    data = ir_to_proto(ir_model).SerializeToString()
    return omr.load_bytes(data)


class TestLGBRuntimeRegressor:
    def test_loads_without_error(self, lgb_regressor):
        m = _load_runtime(from_lightgbm(lgb_regressor))
        assert m is not None

    def test_n_features_in(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_lightgbm(lgb_regressor))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_matches_native(self, lgb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_lightgbm(lgb_regressor))
        omle_pred = m.predict(X)
        native_pred = lgb_regressor.predict(X)
        np.testing.assert_allclose(omle_pred, native_pred, rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_runtime_consistent(self, lgb_regressor, regression_data, tmp_path):
        X, _ = regression_data
        ir = from_lightgbm(lgb_regressor)
        _assert_std_io(ir, task="regression")
        p = tmp_path / "reg.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestLGBRuntimeBinary:
    def test_loads_without_error(self, lgb_binary):
        m = _load_runtime(from_lightgbm(lgb_binary))
        assert m is not None

    def test_n_features_in(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_lightgbm(lgb_binary))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_proba_matches_native(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_lightgbm(lgb_binary))
        omle_proba = m.predict_proba(X)
        native_proba = lgb_binary.predict_proba(X)
        np.testing.assert_allclose(omle_proba, native_proba, rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, lgb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_lightgbm(lgb_binary))
        assert np.array_equal(m.predict(X), lgb_binary.predict(X))

    def test_json_roundtrip_runtime_consistent(self, lgb_binary, binary_data, tmp_path):
        X, _ = binary_data
        ir = from_lightgbm(lgb_binary)
        _assert_std_io(ir, task="binary")
        p = tmp_path / "bin.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestLGBRuntimeMulticlass:
    def test_loads_without_error(self, lgb_multiclass):
        m = _load_runtime(from_lightgbm(lgb_multiclass))
        assert m is not None

    def test_n_features_in(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_lightgbm(lgb_multiclass))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_proba_matches_native(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_lightgbm(lgb_multiclass))
        omle_proba = m.predict_proba(X)
        native_proba = lgb_multiclass.predict_proba(X)
        np.testing.assert_allclose(omle_proba, native_proba, rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_lightgbm(lgb_multiclass))
        native_class = lgb_multiclass.predict(X)
        assert np.array_equal(m.predict(X), native_class)

    def test_json_roundtrip_runtime_consistent(self, lgb_multiclass, multiclass_data, tmp_path):
        X, _ = multiclass_data
        ir = from_lightgbm(lgb_multiclass)
        _assert_std_io(ir, task="multiclass", n_classes=3)
        p = tmp_path / "mc.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict_proba(X)
        r2 = _load_runtime(ir2).predict_proba(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ── Mixed-pipeline runtime numerical verification ─────────────────────────────
class TestLGBRuntimeMixedRegression:

    def test_loads_without_error(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        assert m is not None

    def test_n_features_in(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        assert m.n_features_in_ == len(cols)

    def test_predict_matches_native(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        omle_pred = m.predict(df[cols])
        native_pred = pipeline.predict(df[cols])
        np.testing.assert_allclose(omle_pred, native_pred, rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_runtime_consistent(self, lgb_mixed_regression, mixed_df, tmp_path):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        ir = from_lightgbm(pipeline, X=df[cols])
        p = tmp_path / "mixed_reg.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(df[cols])
        r2 = _load_runtime(ir2).predict(df[cols])
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestLGBRuntimeMixedBinary:

    def test_loads_without_error(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, _, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        assert m is not None

    def test_n_features_in(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, _, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        assert m.n_features_in_ == len(cols)

    def test_predict_proba_matches_native(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, y_bin, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        np.testing.assert_allclose(m.predict_proba(df[cols]), pipeline.predict_proba(df[cols]), rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, y_bin, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        assert np.array_equal(m.predict(df[cols]), pipeline.predict(df[cols]))

    def test_json_roundtrip_runtime_consistent(self, lgb_mixed_binary, mixed_df, tmp_path):
        pipeline, cols = lgb_mixed_binary
        df, _, _, _ = mixed_df
        ir = from_lightgbm(pipeline, X=df[cols])
        p = tmp_path / "mixed_bin.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(df[cols])
        r2 = _load_runtime(ir2).predict(df[cols])
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestLGBRuntimeMixedMulticlass:

    def test_loads_without_error(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        assert m is not None

    def test_n_features_in(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        assert m.n_features_in_ == len(cols)

    def test_predict_proba_matches_native(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, y_mc = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        omle_proba = m.predict_proba(df[cols])
        native_proba = pipeline.predict_proba(df[cols])
        np.testing.assert_allclose(omle_proba, native_proba, rtol=1e-4, atol=1e-4)

    def test_predict_class_matches_native(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, y_mc = mixed_df
        m = _load_runtime(from_lightgbm(pipeline, X=df[cols]))
        native_class = pipeline.predict(df[cols])
        assert np.array_equal(m.predict(df[cols]), native_class)

    def test_json_roundtrip_runtime_consistent(self, lgb_mixed_multiclass, mixed_df, tmp_path):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        ir = from_lightgbm(pipeline, X=df[cols])
        p = tmp_path / "mixed_mc.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict_proba(df[cols])
        r2 = _load_runtime(ir2).predict_proba(df[cols])
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ── Float64 runtime numerical verification ────────────────────────────────────

class TestLGBRuntimeFloat64:
    def test_regression_predict_matches_native(self, lgb_regressor_f64, regression_data_f64):
        X, _ = regression_data_f64
        m = _load_runtime(from_lightgbm(lgb_regressor_f64))
        np.testing.assert_allclose(m.predict(X), lgb_regressor_f64.predict(X), rtol=1e-4, atol=1e-4)

    def test_binary_predict_proba_matches_native(self, lgb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = _load_runtime(from_lightgbm(lgb_binary_f64))
        np.testing.assert_allclose(
            m.predict_proba(X), lgb_binary_f64.predict_proba(X),
            rtol=1e-4, atol=1e-4,
        )

    def test_multiclass_predict_proba_matches_native(self, lgb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        m = _load_runtime(from_lightgbm(lgb_multiclass_f64))
        np.testing.assert_allclose(
            m.predict_proba(X), lgb_multiclass_f64.predict_proba(X),
            rtol=1e-4, atol=1e-4,
        )


# ── CategoricalDtype string column (direct, no Pipeline) ──────────────────────

@pytest.fixture(scope="module")
def lgb_mixed_cat(cat_mixed_df):
    df, _, y_binary, _ = cat_mixed_df
    model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    model.fit(df, y_binary)
    return model


class TestLGBCatStringDataFrame:
    """LightGBM trained directly on cat_mixed_df (mixed_df with string cols as CategoricalDtype).

    Raw object/string columns are rejected; pandas CategoricalDtype is the supported path.
    The converter emits a LabelEncoder node per categorical string column, mapping string
    categories to integer codes in pandas sorted-category order to match LightGBM's internal
    code representation.
    """

    def test_raw_string_column_raises(self, cat_mixed_df):
        df, _, y_binary, _ = cat_mixed_df
        df_raw = df.copy()
        df_raw["gender"] = df_raw["gender"].astype(str)  # object dtype
        model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
        with pytest.raises(Exception):
            model.fit(df_raw, y_binary)

    def test_converts(self, lgb_mixed_cat, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
        assert m is not None

    def test_categorical_input_specs_are_string(self, lgb_mixed_cat, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["gender"].type.dtype  == omle.DataType.STRING
        assert input_by_name["segment"].type.dtype == omle.DataType.STRING

    def test_numeric_input_spec_dtypes(self, lgb_mixed_cat, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["income"].type.dtype == omle.DataType.FLOAT64
        assert input_by_name["stock"].type.dtype  == omle.DataType.FLOAT32

    def test_categorical_features_are_string_nominal(self, lgb_mixed_cat, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        assert feat_by_name["gender"].type.dtype    == omle.DataType.STRING
        assert feat_by_name["gender"].measure_level  == omle.MeasureLevel.NOMINAL
        assert feat_by_name["segment"].type.dtype   == omle.DataType.STRING
        assert feat_by_name["segment"].measure_level == omle.MeasureLevel.NOMINAL

    def test_categorical_domain_values(self, lgb_mixed_cat, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        gender_cats = sorted(dv.value.string_value
                             for dv in feat_by_name["gender"].domain.discrete.values)
        assert gender_cats == ["F", "M"]
        segment_cats = sorted(dv.value.string_value
                              for dv in feat_by_name["segment"].domain.discrete.values)
        assert segment_cats == ["high", "low", "mid", "premium"]

    def test_float_feature_dtype_is_float64(self, lgb_mixed_cat, cat_mixed_df):
        # float64 column → Feature.type is FLOAT64 (LightGBM internal precision)
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        assert feat_by_name["income"].type.dtype == omle.DataType.FLOAT64

    def test_label_encoder_nodes_present(self, lgb_mixed_cat, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
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

    def test_ensemble_uses_encoded_columns(self, lgb_mixed_cat, cat_mixed_df):
        df, _, _, _ = cat_mixed_df
        m = from_lightgbm(lgb_mixed_cat, X=df)
        ensemble_node = m.nodes[-1]
        input_names = {ni.name for ni in ensemble_node.inputs}
        assert "label_encoded" in input_names
        assert "gender"  not in input_names
        assert "segment" not in input_names


class TestLGBCatStringDataFrameNoX:
    """Same model as TestLGBCatStringDataFrame, converted without X (CLI path).

    LightGBM stores pandas_categorical in dump_model() and categorical feature indices
    in the model text. The converter detects this and emits LabelEncoder nodes with the
    exact category labels, plus per-column InputSpecs with STRING type for categorical cols.
    """

    def test_converts(self, lgb_mixed_cat):
        m = from_lightgbm(lgb_mixed_cat)
        assert m is not None

    def test_per_column_inputs_with_detection(self, lgb_mixed_cat):
        m = from_lightgbm(lgb_mixed_cat)
        # pandas_categorical detection → per-column InputSpecs; categorical columns are STRING
        input_by_name = {s.name: s for s in m.inputs}
        assert "gender" in input_by_name
        assert "segment" in input_by_name
        assert input_by_name["gender"].type.dtype  == omle.DataType.STRING
        assert input_by_name["segment"].type.dtype == omle.DataType.STRING

    def test_label_encoder_nodes_present_with_detection(self, lgb_mixed_cat):
        m = from_lightgbm(lgb_mixed_cat)
        # LightGBM stores the exact category labels in pandas_categorical
        le_nodes = [n for n in m.nodes if n.op == "LabelEncoder"]
        assert len(le_nodes) == 1
        le = le_nodes[0]
        le_input_names = {ni.name for ni in le.inputs}
        assert "gender"  in le_input_names
        assert "segment" in le_input_names
        te_map = {te.id: te for te in m.tensor_entries}
        def _str_data(attr):
            if attr.tensor_ref is not None and attr.tensor_ref.id:
                return list(te_map[attr.tensor_ref.id].dense.string_data)
            return list(attr.tensor.string_data)
        def _int_data(attr):
            if attr.tensor_ref is not None and attr.tensor_ref.id:
                return list(te_map[attr.tensor_ref.id].dense.int64_data)
            return list(attr.tensor.int64_data)
        labels_attr  = next(a for a in le.attributes if a.name == "labels")
        offsets_attr = next(a for a in le.attributes if a.name == "label_offsets")
        all_labels = _str_data(labels_attr)
        offsets    = _int_data(offsets_attr)
        # Recover per-column labels using offsets
        col_order = [ni.name for ni in le.inputs]
        gender_idx  = col_order.index("gender")
        segment_idx = col_order.index("segment")
        gender_labels  = all_labels[offsets[gender_idx] : offsets[gender_idx + 1]]
        segment_labels = all_labels[offsets[segment_idx] : offsets[segment_idx + 1]]
        assert set(gender_labels)  == {"F", "M"}
        assert set(segment_labels) == {"high", "low", "mid", "premium"}

    def test_feature_names_from_model(self, lgb_mixed_cat):
        m = from_lightgbm(lgb_mixed_cat)
        feat_names = {f.name for f in m.model_schema.features}
        assert "gender"  in feat_names
        assert "segment" in feat_names


# ── Mixed numeric DataFrame (direct, no Pipeline) ─────────────────────────────

@pytest.fixture(scope="module")
def lgb_numeric_mixed(numeric_mixed_df):
    df, y = numeric_mixed_df
    model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    model.fit(df, y)
    return model


class TestLGBNumericMixedDataFrame:
    """LightGBM trained directly on a mixed-dtype numeric DataFrame (no Pipeline).

    InputSpec.type.dtype reflects the actual column dtype.
    Feature.type.dtype reflects LightGBM's internal float64 for float columns;
    non-float columns (int, bool) keep their own dtype.
    """

    def test_converts(self, lgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_lightgbm(lgb_numeric_mixed, X=df)
        assert m is not None

    def test_per_column_input_specs(self, lgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_lightgbm(lgb_numeric_mixed, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert set(input_by_name) == {"a_int", "b_float32", "c_float64", "d_bool"}

    def test_input_spec_dtypes_match_columns(self, lgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_lightgbm(lgb_numeric_mixed, X=df)
        input_by_name = {s.name: s for s in m.inputs}
        assert input_by_name["a_int"].type.dtype     == omle.DataType.INT64
        assert input_by_name["b_float32"].type.dtype == omle.DataType.FLOAT32
        assert input_by_name["c_float64"].type.dtype == omle.DataType.FLOAT64
        assert input_by_name["d_bool"].type.dtype    == omle.DataType.BOOL

    def test_feature_dtypes_reflect_lgb_precision(self, lgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_lightgbm(lgb_numeric_mixed, X=df)
        feat_by_name = {f.name: f for f in m.model_schema.features}
        # non-float columns: Feature dtype = InputSpec dtype (no cast)
        assert feat_by_name["a_int"].type.dtype   == omle.DataType.INT64
        assert feat_by_name["d_bool"].type.dtype  == omle.DataType.BOOL
        # float columns: Feature dtype = FLOAT64 regardless of input precision
        assert feat_by_name["b_float32"].type.dtype == omle.DataType.FLOAT64
        assert feat_by_name["c_float64"].type.dtype == omle.DataType.FLOAT64

    def test_no_label_encode_nodes(self, lgb_numeric_mixed, numeric_mixed_df):
        df, _ = numeric_mixed_df
        m = from_lightgbm(lgb_numeric_mixed, X=df)
        label_encode_nodes = [n for n in m.nodes if n.op == "LabelEncoder"]
        assert label_encode_nodes == []


class TestLGBNumericMixedDataFrameNoX:
    """Same model as TestLGBNumericMixedDataFrame, converted without X (CLI path)."""

    def test_converts(self, lgb_numeric_mixed):
        m = from_lightgbm(lgb_numeric_mixed)
        assert m is not None

    def test_single_matrix_input(self, lgb_numeric_mixed):
        m = from_lightgbm(lgb_numeric_mixed)
        # Without X, no per-column InputSpecs — single matrix input at FLOAT64
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64

    def test_feature_names_from_model(self, lgb_numeric_mixed):
        m = from_lightgbm(lgb_numeric_mixed)
        # Feature names are preserved in model_schema (from booster's feature names)
        feat_names = {f.name for f in m.model_schema.features}
        assert feat_names == {"a_int", "b_float32", "c_float64", "d_bool"}

    def test_no_label_encode_nodes(self, lgb_numeric_mixed):
        m = from_lightgbm(lgb_numeric_mixed)
        assert [n for n in m.nodes if n.op == "LabelEncoder"] == []


# ── Homogeneous DataFrame (all columns same dtype) ────────────────────────────

@pytest.fixture(scope="module")
def lgb_homogeneous_f32(regression_data):
    import pandas as pd
    X, y = regression_data  # float32
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    model.fit(df, y)
    return model, df


@pytest.fixture(scope="module")
def lgb_homogeneous_f64(regression_data_f64):
    import pandas as pd
    X, y = regression_data_f64  # float64
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, random_state=0, verbose=-1)
    model.fit(df, y)
    return model, df


class TestLGBHomogeneousDataFrame:
    """When all DataFrame columns share the same dtype, a single 'X' matrix InputSpec is used."""

    def test_converts_f32(self, lgb_homogeneous_f32):
        model, df = lgb_homogeneous_f32
        assert from_lightgbm(model, X=df) is not None

    def test_converts_f64(self, lgb_homogeneous_f64):
        model, df = lgb_homogeneous_f64
        assert from_lightgbm(model, X=df) is not None

    def test_single_input_spec_f32(self, lgb_homogeneous_f32):
        model, df = lgb_homogeneous_f32
        m = from_lightgbm(model, X=df)
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert m.inputs[0].type.shape == [-1, len(df.columns)]

    def test_single_input_spec_f64(self, lgb_homogeneous_f64):
        model, df = lgb_homogeneous_f64
        m = from_lightgbm(model, X=df)
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, len(df.columns)]

    def test_features_name_range_f32(self, lgb_homogeneous_f32):
        model, df = lgb_homogeneous_f32
        m = from_lightgbm(model, X=df)
        assert len(m.model_schema.features) == 1
        feat = m.model_schema.features[0]
        assert feat.range is not None
        assert feat.range.start == 0
        assert feat.range.end == len(df.columns)
        assert feat.source == "X"
        assert feat.index == 0

    def test_features_name_range_f64(self, lgb_homogeneous_f64):
        model, df = lgb_homogeneous_f64
        m = from_lightgbm(model, X=df)
        assert len(m.model_schema.features) == 1
        feat = m.model_schema.features[0]
        assert feat.range is not None
        assert feat.range.start == 0
        assert feat.range.end == len(df.columns)
        assert feat.source == "X"
        assert feat.index == 0

    def test_feature_dtype_is_lgb_internal_f32(self, lgb_homogeneous_f32):
        model, df = lgb_homogeneous_f32
        m = from_lightgbm(model, X=df)
        # LightGBM uses float64 internally regardless of input dtype
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT64

    def test_feature_dtype_is_lgb_internal_f64(self, lgb_homogeneous_f64):
        model, df = lgb_homogeneous_f64
        m = from_lightgbm(model, X=df)
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT64

    def test_estimator_node_input_is_X(self, lgb_homogeneous_f32):
        model, df = lgb_homogeneous_f32
        m = from_lightgbm(model, X=df)
        assert len(m.nodes) == 1
        assert m.nodes[0].inputs[0].name.value == "X"


# ── Numpy input dtype ─────────────────────────────────────────────────────────
#
# LightGBM always uses float64 internally regardless of the input numpy dtype.
# The two dtypes tracked in the schema are:
#   InputSpec.type.dtype  — actual numpy array dtype (what the caller provides)
#   Feature.type.dtype    — LightGBM's internal float64 precision

class TestLGBNumpyInputDtype:
    def test_float32_input_spec_dtype(self, lgb_regressor, regression_data):
        # float32 numpy → InputSpec should be FLOAT32
        X, _ = regression_data  # float32
        m = from_lightgbm(lgb_regressor, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_float32_feature_dtype_is_float64(self, lgb_regressor, regression_data):
        # LightGBM internal precision is always float64
        X, _ = regression_data  # float32
        m = from_lightgbm(lgb_regressor, X=X)
        feat = m.model_schema.features[0]
        assert feat.type.dtype == omle.DataType.FLOAT64

    def test_float64_input_spec_dtype(self, lgb_regressor_f64, regression_data_f64):
        # float64 numpy → InputSpec should reflect actual input dtype (FLOAT64)
        X, _ = regression_data_f64  # float64
        m = from_lightgbm(lgb_regressor_f64, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64

    def test_float64_feature_dtype_is_float64(self, lgb_regressor_f64, regression_data_f64):
        # float64 input and float64 internal — both agree
        X, _ = regression_data_f64  # float64
        m = from_lightgbm(lgb_regressor_f64, X=X)
        feat = m.model_schema.features[0]
        assert feat.type.dtype == omle.DataType.FLOAT64

    def test_no_X_uses_feature_dtype(self, lgb_regressor):
        # No training data → InputSpec and Feature both use framework internal (FLOAT64)
        m = from_lightgbm(lgb_regressor)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT64

    def test_binary_float32_input_spec(self, lgb_binary, binary_data):
        X, _ = binary_data  # float32
        m = from_lightgbm(lgb_binary, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT64

    def test_multiclass_float64_input_spec(self, lgb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        m = from_lightgbm(lgb_multiclass_f64, X=X)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT64


# ── LGBMRanker ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lgb_ranker(regression_data):
    X, y = regression_data
    y_rank = (y - y.min()).astype(int) % 5
    groups = np.array([X.shape[0] // 2, X.shape[0] - X.shape[0] // 2])
    model = lgb.LGBMRanker(n_estimators=N_ESTIMATORS, max_depth=3,
                            random_state=0, verbose=-1)
    model.fit(X, y_rank, group=groups)
    return model


class TestLGBRanker:
    def test_returns_omle_model(self, lgb_ranker):
        m = from_lightgbm(lgb_ranker)
        assert isinstance(m, omle.OMLEModel)

    def test_task_type_regression(self, lgb_ranker):
        from omle.ir.enums import TaskType
        m = from_lightgbm(lgb_ranker)
        assert m.nodes[-1].tree_ensemble.task_type == TaskType.REGRESSION

    def test_n_trees(self, lgb_ranker):
        m = from_lightgbm(lgb_ranker)
        assert len(m.nodes[-1].tree_ensemble.trees) == N_ESTIMATORS

    def test_source_framework_is_lightgbm(self, lgb_ranker):
        m = from_lightgbm(lgb_ranker)
        assert m.metadata.source_frameworks[0].name == "lightgbm"

    def test_single_output(self, lgb_ranker):
        m = from_lightgbm(lgb_ranker)
        assert len(m.outputs) == 1
        assert m.outputs[0].name == "y_pred"

    def test_node_name(self, lgb_ranker):
        m = from_lightgbm(lgb_ranker)
        assert m.nodes[-1].name == "lgbm_ranker"


class TestLGBRankerRuntime:
    def test_loads_without_error(self, lgb_ranker):
        m = _load_runtime(from_lightgbm(lgb_ranker))
        assert m is not None

    def test_predict_matches_native(self, lgb_ranker, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_lightgbm(lgb_ranker))
        np.testing.assert_allclose(m.predict(X), lgb_ranker.predict(X), rtol=1e-4, atol=1e-4)

    def test_predict_dtype(self, lgb_ranker, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_lightgbm(lgb_ranker))
        assert m.predict(X).dtype == np.float64
