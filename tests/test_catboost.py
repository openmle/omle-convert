"""Tests for the CatBoost → OMLE converter."""

import os
import tempfile

import numpy as np
import pytest

cb = pytest.importorskip("catboost", reason="catboost not installed")

import omle
from omle.ir.enums import (
    OutputRole,
    PostTransform,
    TargetKind,
    TreeAggregation,
    TreeNodeKind,
    TreeSplitOp,
)
from omle_convert.catboost import from_catboost, from_catboost_file


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
DEPTH = 4


@pytest.fixture(scope="module")
def cb_regressor(regression_data):
    X, y = regression_data
    model = cb.CatBoostRegressor(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                  random_seed=0, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def cb_binary(binary_data):
    X, y = binary_data
    model = cb.CatBoostClassifier(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                   random_seed=0, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def cb_multiclass(multiclass_data):
    X, y = multiclass_data
    model = cb.CatBoostClassifier(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                   random_seed=0, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def cb_named(binary_data):
    import pandas as pd
    X_np, y = binary_data
    cols = [f"feat_{i}" for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=cols)
    model = cb.CatBoostClassifier(n_estimators=N_ESTIMATORS, depth=3,
                                   random_seed=0, verbose=0)
    model.fit(X, y)
    return model, cols


@pytest.fixture(scope="module")
def cb_regressor_f64(regression_data_f64):
    X, y = regression_data_f64
    model = cb.CatBoostRegressor(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                  random_seed=0, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def cb_binary_f64(binary_data_f64):
    X, y = binary_data_f64
    model = cb.CatBoostClassifier(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                   random_seed=0, verbose=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def cb_multiclass_f64(multiclass_data_f64):
    X, y = multiclass_data_f64
    model = cb.CatBoostClassifier(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                   random_seed=0, verbose=0)
    model.fit(X, y)
    return model


# ── Structure tests ───────────────────────────────────────────────────────────

class TestCatBoostRegressor:
    def test_returns_model(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_metadata(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        assert m.metadata.source_frameworks[0].name == "catboost"

    def test_uses_tree_ensemble(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].op == "TreeEnsemble"
        assert m.nodes[0].tree_ensemble is not None

    def test_post_transform_regression(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        te = m.nodes[0].tree_ensemble
        assert te.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_aggregation_sum(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        te = m.nodes[0].tree_ensemble
        assert te.aggregation == TreeAggregation.SUM

    def test_n_trees(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        te = m.nodes[0].tree_ensemble
        assert len(te.trees) >= N_ESTIMATORS

    def test_tree_structure_balanced(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        te = m.nodes[0].tree_ensemble
        real_trees = [t for t in te.trees if t.num_nodes > 1]
        assert real_trees
        t = real_trees[0]
        n = t.num_nodes
        assert (n & (n + 1)) == 0, f"Expected balanced tree, got {n} nodes"

    def test_output_spec(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        assert m.outputs[0].role == OutputRole.PREDICTION

    def test_schema_regression_target(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_json_roundtrip(self, cb_regressor, tmp_path):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        p = tmp_path / "reg.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        _assert_std_io(loaded, task="regression")
        assert len(loaded.nodes[0].tree_ensemble.trees) >= N_ESTIMATORS


class TestCatBoostBinaryClassifier:
    def test_returns_model(self, cb_binary):
        m = from_catboost(cb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        assert isinstance(m, omle.OMLEModel)

    def test_post_transform_sigmoid(self, cb_binary):
        m = from_catboost(cb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        te = m.nodes[0].tree_ensemble
        assert te.post_transform == PostTransform.SIGMOID_BINARY

    def test_output_roles(self, cb_binary):
        m = from_catboost(cb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        roles = {o.role for o in m.outputs}
        assert OutputRole.PREDICTION in roles
        assert OutputRole.PROBABILITY in roles

    def test_n_trees(self, cb_binary):
        m = from_catboost(cb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        te = m.nodes[0].tree_ensemble
        assert len(te.trees) >= N_ESTIMATORS

    def test_no_tree_group_for_binary(self, cb_binary):
        m = from_catboost(cb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        te = m.nodes[0].tree_ensemble
        assert te.tree_group == []

    def test_leaf_op_less_or_equal(self, cb_binary):
        m = from_catboost(cb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        te = m.nodes[0].tree_ensemble
        real_trees = [t for t in te.trees if t.num_nodes > 1]
        t = real_trees[0]
        branch_ops = [t.split_op[i] for i in range(t.num_nodes)
                      if t.node_kind[i] == TreeNodeKind.BRANCH]
        assert all(op == TreeSplitOp.LESS_OR_EQUAL for op in branch_ops)

    def test_schema_binary(self, cb_binary):
        m = from_catboost(cb_binary)
        _assert_std_io(m, task="binary", n_classes=2)
        assert m.model_schema.targets[0].kind == TargetKind.BINARY


class TestCatBoostMulticlassClassifier:
    def test_returns_model(self, cb_multiclass):
        m = from_catboost(cb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert isinstance(m, omle.OMLEModel)

    def test_post_transform_softmax(self, cb_multiclass):
        m = from_catboost(cb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        te = m.nodes[0].tree_ensemble
        assert te.post_transform == PostTransform.SOFTMAX

    def test_tree_group_assigned(self, cb_multiclass):
        m = from_catboost(cb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        te = m.nodes[0].tree_ensemble
        assert len(te.tree_group) == len(te.trees)
        assert set(te.tree_group) == {0, 1, 2}

    def test_n_trees_multiclass(self, cb_multiclass):
        m = from_catboost(cb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        te = m.nodes[0].tree_ensemble
        assert len(te.trees) >= N_ESTIMATORS * 3

    def test_schema_multiclass(self, cb_multiclass):
        m = from_catboost(cb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS


# ── WithX tests ───────────────────────────────────────────────────────────────

class TestCatBoostRegressorWithX:
    def test_converts(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_tree_count(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) >= N_ESTIMATORS

    def test_no_post_transform(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_input_spec_dtype(self, cb_regressor, regression_data):
        X, _ = regression_data  # float32
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_verification_populated(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None


class TestCatBoostBinaryWithX:
    def test_converts(self, cb_binary, binary_data):
        X, _ = binary_data
        m = from_catboost(cb_binary, X=X)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_sigmoid(self, cb_binary, binary_data):
        X, _ = binary_data
        m = from_catboost(cb_binary, X=X)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_tree_count(self, cb_binary, binary_data):
        X, _ = binary_data
        m = from_catboost(cb_binary, X=X)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) >= N_ESTIMATORS

    def test_input_spec_dtype(self, cb_binary, binary_data):
        X, _ = binary_data  # float32
        m = from_catboost(cb_binary, X=X)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32


class TestCatBoostMulticlassWithX:
    def test_converts(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_catboost(cb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert isinstance(m, omle.OMLEModel)

    def test_softmax(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_catboost(cb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_tree_count(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_catboost(cb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert len(m.nodes[0].tree_ensemble.trees) >= N_ESTIMATORS * 3

    def test_input_spec_dtype(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data  # float32
        m = from_catboost(cb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32


# ── Feature names ─────────────────────────────────────────────────────────────

class TestCatBoostFeatureNames:
    def test_feature_names_preserved(self, cb_named):
        model, cols = cb_named
        m = from_catboost(model)
        _assert_std_io(m, task="binary", n_classes=2)
        schema = m.model_schema
        if schema.features[0].range is not None:
            nr = schema.features[0].range
            assert nr.end - nr.start == len(cols)
        else:
            assert [f.name for f in schema.features] == cols

    def test_feature_count(self, cb_named):
        model, cols = cb_named
        m = from_catboost(model)
        _assert_std_io(m, task="binary", n_classes=2)
        assert m.inputs[0].type.shape[-1] == len(cols)

    def test_explicit_feature_names(self, cb_binary):
        names = [f"col_{i}" for i in range(6)]
        m = from_catboost(cb_binary, feature_names=names)
        _assert_std_io(m, task="binary", n_classes=2)
        expanded = [n for f in m.model_schema.features for n in f.expand_names()]
        assert expanded == names

    def test_custom_target_name(self, cb_regressor):
        m = from_catboost(cb_regressor, target_name="price")
        _assert_std_io(m, task="regression", target_name="price")
        assert m.model_schema.targets[0].name == "price"

    def test_custom_class_labels(self, cb_binary):
        m = from_catboost(cb_binary, class_labels=["no", "yes"])
        _assert_std_io(m, task="binary", n_classes=2)
        labels = [s.value for s in m.model_schema.targets[0].class_labels]
        assert labels == ["no", "yes"]


# ── Auxiliary data (verification / warmup / sample inputs) ───────────────────

class TestCatBoostAuxiliaryData:
    def test_no_auxiliary_by_default(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        assert m.verification is None
        assert m.warmup is None
        assert m.sample_inputs is None
        assert m.tensor_entries == []

    def test_tensor_entries_populated(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.tensor_entries) > 0

    def test_verification_populated(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        assert len(m.verification.cases) == 1
        assert m.verification.tolerance is not None
        assert m.verification.tolerance.atol is not None
        assert m.verification.tolerance.rtol is not None

    def test_warmup_populated(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup is not None
        assert len(m.warmup.cases) == 1
        assert m.warmup.cases[0].repeat == 3

    def test_sample_inputs_populated(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.sample_inputs is not None
        assert len(m.sample_inputs.cases) == 3

    def test_n_verify_zero_skips_verification(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X, n_verify=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is None
        assert m.warmup is not None

    def test_n_warmup_zero_skips_warmup(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X, n_warmup=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup is None
        assert m.sample_inputs is not None

    def test_n_sample_zero_skips_sample_inputs(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X, n_sample=0)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.sample_inputs is None

    def test_custom_n_sample(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X, n_sample=3)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.sample_inputs.cases) == 3

    def test_verify_inputs_reference_tensor_entries(self, cb_binary, binary_data):
        X, _ = binary_data
        m = from_catboost(cb_binary, X=X)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        entry_ids = {te.id for te in m.tensor_entries}
        case = m.verification.cases[0]
        assert all(r.id in entry_ids for r in case.inputs)
        assert all(r.id in entry_ids for r in case.expected_outputs)

    def test_regression_verify_output_name(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_pred" in out_names

    def test_binary_verify_output_name(self, cb_binary, binary_data):
        X, _ = binary_data
        m = from_catboost(cb_binary, X=X)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_multiclass_verify_output_name(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_catboost(cb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_warmup_repeat_param(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X, n_warmup_repeat=7)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup.cases[0].repeat == 7

    def test_tolerance_values(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = from_catboost(cb_regressor, X=X, verify_atol=1e-3, verify_rtol=2e-3)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        tol = m.verification.tolerance
        assert abs(tol.atol.float_value - 1e-3) < 1e-10
        assert abs(tol.rtol.float_value - 2e-3) < 1e-10

    def test_json_roundtrip_preserves_auxiliary(self, cb_binary, binary_data, tmp_path):
        X, _ = binary_data
        m = from_catboost(cb_binary, X=X, n_verify=2, n_warmup=4, n_sample=2)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        p = tmp_path / "with_aux.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        _assert_std_io(loaded, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT32)
        assert loaded.verification is not None
        assert loaded.warmup is not None
        assert loaded.sample_inputs is not None
        assert len(loaded.sample_inputs.cases) == 2
        assert len(loaded.tensor_entries) == len(m.tensor_entries)


# ── Float64 input data ────────────────────────────────────────────────────────

class TestCatBoostFloat64:
    def test_regression_converts(self, cb_regressor_f64):
        m = from_catboost(cb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_regression_tree_count(self, cb_regressor_f64):
        m = from_catboost(cb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) >= N_ESTIMATORS

    def test_regression_no_post_transform(self, cb_regressor_f64):
        m = from_catboost(cb_regressor_f64)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_binary_converts(self, cb_binary_f64):
        m = from_catboost(cb_binary_f64)
        _assert_std_io(m, task="binary", n_classes=2)
        assert isinstance(m, omle.OMLEModel)

    def test_binary_sigmoid(self, cb_binary_f64):
        m = from_catboost(cb_binary_f64)
        _assert_std_io(m, task="binary", n_classes=2)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_multiclass_converts(self, cb_multiclass_f64):
        m = from_catboost(cb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert isinstance(m, omle.OMLEModel)

    def test_multiclass_softmax(self, cb_multiclass_f64):
        m = from_catboost(cb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_n_features(self, cb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = from_catboost(cb_binary_f64)
        _assert_std_io(m, task="binary", n_classes=2)
        assert m.inputs[0].type.shape[-1] == X.shape[1]


# ── Numpy input dtype ─────────────────────────────────────────────────────────
#
# CatBoost uses float32 internally.
#   InputSpec.type.dtype  — actual numpy array dtype (what the caller provides)
#   Feature.type.dtype    — CatBoost's internal float32 precision

class TestCatBoostNumpyInputDtype:
    def test_float32_input_spec_dtype(self, cb_regressor, regression_data):
        X, _ = regression_data  # float32
        m = from_catboost(cb_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_float64_input_spec_dtype(self, cb_regressor_f64, regression_data_f64):
        X, _ = regression_data_f64  # float64
        m = from_catboost(cb_regressor_f64, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT64)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64

    def test_no_X_uses_float32(self, cb_regressor):
        m = from_catboost(cb_regressor)
        _assert_std_io(m, task="regression")
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32

    def test_binary_float64_input_spec(self, cb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = from_catboost(cb_binary_f64, X=X)
        _assert_std_io(m, task="binary", n_classes=2, input_dtype=omle.DataType.FLOAT64)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64

    def test_multiclass_float32_input_spec(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data  # float32
        m = from_catboost(cb_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32


# ── File serialization ────────────────────────────────────────────────────────

class TestCatBoostFileSerialization:
    def test_save_load_cbm(self, cb_binary, binary_data):
        X, _ = binary_data
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
            fname = f.name
        try:
            cb_binary.save_model(fname)
            m = from_catboost_file(fname)
            _assert_std_io(m, task="binary", n_classes=2)
            assert isinstance(m, omle.OMLEModel)
        finally:
            os.unlink(fname)

    def test_save_load_json(self, cb_binary, binary_data):
        X, _ = binary_data
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            cb_binary.save_model(fname, format="json")
            m = from_catboost_file(fname)
            _assert_std_io(m, task="binary", n_classes=2)
            assert isinstance(m, omle.OMLEModel)
        finally:
            os.unlink(fname)

    def test_kwargs_forwarded(self, cb_regressor, tmp_path):
        fname = str(tmp_path / "reg.cbm")
        cb_regressor.save_model(fname)
        m = from_catboost_file(fname, target_name="price", model_name="my_model")
        _assert_std_io(m, task="regression", target_name="price")
        assert m.model_schema.targets[0].name == "price"
        assert m.metadata.name == "my_model"


# ── Runtime prediction tests ──────────────────────────────────────────────────

def _load_runtime(omle_model):
    """Load model into the C++ runtime."""
    try:
        import omle_runtime as omr
    except ImportError:
        pytest.skip("omle_runtime not installed")
    from omle.proto.convert import ir_to_proto
    data = ir_to_proto(omle_model).SerializeToString()
    return omr.load_bytes(data)


class TestCatBoostRuntimeRegressor:
    def test_loads_without_error(self, cb_regressor):
        m = _load_runtime(from_catboost(cb_regressor))
        assert m is not None

    def test_n_features_in(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_catboost(cb_regressor))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_dtype(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_catboost(cb_regressor))
        assert m.predict(X).dtype == np.float64

    def test_regressor_matches_native(self, cb_regressor, regression_data):
        X, _ = regression_data
        m = _load_runtime(from_catboost(cb_regressor))
        expected = cb_regressor.predict(X)
        got = m.predict(X)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_runtime_consistent(self, cb_regressor, regression_data, tmp_path):
        X, _ = regression_data
        ir = from_catboost(cb_regressor)
        _assert_std_io(ir, task="regression")
        p = tmp_path / "reg.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestCatBoostRuntimeBinary:
    def test_loads_without_error(self, cb_binary):
        m = _load_runtime(from_catboost(cb_binary))
        assert m is not None

    def test_n_features_in(self, cb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_catboost(cb_binary))
        assert m.n_features_in_ == X.shape[1]

    def test_predict_proba_dtype(self, cb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_catboost(cb_binary))
        assert m.predict_proba(X).dtype == np.float64

    def test_binary_proba_matches_native(self, cb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_catboost(cb_binary))
        np.testing.assert_allclose(m.predict_proba(X), cb_binary.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_binary_predict_matches_native(self, cb_binary, binary_data):
        X, _ = binary_data
        m = _load_runtime(from_catboost(cb_binary))
        expected = cb_binary.predict(X)
        got = m.predict(X)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_runtime_consistent(self, cb_binary, binary_data, tmp_path):
        X, _ = binary_data
        ir = from_catboost(cb_binary)
        _assert_std_io(ir, task="binary")
        p = tmp_path / "bin.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestCatBoostRuntimeMulticlass:
    def test_loads_without_error(self, cb_multiclass):
        m = _load_runtime(from_catboost(cb_multiclass))
        assert m is not None

    def test_n_features_in(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_catboost(cb_multiclass))
        assert m.n_features_in_ == X.shape[1]

    def test_multiclass_proba_matches_native(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_catboost(cb_multiclass))
        expected = cb_multiclass.predict_proba(X)
        got = m.predict_proba(X)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_multiclass_predict_matches_native(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = _load_runtime(from_catboost(cb_multiclass))
        expected = cb_multiclass.predict(X).ravel()
        got = m.predict(X).ravel()
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_runtime_consistent(self, cb_multiclass, multiclass_data, tmp_path):
        X, _ = multiclass_data
        ir = from_catboost(cb_multiclass)
        _assert_std_io(ir, task="multiclass", n_classes=3)
        p = tmp_path / "mc.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        r1 = _load_runtime(ir).predict_proba(X)
        r2 = _load_runtime(ir2).predict_proba(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ── Categorical features ──────────────────────────────────────────────────────

class TestCatBoostCategoricalFeatures:
    def test_raises_on_ctr_splits(self, binary_data):
        """Models with CatBoost CTR (categorical) splits raise NotImplementedError."""
        import pandas as pd
        X_np, y = binary_data
        cats = np.array(["a", "b"] * (len(y) // 2))
        X = pd.DataFrame({"num": X_np[:, 0], "cat": cats})
        model = cb.CatBoostClassifier(n_estimators=5, depth=2, verbose=0,
                                       cat_features=["cat"])
        model.fit(X, y)
        with pytest.raises(NotImplementedError, match="FloatFeature"):
            from_catboost(model)


# ── Homogeneous DataFrame (all columns same dtype) ────────────────────────────

@pytest.fixture(scope="module")
def cb_homogeneous_f32(regression_data):
    import pandas as pd
    X, y = regression_data  # float32
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    model = cb.CatBoostRegressor(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                  random_seed=0, verbose=0)
    model.fit(df, y)
    return model, df


@pytest.fixture(scope="module")
def cb_homogeneous_f64(regression_data_f64):
    import pandas as pd
    X, y = regression_data_f64  # float64
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    model = cb.CatBoostRegressor(n_estimators=N_ESTIMATORS, depth=DEPTH,
                                  random_seed=0, verbose=0)
    model.fit(df, y)
    return model, df


class TestCatBoostHomogeneousDataFrame:
    """When all DataFrame columns share the same dtype, a single 'X' matrix InputSpec is used."""

    def test_converts_f32(self, cb_homogeneous_f32):
        model, df = cb_homogeneous_f32
        assert from_catboost(model, X=df) is not None

    def test_converts_f64(self, cb_homogeneous_f64):
        model, df = cb_homogeneous_f64
        assert from_catboost(model, X=df) is not None

    def test_single_input_spec_f32(self, cb_homogeneous_f32):
        model, df = cb_homogeneous_f32
        m = from_catboost(model, X=df)
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert m.inputs[0].type.shape == [-1, len(df.columns)]

    def test_single_input_spec_f64(self, cb_homogeneous_f64):
        model, df = cb_homogeneous_f64
        m = from_catboost(model, X=df)
        assert len(m.inputs) == 1
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, len(df.columns)]

    def test_features_name_range_f32(self, cb_homogeneous_f32):
        model, df = cb_homogeneous_f32
        m = from_catboost(model, X=df)
        assert len(m.model_schema.features) == 1
        feat = m.model_schema.features[0]
        assert feat.range is not None
        assert feat.range.start == 0
        assert feat.range.end == len(df.columns)
        assert feat.source == "X"
        assert feat.index == 0

    def test_features_name_range_f64(self, cb_homogeneous_f64):
        model, df = cb_homogeneous_f64
        m = from_catboost(model, X=df)
        assert len(m.model_schema.features) == 1
        feat = m.model_schema.features[0]
        assert feat.range is not None
        assert feat.range.start == 0
        assert feat.range.end == len(df.columns)
        assert feat.source == "X"
        assert feat.index == 0

    def test_feature_dtype_is_cb_internal_f32(self, cb_homogeneous_f32):
        model, df = cb_homogeneous_f32
        m = from_catboost(model, X=df)
        # CatBoost uses float32 internally
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32

    def test_feature_dtype_is_cb_internal_f64(self, cb_homogeneous_f64):
        model, df = cb_homogeneous_f64
        m = from_catboost(model, X=df)
        # Even with float64 input, CatBoost Feature.type is float32 (internal precision)
        assert m.model_schema.features[0].type.dtype == omle.DataType.FLOAT32

    def test_estimator_node_input_is_X(self, cb_homogeneous_f32):
        model, df = cb_homogeneous_f32
        m = from_catboost(model, X=df)
        assert len(m.nodes) == 1
        assert m.nodes[0].inputs[0].name.value == "X"
