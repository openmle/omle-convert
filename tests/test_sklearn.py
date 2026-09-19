"""Tests for the scikit-learn → OMLE converter."""

import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

import numpy as np

import omle
from omle.ir.enums import (
    OutputRole,
    PostTransform,
    TargetKind,
    TreeAggregation,
    TreeNodeKind,
    TreeSplitOp,
)
from omle_convert.sklearn import from_sklearn

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tv_float64(tv, model):
    """Resolve a TensorValue to a flat float64 list, handling both inline and tensor_ref storage."""
    if tv is None:
        return []
    if tv.tensor is not None:
        return list(tv.tensor.float64_data or tv.tensor.float32_data or [])
    if tv.tensor_ref is not None:
        idx = {e.id: e for e in (model.tensor_entries or [])}
        entry = idx.get(tv.tensor_ref.id)
        if entry and entry.dense:
            return list(entry.dense.float64_data or entry.dense.float32_data or [])
    return []


def _all_nodes(nodes):
    """Flatten a node list recursively, including nodes inside CompositeNode bodies."""
    result = []
    for n in nodes:
        result.append(n)
        if n.composite is not None:
            result.extend(_all_nodes(n.composite.nodes))
    return result


def _load_runtime(ir_model):
    omr = pytest.importorskip("omle_runtime", reason="omle_runtime not installed")
    from omle.proto.convert import ir_to_proto
    data = ir_to_proto(ir_model).SerializeToString()
    return omr.load_bytes(data)


def _tensor_strings(entries, attr):
    """Return string_data from attr — inline tensor or tensor_ref lookup."""
    if attr.tensor is not None:
        return attr.tensor.string_data
    entry = next(e for e in entries if e.id == attr.tensor_ref.id)
    return entry.dense.string_data


def _attr_has_tensor(attr) -> bool:
    """True if the attribute carries tensor data (inline or by reference)."""
    return attr.tensor is not None or attr.tensor_ref is not None


def _attr_float64_data(entries, attr) -> list:
    """Return float64_data from attr — inline tensor or tensor_ref lookup."""
    if attr.tensor is not None:
        return attr.tensor.float64_data
    entry = next(e for e in entries if e.id == attr.tensor_ref.id)
    return entry.dense.float64_data


def _attr_int64_data(entries, attr) -> list:
    """Return int64_data from attr — inline tensor or tensor_ref lookup."""
    if attr.tensor is not None:
        return attr.tensor.int64_data
    entry = next(e for e in entries if e.id == attr.tensor_ref.id)
    return entry.dense.int64_data


def _assert_std_io(
    m,
    *,
    task: str,
    n_features: int = 6,
    n_classes: int = 2,
    input_name: str = "X",
    input_dtype=None,
    target_name: str = "y",
) -> None:
    """Assert the standard IR input/output/schema invariants.

    task: "regression" | "binary" | "multiclass" | "clustering" | "transformer"
    """
    if input_dtype is None:
        input_dtype = omle.DataType.FLOAT64
    assert m.inputs[0].name == input_name
    assert m.inputs[0].type.dtype == input_dtype
    assert m.inputs[0].type.shape == [-1, n_features]
    if task == "regression":
        out = next(o for o in m.outputs if o.name == "y_pred")
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
        assert m.model_schema.targets[0].kind == (TargetKind.BINARY if task == "binary" else TargetKind.MULTICLASS)
        assert m.model_schema.targets[0].name == target_name
    elif task == "clustering":
        cid = next(o for o in m.outputs if o.name == "cluster_id")
        assert cid.role == OutputRole.ENTITY_ID
        assert cid.type.dtype == omle.DataType.INT32
        assert cid.type.shape == [-1]


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_ESTIMATORS = 10


@pytest.fixture(scope="module")
def dt_regressor(regression_data):
    from sklearn.tree import DecisionTreeRegressor
    X, y = regression_data
    return DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def dt_binary(binary_data):
    from sklearn.tree import DecisionTreeClassifier
    X, y = binary_data
    return DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def dt_multiclass(multiclass_data):
    from sklearn.tree import DecisionTreeClassifier
    X, y = multiclass_data
    return DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def rf_regressor(regression_data):
    from sklearn.ensemble import RandomForestRegressor
    X, y = regression_data
    return RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=3,
                                 random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def rf_binary(binary_data):
    from sklearn.ensemble import RandomForestClassifier
    X, y = binary_data
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                  random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def rf_multiclass(multiclass_data):
    from sklearn.ensemble import RandomForestClassifier
    X, y = multiclass_data
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                  random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def et_binary(binary_data):
    from sklearn.ensemble import ExtraTreesClassifier
    X, y = binary_data
    return ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def gb_regressor(regression_data):
    from sklearn.ensemble import GradientBoostingRegressor
    X, y = regression_data
    return GradientBoostingRegressor(n_estimators=N_ESTIMATORS, max_depth=3,
                                     learning_rate=0.1, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def gb_binary(binary_data):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = binary_data
    return GradientBoostingClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                      learning_rate=0.1, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def gb_multiclass(multiclass_data):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = multiclass_data
    return GradientBoostingClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                      learning_rate=0.1, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def hgb_regressor(regression_data):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y = regression_data
    return HistGradientBoostingRegressor(max_iter=N_ESTIMATORS, max_depth=3,
                                         random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def hgb_binary(binary_data):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = binary_data
    return HistGradientBoostingClassifier(max_iter=N_ESTIMATORS, max_depth=3,
                                          random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def hgb_multiclass(multiclass_data):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = multiclass_data
    return HistGradientBoostingClassifier(max_iter=N_ESTIMATORS, max_depth=3,
                                          random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def rf_named_features(binary_data):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    X_np, y = binary_data
    cols = [f"feat_{i}" for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=cols)
    return RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y), cols


@pytest.fixture(scope="module")
def pipeline_rf(binary_data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = binary_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def dt_regressor_f64(regression_data_f64):
    from sklearn.tree import DecisionTreeRegressor
    X, y = regression_data_f64
    return DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def rf_binary_f64(binary_data_f64):
    from sklearn.ensemble import RandomForestClassifier
    X, y = binary_data_f64
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                  random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def gb_multiclass_f64(multiclass_data_f64):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = multiclass_data_f64
    return GradientBoostingClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                      learning_rate=0.1, random_state=0).fit(X, y)


# ── DecisionTree ──────────────────────────────────────────────────────────────

class TestDecisionTreeRegressor:
    def test_returns_omle_model(self, dt_regressor):
        m = from_sklearn(dt_regressor)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_metadata(self, dt_regressor):
        m = from_sklearn(dt_regressor)
        _assert_std_io(m, task="regression")
        assert m.metadata.source_frameworks[0].name == "sklearn"

    def test_uses_tree_op(self, dt_regressor):
        m = from_sklearn(dt_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].op == "Tree"
        assert m.nodes[0].tree is not None

    def test_schema_regression_target(self, dt_regressor):
        m = from_sklearn(dt_regressor)
        _assert_std_io(m, task="regression")
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_single_output(self, dt_regressor):
        m = from_sklearn(dt_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.outputs) == 1
        assert m.outputs[0].role == OutputRole.PREDICTION

    def test_runtime_matches_native(self, dt_regressor, regression_data):
        X, _ = regression_data
        _ir = from_sklearn(dt_regressor)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(dt_regressor, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), dt_regressor.predict(X), rtol=1e-4, atol=1e-4)


class TestDecisionTreeBinary:
    def test_uses_tree_op(self, dt_binary):
        m = from_sklearn(dt_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].op == "Tree"
        assert m.nodes[0].tree is not None

    def test_schema_binary_target(self, dt_binary):
        m = from_sklearn(dt_binary)
        _assert_std_io(m, task="binary")
        assert m.model_schema.targets[0].kind == TargetKind.BINARY

    def test_leaf_values_in_01(self, dt_binary):
        # Binary DT uses leaf_vector (width 2) with per-class probabilities.
        m = from_sklearn(dt_binary)
        _assert_std_io(m, task="binary")
        tree = m.nodes[0].tree
        assert tree.leaf_width == 2
        assert tree.leaf_vector is not None
        data = _tv_float64(tree.leaf_vector, m)
        for i, kind in enumerate(tree.node_kind):
            if kind == TreeNodeKind.LEAF:
                offset = tree.leaf_vector_index[i]
                for c in range(2):
                    assert 0.0 <= data[offset + c] <= 1.0

    def test_runtime_matches_native(self, dt_binary, binary_data):
        X, _ = binary_data
        _ir = from_sklearn(dt_binary)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(dt_binary, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), dt_binary.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), dt_binary.predict_proba(X), rtol=1e-4, atol=1e-4)


class TestDecisionTreeMulticlass:
    def test_uses_tree_op(self, dt_multiclass):
        m = from_sklearn(dt_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].op == "Tree"
        assert m.nodes[0].tree is not None

    def test_vector_leaves(self, dt_multiclass):
        m = from_sklearn(dt_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tree = m.nodes[0].tree
        assert tree.leaf_width == 3
        assert tree.leaf_vector is not None
        assert len(_tv_float64(tree.leaf_vector, m)) > 0

    def test_leaf_probs_sum_to_one(self, dt_multiclass):
        m = from_sklearn(dt_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tree = m.nodes[0].tree
        data = _tv_float64(tree.leaf_vector, m)
        for i, kind in enumerate(tree.node_kind):
            if kind == TreeNodeKind.LEAF:
                offset = tree.leaf_vector_index[i]
                probs = data[offset:offset + tree.leaf_width]
                assert abs(sum(probs) - 1.0) < 1e-6

    def test_schema_multiclass_target(self, dt_multiclass):
        m = from_sklearn(dt_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS

    def test_runtime_matches_native(self, dt_multiclass, multiclass_data):
        X, _ = multiclass_data
        _ir = from_sklearn(dt_multiclass)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(dt_multiclass, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), dt_multiclass.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), dt_multiclass.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── RandomForest ──────────────────────────────────────────────────────────────

class TestRandomForestRegressor:
    def test_correct_tree_count(self, rf_regressor):
        m = from_sklearn(rf_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_average_aggregation(self, rf_regressor):
        m = from_sklearn(rf_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.AVERAGE

    def test_no_post_transform(self, rf_regressor):
        m = from_sklearn(rf_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_no_tree_group(self, rf_regressor):
        m = from_sklearn(rf_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.tree_group == []

    def test_json_roundtrip(self, rf_regressor, tmp_path):
        m = from_sklearn(rf_regressor)
        _assert_std_io(m, task="regression")
        p = tmp_path / "reg.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        assert len(loaded.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_runtime_matches_native(self, rf_regressor, regression_data):
        X, _ = regression_data
        _ir = from_sklearn(rf_regressor)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(rf_regressor, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), rf_regressor.predict(X), rtol=1e-4, atol=1e-4)


class TestRandomForestBinary:
    def test_correct_tree_count(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_average_aggregation(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.AVERAGE

    def test_no_tree_group(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.tree_group == []

    def test_schema_binary_target(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        assert m.model_schema.targets[0].kind == TargetKind.BINARY

    def test_leaf_values_in_01(self, rf_binary):
        # Binary RF uses leaf_vector (width 2) with per-class probabilities.
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            assert tree.leaf_width == 2
            assert tree.leaf_vector is not None
            data = _tv_float64(tree.leaf_vector, m)
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.LEAF:
                    offset = tree.leaf_vector_index[i]
                    for c in range(2):
                        assert 0.0 <= data[offset + c] <= 1.0

    def test_runtime_matches_native(self, rf_binary, binary_data):
        X, _ = binary_data
        _ir = from_sklearn(rf_binary)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(rf_binary, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), rf_binary.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), rf_binary.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_json_roundtrip_consistent(self, rf_binary, binary_data, tmp_path):
        X, _ = binary_data
        ir = from_sklearn(rf_binary)
        _assert_std_io(ir, task="binary")
        p = tmp_path / "rf_bin.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        _assert_std_io(ir2, task="binary")
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


class TestRandomForestMulticlass:
    def test_total_trees(self, rf_multiclass):
        m = from_sklearn(rf_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3

    def test_soft_vote_aggregation(self, rf_multiclass):
        m = from_sklearn(rf_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SOFT_VOTE

    def test_no_post_transform(self, rf_multiclass):
        # Leaf values are already normalized class probabilities; no post-transform needed
        m = from_sklearn(rf_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_tree_group_round_robin(self, rf_multiclass):
        m = from_sklearn(rf_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        assert len(tg) == N_ESTIMATORS * 3
        for i, g in enumerate(tg):
            assert g == i % 3

    def test_leaf_values_in_01(self, rf_multiclass):
        m = from_sklearn(rf_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        for tree in m.nodes[0].tree_ensemble.trees:
            data = _tv_float64(tree.leaf_value, m)
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.LEAF:
                    assert 0.0 <= data[i] <= 1.0

    def test_runtime_matches_native(self, rf_multiclass, multiclass_data):
        X, _ = multiclass_data
        _ir = from_sklearn(rf_multiclass)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(rf_multiclass, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), rf_multiclass.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), rf_multiclass.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── ExtraTrees ────────────────────────────────────────────────────────────────

class TestExtraTrees:
    def test_correct_tree_count(self, et_binary):
        m = from_sklearn(et_binary)
        _assert_std_io(m, task="binary")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_average_aggregation(self, et_binary):
        m = from_sklearn(et_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.AVERAGE

    def test_regressor_converts(self, regression_data):
        from sklearn.ensemble import ExtraTreesRegressor
        X, y = regression_data
        m = from_sklearn(ExtraTreesRegressor(n_estimators=N_ESTIMATORS, max_depth=3, random_state=0).fit(X, y))
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.AVERAGE

    def test_runtime_binary_matches_native(self, binary_data):
        from sklearn.ensemble import ExtraTreesClassifier
        X, y = binary_data
        est = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=3, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_multiclass_matches_native(self, multiclass_data):
        from sklearn.ensemble import ExtraTreesClassifier
        X, y = multiclass_data
        est = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=3, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_regressor_matches_native(self, regression_data):
        from sklearn.ensemble import ExtraTreesRegressor
        X, y = regression_data
        est = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, max_depth=3, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)


# ── GradientBoosting ──────────────────────────────────────────────────────────

class TestGradientBoostingRegressor:
    def test_correct_tree_count(self, gb_regressor):
        m = from_sklearn(gb_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_sum_aggregation(self, gb_regressor):
        m = from_sklearn(gb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_no_post_transform(self, gb_regressor):
        m = from_sklearn(gb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_base_score_set(self, gb_regressor):
        m = from_sklearn(gb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.base_scores is not None

    def test_no_tree_group(self, gb_regressor):
        m = from_sklearn(gb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.tree_group == []

    def test_runtime_matches_native(self, gb_regressor, regression_data):
        X, _ = regression_data
        _ir = from_sklearn(gb_regressor)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(gb_regressor, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), gb_regressor.predict(X), rtol=1e-4, atol=1e-4)


class TestGradientBoostingBinary:
    def test_correct_tree_count(self, gb_binary):
        m = from_sklearn(gb_binary)
        _assert_std_io(m, task="binary")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_sum_aggregation(self, gb_binary):
        m = from_sklearn(gb_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_sigmoid_post_transform(self, gb_binary):
        m = from_sklearn(gb_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_base_score_set(self, gb_binary):
        m = from_sklearn(gb_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.base_scores is not None

    def test_schema_binary_target(self, gb_binary):
        m = from_sklearn(gb_binary)
        _assert_std_io(m, task="binary")
        assert m.model_schema.targets[0].kind == TargetKind.BINARY

    def test_runtime_matches_native(self, gb_binary, binary_data):
        X, _ = binary_data
        _ir = from_sklearn(gb_binary)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(gb_binary, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), gb_binary.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), gb_binary.predict_proba(X), rtol=1e-4, atol=1e-4)


class TestGradientBoostingMulticlass:
    def test_total_trees(self, gb_multiclass):
        m = from_sklearn(gb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        # 3 per-class bias trees prepended + N_ESTIMATORS * 3 boosting trees
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3 + 3

    def test_sum_aggregation(self, gb_multiclass):
        m = from_sklearn(gb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_softmax_post_transform(self, gb_multiclass):
        m = from_sklearn(gb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_tree_group_round_robin(self, gb_multiclass):
        m = from_sklearn(gb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        assert len(tg) == N_ESTIMATORS * 3 + 3
        # first 3 are bias trees [0,1,2]; remainder cycle through classes
        assert tg[:3] == [0, 1, 2]
        for i, g in enumerate(tg[3:]):
            assert g == i % 3

    def test_schema_multiclass_target(self, gb_multiclass):
        m = from_sklearn(gb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS

    def test_runtime_matches_native(self, gb_multiclass, multiclass_data):
        X, _ = multiclass_data
        _ir = from_sklearn(gb_multiclass)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(gb_multiclass, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), gb_multiclass.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), gb_multiclass.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── HistGradientBoosting ──────────────────────────────────────────────────────

class TestHistGradientBoostingRegressor:
    def test_returns_omle_model(self, hgb_regressor):
        m = from_sklearn(hgb_regressor)
        _assert_std_io(m, task="regression")
        assert isinstance(m, omle.OMLEModel)

    def test_correct_tree_count(self, hgb_regressor):
        m = from_sklearn(hgb_regressor)
        _assert_std_io(m, task="regression")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_sum_aggregation(self, hgb_regressor):
        m = from_sklearn(hgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.SUM

    def test_no_post_transform(self, hgb_regressor):
        m = from_sklearn(hgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_base_score_set(self, hgb_regressor):
        m = from_sklearn(hgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.nodes[0].tree_ensemble.base_scores is not None

    def test_schema_regression_target(self, hgb_regressor):
        m = from_sklearn(hgb_regressor)
        _assert_std_io(m, task="regression")
        assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_runtime_matches_native(self, hgb_regressor, regression_data):
        X, _ = regression_data
        _ir = from_sklearn(hgb_regressor)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(hgb_regressor, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), hgb_regressor.predict(X), rtol=1e-4, atol=1e-4)


class TestHistGradientBoostingBinary:
    def test_correct_tree_count(self, hgb_binary):
        m = from_sklearn(hgb_binary)
        _assert_std_io(m, task="binary")
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_sigmoid_post_transform(self, hgb_binary):
        m = from_sklearn(hgb_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SIGMOID_BINARY

    def test_base_score_set(self, hgb_binary):
        m = from_sklearn(hgb_binary)
        _assert_std_io(m, task="binary")
        assert m.nodes[0].tree_ensemble.base_scores is not None

    def test_schema_binary_target(self, hgb_binary):
        m = from_sklearn(hgb_binary)
        _assert_std_io(m, task="binary")
        assert m.model_schema.targets[0].kind == TargetKind.BINARY

    def test_runtime_matches_native(self, hgb_binary, binary_data):
        X, _ = binary_data
        _ir = from_sklearn(hgb_binary)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(hgb_binary, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), hgb_binary.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), hgb_binary.predict_proba(X), rtol=1e-4, atol=1e-4)


class TestHistGradientBoostingMulticlass:
    def test_total_trees(self, hgb_multiclass):
        m = from_sklearn(hgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        # 3 per-class bias trees prepended + N_ESTIMATORS * 3 boosting trees
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS * 3 + 3

    def test_softmax_post_transform(self, hgb_multiclass):
        m = from_sklearn(hgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_tree_group_round_robin(self, hgb_multiclass):
        m = from_sklearn(hgb_multiclass)
        _assert_std_io(m, task="multiclass", n_classes=3)
        tg = m.nodes[0].tree_ensemble.tree_group
        assert len(tg) == N_ESTIMATORS * 3 + 3
        # first 3 are bias trees [0,1,2]; remainder cycle through classes
        assert tg[:3] == [0, 1, 2]
        for i, g in enumerate(tg[3:]):
            assert g == i % 3

    def test_runtime_matches_native(self, hgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        _ir = from_sklearn(hgb_multiclass)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(hgb_multiclass, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), hgb_multiclass.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), hgb_multiclass.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── Feature names & options ───────────────────────────────────────────────────

class TestSklearnOptions:
    def test_named_features_inferred(self, rf_named_features):
        model, cols = rf_named_features
        m = from_sklearn(model)
        _assert_std_io(m, task="binary")
        assert [n for f in m.model_schema.features for n in f.expand_names()] == cols

    def test_explicit_feature_names(self, rf_binary):
        names = [f"col_{i}" for i in range(6)]
        m = from_sklearn(rf_binary, feature_names=names)
        _assert_std_io(m, task="binary")
        assert [n for f in m.model_schema.features for n in f.expand_names()] == names

    def test_custom_target_name(self, rf_regressor):
        m = from_sklearn(rf_regressor, target_name="price")
        _assert_std_io(m, task="regression", target_name="price")
        assert m.model_schema.targets[0].name == "price"

    def test_custom_class_labels(self, rf_binary):
        m = from_sklearn(rf_binary, class_labels=["no", "yes"])
        _assert_std_io(m, task="binary")
        labels = [s.value for s in m.model_schema.targets[0].class_labels]
        assert labels == ["no", "yes"]

    def test_model_name(self, rf_binary):
        m = from_sklearn(rf_binary, model_name="my_rf")
        _assert_std_io(m, task="binary")
        assert m.metadata.name == "my_rf"

    def test_dataframe_columns_become_input_names(self, rf_binary, binary_data):
        """When X is a DataFrame, model.inputs use the column names, not 'X'."""
        import pandas as pd
        X_np, _ = binary_data
        cols = [f"feat_{i}" for i in range(X_np.shape[1])]
        X_df = pd.DataFrame(X_np, columns=cols)
        m = from_sklearn(rf_binary, X=X_df)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        # Uniform numeric dtype: single 2-D tensor named "X" (all-same dtype)
        # or per-column if mixed. For all-float, the result is a single "X" tensor.
        # Either way, the schema features should reflect the column names.
        schema_names = [n for f in m.model_schema.features for n in f.expand_names()]
        assert schema_names == cols

    def test_mixed_dataframe_inputs_are_per_column(self, rf_binary):
        """Mixed-dtype DataFrame → per-column InputSpecs with correct names."""
        import pandas as pd
        X = pd.DataFrame({
            "age":    [25.0, 30.0, 35.0, 40.0],
            "income": [50.0, 60.0, 70.0, 80.0],
        })
        y = [0, 1, 0, 1]
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        m = from_sklearn(clf, X=X)
        _assert_std_io(m, task="binary", n_features=2, input_dtype=omle.DataType.FLOAT64)
        # Uniform float → single "X" tensor; schema features should map column names
        schema_names = [n for f in m.model_schema.features for n in f.expand_names()]
        assert schema_names == ["age", "income"]

    def test_output_names_classification(self, rf_binary):
        """Classifier outputs are named y_pred and y_prob."""
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        output_names = {o.name for o in m.outputs}
        assert "y_pred" in output_names
        assert "y_prob" in output_names

    def test_output_names_regression(self, rf_regressor):
        """Regressor output is named y_pred (no probability)."""
        m = from_sklearn(rf_regressor)
        _assert_std_io(m, task="regression")
        output_names = {o.name for o in m.outputs}
        assert "y_pred" in output_names
        assert "y_prob" not in output_names

    def test_unsupported_estimator_raises(self):
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        clf = GaussianProcessRegressor().fit(X, y)
        with pytest.raises(NotImplementedError):
            from_sklearn(clf)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_converts(self, pipeline_rf):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        assert isinstance(m, omle.OMLEModel)

    def test_pipeline_has_scaler_node(self, pipeline_rf):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "StandardScaler" in ops

    def test_pipeline_has_ensemble_node(self, pipeline_rf):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        ensemble_node = m.get_node("clf_random_forest_classifier")
        assert ensemble_node is not None
        assert len(ensemble_node.tree_ensemble.trees) == 5

    def test_pipeline_binary_target(self, pipeline_rf):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        assert m.model_schema.targets[0].kind == TargetKind.BINARY

    def test_pipeline_scaler_constants(self, pipeline_rf):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        # StandardScaler stores mean and scale as inline tensor attributes
        scaler_node = m.get_node("scaler_standard_scaler")
        attr_names = {a.name for a in scaler_node.attributes}
        assert "mean" in attr_names
        assert "scale" in attr_names

    def test_pipeline_scaler_wires_to_ensemble(self, pipeline_rf):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        # ensemble node input should reference the scaler output, not raw X
        ensemble_node = m.get_node("clf_random_forest_classifier")
        assert ensemble_node.inputs[0].name != "X"

    def test_scaler_field_names_from_dataframe(self):
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        X = pd.DataFrame({"age": [25.0, 30.0, 35.0], "income": [50.0, 60.0, 70.0]})
        y = [0, 1, 0]
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", DecisionTreeClassifier()),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary", n_features=2)
        sc_node = next(n for n in m.nodes if n.op == "StandardScaler")
        # field_names come from get_feature_names_out() — same as input columns
        assert sc_node.outputs[0].field_names == ["age", "income"]
        # output tensor name must be different from any input name
        input_names = {i.name for i in m.inputs}
        assert sc_node.outputs[0].name not in input_names

    def test_runtime_matches_native(self, pipeline_rf, binary_data):
        X, _ = binary_data
        _ir = from_sklearn(pipeline_rf)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipeline_rf, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), pipeline_rf.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), pipeline_rf.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_json_roundtrip_consistent(self, pipeline_rf, binary_data, tmp_path):
        X, _ = binary_data
        ir = from_sklearn(pipeline_rf)
        _assert_std_io(ir, task="binary")
        p = tmp_path / "pipe_rf.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        _assert_std_io(ir2, task="binary")
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ── Pipeline preprocessing ────────────────────────────────────────────────────

class TestPipelinePreprocessing:
    """Tests for the various preprocessing transformer converters."""

    def _make_pipe(self, transformer, estimator_cls, data):
        from sklearn.pipeline import Pipeline
        X, y = data
        pipe = Pipeline([
            ("prep", transformer),
            ("clf",  estimator_cls(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        return from_sklearn(pipe)

    def test_minmax_scaler(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import MinMaxScaler
        m = self._make_pipe(MinMaxScaler(), RandomForestClassifier, binary_data)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "MinMaxScaler" in ops
        node = next(n for n in m.nodes if n.op == "MinMaxScaler")
        attr_names = {a.name for a in node.attributes}
        assert "data_min" in attr_names
        assert "data_max" in attr_names

    def test_robust_scaler(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import RobustScaler
        m = self._make_pipe(RobustScaler(), RandomForestClassifier, binary_data)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "RobustScaler" in ops

    def test_maxabs_scaler(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import MaxAbsScaler
        m = self._make_pipe(MaxAbsScaler(), RandomForestClassifier, binary_data)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "MaxAbsScaler" in ops

    def test_normalizer(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import Normalizer
        m = self._make_pipe(Normalizer(), RandomForestClassifier, binary_data)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Normalizer" in ops
        norm_node = next(n for n in m.nodes if n.op == "Normalizer")
        assert any(a.name == "norm" for a in norm_node.attributes)

    def test_simple_imputer(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        m = self._make_pipe(SimpleImputer(strategy="mean"), RandomForestClassifier, binary_data)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Imputer" in ops

    def test_binarizer(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Binarizer
        X, y = regression_data
        pipe = Pipeline([
            ("bin", Binarizer(threshold=0.5)),
            ("reg", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "Binarizer" in ops
        bin_node = next(n for n in m.nodes if n.op == "Binarizer")
        thresh_attr = next(a for a in bin_node.attributes if a.name == "threshold")
        assert abs(thresh_attr.f64 - 0.5) < 1e-6

    def test_pca(self, binary_data):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        m = self._make_pipe(PCA(n_components=3), RandomForestClassifier, binary_data)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Linear" in ops
        assert len(m.tensor_entries) >= 1  # at least the components matrix

    def test_ordinal_encoder(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
        X, y = binary_data
        pipe = Pipeline([
            ("enc", OrdinalEncoder()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "OrdinalEncoder" in ops

    def test_ohe_single_feature(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        # Use a single categorical-like feature
        X, y = binary_data
        X_cat = (X[:, :1] * 3).astype(int).astype(str)
        pipe = Pipeline([
            ("enc", OneHotEncoder(sparse_output=False)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X_cat, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary", n_features=1)
        ops = [n.op for n in m.nodes]
        assert "OneHotEncoder" in ops

    def test_column_transformer(self, binary_data):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        X, y = binary_data
        ct = ColumnTransformer([
            ("std", StandardScaler(),  [0, 1, 2]),
            ("mm",  MinMaxScaler(),    [3, 4, 5]),
        ])
        pipe = Pipeline([
            ("prep", ct),
            ("clf",  RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ct_node = next(n for n in m.nodes if n.op == "Composite")
        assert ct_node.composite is not None
        inner_ops = [n.op for n in ct_node.composite.nodes]
        assert "TakeSlots"      in inner_ops
        assert "StandardScaler" in inner_ops
        assert "MinMaxScaler"   in inner_ops
        assert "Concat"         in inner_ops

    def test_column_transformer_passthrough(self, binary_data):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = binary_data
        ct = ColumnTransformer([
            ("std",  StandardScaler(),  [0, 1, 2]),
            ("pass", "passthrough",     [3, 4, 5]),
        ])
        pipe = Pipeline([
            ("prep", ct),
            ("clf",  RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ct_node = next(n for n in m.nodes if n.op == "Composite")
        assert ct_node.composite is not None
        inner_ops = [n.op for n in ct_node.composite.nodes]
        assert "TakeSlots"      in inner_ops
        assert "StandardScaler" in inner_ops
        assert "Concat"         in inner_ops

    def test_multi_step_pipeline(self, binary_data):
        """Pipeline with multiple preprocessing steps is fully wired."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = binary_data
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale",  StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Imputer"        in ops
        assert "StandardScaler" in ops
        assert "TreeEnsemble"   in ops
        # Nodes are in order: impute → scale → ensemble
        assert ops.index("Imputer") < ops.index("StandardScaler") < ops.index("TreeEnsemble")

    def test_pipeline_namespace_imports(self, pipeline_rf):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        ns = {ni.namespace for ni in m.operator_imports}
        assert "omle.feature" in ns
        assert "omle.ml"      in ns

    def test_json_roundtrip_with_preprocessing(self, pipeline_rf, tmp_path):
        m = from_sklearn(pipeline_rf)
        _assert_std_io(m, task="binary")
        p = tmp_path / "pipe.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        ops = [n.op for n in loaded.nodes]
        assert "StandardScaler" in ops
        assert "TreeEnsemble"   in ops

    def test_runtime_standard_scaler_lr_binary_matches_native(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = binary_data
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=200, random_state=0))]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), pipe.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_standard_scaler_lr_multiclass_matches_native(self, multiclass_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = multiclass_data
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=200, random_state=0))]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), pipe.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_standard_scaler_linear_regression_matches_native(self, regression_data):
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = regression_data
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_minmax_scaler_rf_binary_matches_native(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        X, y = binary_data
        pipe = Pipeline([("mm", MinMaxScaler()), ("rf", RandomForestClassifier(n_estimators=5, random_state=0))]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), pipe.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_pca_rf_binary_matches_native(self, binary_data):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        X, y = binary_data
        pipe = Pipeline([("pca", PCA(n_components=4, random_state=0)), ("rf", RandomForestClassifier(n_estimators=5, random_state=0))]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), pipe.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_standard_scaler_svc_binary_matches_native(self, binary_data):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        X, y = binary_data
        pipe = Pipeline([("sc", StandardScaler()), ("svc", SVC(probability=True, random_state=0))]).fit(X, y)
        # atol=0.005: libsvm iterative multiclass_probability with eps=0.0025
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X, verify_atol=0.005, verify_rtol=0.005)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), pipe.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=0.005, atol=0.005)

    def test_runtime_standard_scaler_mlp_regression_matches_native(self, regression_data):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = regression_data
        pipe = Pipeline([("sc", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(16,), max_iter=200, random_state=0))]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_robust_scaler_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        X, y = regression_data
        pipe = Pipeline([("t", RobustScaler()), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_maxabs_scaler_binary_matches_native(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MaxAbsScaler
        X, y = binary_data
        pipe = Pipeline([("t", MaxAbsScaler()), ("m", LogisticRegression(max_iter=200, random_state=0))]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_normalizer_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer
        X, y = regression_data
        pipe = Pipeline([("t", Normalizer()), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_simple_imputer_regression_matches_native(self, regression_data):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        X, y = regression_data
        Xnan = X.copy().astype(float)
        Xnan[::5, 0] = np.nan
        pipe = Pipeline([("t", SimpleImputer()), ("m", Ridge())]).fit(Xnan, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(Xnan), m_x.predict(Xnan), rtol=1e-6)
        np.testing.assert_allclose(m.predict(Xnan), pipe.predict(Xnan), rtol=1e-4, atol=1e-4)

    def test_runtime_binarizer_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Binarizer
        X, y = regression_data
        pipe = Pipeline([("t", Binarizer(threshold=0.0)), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_power_transformer_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PowerTransformer
        X, y = regression_data
        Xpos = np.abs(X) + 0.1
        pipe = Pipeline([("t", PowerTransformer()), ("m", Ridge())]).fit(Xpos, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(Xpos), m_x.predict(Xpos), rtol=1e-6)
        np.testing.assert_allclose(m.predict(Xpos), pipe.predict(Xpos), rtol=1e-4, atol=1e-4)

    def test_runtime_quantile_transformer_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import QuantileTransformer
        X, y = regression_data
        pipe = Pipeline([("t", QuantileTransformer(random_state=0)), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_polynomial_features_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        X, y = regression_data
        pipe = Pipeline([("t", PolynomialFeatures(degree=2)), ("m", Ridge())]).fit(X[:, :3], y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression", n_features=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X[:, :3])
        _assert_std_io(_ir_x, task="regression", n_features=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X[:, :3]), m_x.predict(X[:, :3]), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X[:, :3]), pipe.predict(X[:, :3]), rtol=1e-4, atol=1e-4)

    def test_runtime_spline_transformer_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import SplineTransformer
        X, y = regression_data
        pipe = Pipeline([("t", SplineTransformer(n_knots=4)), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_truncated_svd_regression_matches_native(self, regression_data):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        X, y = regression_data
        pipe = Pipeline([("t", TruncatedSVD(n_components=3, random_state=0)), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_ohe_binary_matches_native(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        rng = np.random.default_rng(42)
        X = rng.integers(0, 4, size=(200, 4)).astype(float)
        y = (X[:, 0] > 1).astype(int)
        pipe = Pipeline([
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ("lr", LogisticRegression(max_iter=200, random_state=0)),
        ]).fit(X.astype(str), y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary", n_features=4)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X.astype(str))
        _assert_std_io(_ir_x, task="binary", n_features=4)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X.astype(str)), m_x.predict_proba(X.astype(str)), rtol=1e-6)
        np.testing.assert_allclose(
            m.predict_proba(X.astype(str)),
            pipe.predict_proba(X.astype(str)), rtol=1e-4, atol=1e-4
        )

    def test_runtime_ordinal_encoder_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder
        rng = np.random.default_rng(42)
        X = rng.integers(0, 4, size=(200, 4)).astype(float)
        y = X[:, 0] * 2.0 + rng.standard_normal(200)
        pipe = Pipeline([("t", OrdinalEncoder()), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression", n_features=4)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", n_features=4, input_dtype=omle.DataType.FLOAT64)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_kbins_ordinal_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import KBinsDiscretizer
        X, y = regression_data
        pipe = Pipeline([("t", KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_kbins_onehot_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import KBinsDiscretizer
        X, y = regression_data
        pipe = Pipeline([("t", KBinsDiscretizer(n_bins=3, encode="onehot-dense", subsample=None)), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_column_transformer_binary_matches_native(self, binary_data):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = binary_data
        pipe = Pipeline([
            ("ct", ColumnTransformer([
                ("sc", StandardScaler(), [0, 1, 2]),
                ("pass", "passthrough", [3, 4, 5]),
            ])),
            ("lr", LogisticRegression(max_iter=200, random_state=0)),
        ]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_feature_union_regression_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        X, y = regression_data
        pipe = Pipeline([
            ("union", FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])),
            ("m", Ridge()),
        ]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_select_kbest_binary_matches_native(self, binary_data):
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        X, y = binary_data
        pipe = Pipeline([
            ("sel", SelectKBest(f_classif, k=4)),
            ("lr", LogisticRegression(max_iter=200, random_state=0)),
        ]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict_proba(X), pipe.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_variance_threshold_regression_matches_native(self, regression_data):
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        X, y = regression_data
        pipe = Pipeline([("t", VarianceThreshold(threshold=0.0)), ("m", Ridge())]).fit(X, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), pipe.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_function_transformer_log_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        X, y = regression_data
        Xpos = np.abs(X) + 0.1
        pipe = Pipeline([("ft", FunctionTransformer(np.log)), ("m", Ridge())]).fit(Xpos, y)
        _ir = from_sklearn(pipe)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(pipe, X=Xpos)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(Xpos), m_x.predict(Xpos), rtol=1e-6)
        np.testing.assert_allclose(m.predict(Xpos), pipe.predict(Xpos), rtol=1e-4, atol=1e-4)

    def test_runtime_json_roundtrip_linear_pipeline_matches_native(self, binary_data, tmp_path):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = binary_data
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=200, random_state=0))]).fit(X, y)
        ir = from_sklearn(pipe)
        _assert_std_io(ir, task="binary")
        p = tmp_path / "lr_pipe.json"
        omle.save_json(ir, p)
        ir2 = omle.load_json(p)
        _assert_std_io(ir2, task="binary")
        r1 = _load_runtime(ir).predict(X)
        r2 = _load_runtime(ir2).predict(X)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)


# ── Tree structure ────────────────────────────────────────────────────────────

class TestSklearnTreeStructure:
    def test_tree_flat_arrays_consistent(self, rf_binary):
        # Binary RF uses leaf_vector (width 2) so leaf_value is absent.
        m = from_sklearn(rf_binary)
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
            assert len(tree.leaf_vector_index) == n
            n_leaves = sum(1 for k in tree.node_kind if k == TreeNodeKind.LEAF)
            assert len(_tv_float64(tree.leaf_vector, m)) == n_leaves * tree.leaf_width

    def test_leaf_nodes_have_zero_children(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.LEAF:
                    assert tree.children_count[i] == 0

    def test_branch_nodes_have_two_children(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.BRANCH:
                    assert tree.children_count[i] == 2

    def test_branch_split_op_less_or_equal(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.BRANCH:
                    assert tree.split_op[i] == TreeSplitOp.LESS_OR_EQUAL

    def test_children_index_within_range(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n = tree.num_nodes
            for c in tree.children_index:
                assert 0 <= c < n

    def test_leaves_one_more_than_branches(self, rf_binary):
        m = from_sklearn(rf_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n_leaves = sum(1 for k in tree.node_kind if k == TreeNodeKind.LEAF)
            n_branch = sum(1 for k in tree.node_kind if k == TreeNodeKind.BRANCH)
            assert n_leaves == n_branch + 1

    def test_hgb_tree_structure(self, hgb_binary):
        m = from_sklearn(hgb_binary)
        _assert_std_io(m, task="binary")
        for tree in m.nodes[0].tree_ensemble.trees:
            n = tree.num_nodes
            assert len(tree.node_kind) == n
            assert len(tree.children_offset) == n
            for i, kind in enumerate(tree.node_kind):
                if kind == TreeNodeKind.BRANCH:
                    assert tree.split_op[i] == TreeSplitOp.LESS_OR_EQUAL


# ── Bagging ───────────────────────────────────────────────────────────────────

class TestBagging:
    def test_bagging_classifier_binary(self, binary_data):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        m = from_sklearn(
            BaggingClassifier(DecisionTreeClassifier(max_depth=3),
                              n_estimators=5, random_state=0).fit(X, y)
        )
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Tree" in ops
        assert "SoftVote" in ops
        soft_node = next(n for n in m.nodes if n.op == "SoftVote")
        assert len(soft_node.inputs) == 5

    def test_bagging_classifier_multiclass(self, multiclass_data):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        m = from_sklearn(
            BaggingClassifier(DecisionTreeClassifier(max_depth=3),
                              n_estimators=4, random_state=0).fit(X, y)
        )
        _assert_std_io(m, task="multiclass", n_classes=3)
        ops = [n.op for n in m.nodes]
        assert "SoftVote" in ops

    def test_bagging_regressor(self, regression_data):
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        m = from_sklearn(
            BaggingRegressor(DecisionTreeRegressor(max_depth=3),
                             n_estimators=5, random_state=0).fit(X, y)
        )
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "Tree" in ops
        assert "Average" in ops
        avg_node = next(n for n in m.nodes if n.op == "Average")
        assert len(avg_node.inputs) == 5

    def test_bagging_feature_subset_emits_take_slots(self, binary_data):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        m = from_sklearn(
            BaggingClassifier(DecisionTreeClassifier(max_depth=3),
                              n_estimators=4, max_features=0.5, random_state=0).fit(X, y)
        )
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "TakeSlots" in ops
        assert "SoftVote" in ops

    def test_bagging_with_linear_base(self, binary_data):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        m = from_sklearn(
            BaggingClassifier(LogisticRegression(random_state=0, max_iter=200),
                              n_estimators=3, random_state=0).fit(X, y)
        )
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Linear" in ops
        assert "SoftVote" in ops

    def test_runtime_bagging_regressor_matches_native(self, regression_data):
        from sklearn.ensemble import BaggingRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        est = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=3), n_estimators=5, random_state=0
        ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_bagging_classifier_multiclass_matches_native(self, multiclass_data):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        est = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3), n_estimators=5, random_state=0
        ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── New operator converters ────────────────────────────────────────────────────

class TestFeatureSelection:
    """All feature selectors → TakeSlots with get_support(indices=True)."""

    def _check(self, selector, data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        X, y = data
        pipe = Pipeline([
            ("sel", selector),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "TakeSlots" in ops
        ts = next(n for n in m.nodes if n.op == "TakeSlots")
        idx_attr = next(a for a in ts.attributes if a.name == "indices")
        expected = selector.get_support(indices=True).tolist()
        assert idx_attr.ints == expected

    def test_select_k_best(self, binary_data):
        from sklearn.feature_selection import SelectKBest, f_classif
        self._check(SelectKBest(f_classif, k=4), binary_data)

    def test_variance_threshold(self, binary_data):
        from sklearn.feature_selection import VarianceThreshold
        self._check(VarianceThreshold(threshold=0.0), binary_data)

    def test_select_percentile(self, binary_data):
        from sklearn.feature_selection import SelectPercentile, f_classif
        self._check(SelectPercentile(f_classif, percentile=50), binary_data)

    def test_select_from_model(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        X, y = binary_data
        base = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        self._check(SelectFromModel(base, prefit=True), binary_data)

    def test_rfe(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE
        self._check(RFE(RandomForestClassifier(n_estimators=5, random_state=0), n_features_to_select=4), binary_data)


class TestMultiLabelBinarizer:
    @pytest.fixture(scope="class")
    def mlb_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
        X_raw = [["cat", "dog"], ["bird"], ["cat", "bird"], ["dog"], ["cat"], ["bird", "dog"]]
        y = [0, 1, 0, 1, 0, 1]
        mlb = MultiLabelBinarizer(classes=["cat", "dog", "bird"]).fit(X_raw)
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(mlb.transform(X_raw), y)
        pipe = Pipeline(steps=[("mlb", mlb), ("clf", clf)])
        return from_sklearn(pipe)

    def test_op_emitted(self, mlb_model):
        assert any(n.op == "MultiLabelBinarizer" for n in mlb_model.nodes)

    def test_classes_stored(self, mlb_model):
        node = next(n for n in mlb_model.nodes if n.op == "MultiLabelBinarizer")
        classes_attr = next(a for a in node.attributes if a.name == "classes")
        assert len(_tensor_strings(mlb_model.tensor_entries, classes_attr)) == 3


class TestLabelBinarizer:
    def _make_model(self, classes, neg_label=0, pos_label=1):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer(neg_label=neg_label, pos_label=pos_label).fit(classes)
        X = lb.transform(classes)
        y = [i % 2 for i in range(len(classes))]
        clf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        return from_sklearn(Pipeline(steps=[("lb", lb), ("clf", clf)]))

    def test_label_binarize_op_emitted(self):
        m = self._make_model([0, 1, 2])
        assert any(n.op == "LabelBinarizer" for n in m.nodes)

    def test_label_binarize_classes_stored(self):
        m = self._make_model([0, 1, 2])
        node = next(n for n in m.nodes if n.op == "LabelBinarizer")
        classes_attr = next(a for a in node.attributes if a.name == "classes")
        assert len(_tensor_strings(m.tensor_entries, classes_attr)) == 3

    def test_label_binarize_neg_pos_labels(self):
        m = self._make_model([0, 1], neg_label=-1, pos_label=2)
        node = next(n for n in m.nodes if n.op == "LabelBinarizer")
        neg = next(a for a in node.attributes if a.name == "neg_label")
        pos = next(a for a in node.attributes if a.name == "pos_label")
        assert neg.i == -1
        assert pos.i == 2


def _import_target_encoder():
    try:
        from sklearn.preprocessing import TargetEncoder
        return TargetEncoder
    except ImportError:
        return None


class TestTargetEncoder:
    @pytest.fixture(scope="class")
    def target_encoder_pipe(self, binary_data):
        te_cls = _import_target_encoder()
        if te_cls is None:
            pytest.skip("TargetEncoder requires sklearn >= 1.3")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        X, y = binary_data
        pipe = Pipeline([
            ("te", te_cls(random_state=0)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        return from_sklearn(pipe)

    def test_target_encode_op_emitted(self, target_encoder_pipe):
        _assert_std_io(target_encoder_pipe, task="binary")
        ops = [n.op for n in target_encoder_pipe.nodes]
        assert "TargetEncoder" in ops

    def test_target_encode_has_tensor_attrs(self, target_encoder_pipe):
        te_node = next(n for n in target_encoder_pipe.nodes if n.op == "TargetEncoder")
        attr_names = {a.name for a in te_node.attributes}
        assert "categories" in attr_names
        assert "category_offsets" in attr_names
        assert "encoded_values" in attr_names
        assert "default_values" in attr_names
        assert "target_kind" in attr_names

    def test_target_encode_single_node(self, target_encoder_pipe):
        te_nodes = [n for n in target_encoder_pipe.nodes if n.op == "TargetEncoder"]
        assert len(te_nodes) == 1, "expected exactly one TargetEncoder node for matrix input"

    def test_target_encode_offsets_shape(self, target_encoder_pipe):
        te_node = next(n for n in target_encoder_pipe.nodes if n.op == "TargetEncoder")
        offsets_attr = next(a for a in te_node.attributes if a.name == "category_offsets")
        offsets = _attr_int64_data(target_encoder_pipe.tensor_entries, offsets_attr)
        cats_attr = next(a for a in te_node.attributes if a.name == "categories")
        total_cats = len(_tensor_strings(target_encoder_pipe.tensor_entries, cats_attr))
        assert offsets[0] == 0
        assert offsets[-1] == total_cats

    def test_target_encode_namespace_imported(self, target_encoder_pipe):
        ns = {ni.namespace for ni in target_encoder_pipe.operator_imports}
        assert "omle.feature" in ns


class TestTextTransformers:
    def _text_data(self):
        docs = [
            "the quick brown fox",
            "jumps over the lazy dog",
            "the dog barked loudly",
            "a quick brown dog",
        ]
        return docs

    def test_count_vectorizer(self):
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        docs = self._text_data()
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("cv", CountVectorizer()),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ]).fit(docs, y)
        m = from_sklearn(pipe)
        ops = [n.op for n in m.nodes]
        assert "RegexTokenizer" in ops
        assert "CountVectorizer" in ops
        assert len(m.inputs) == 1
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        text_feat = m.model_schema.features[0]
        assert text_feat.type.dtype == omle.DataType.STRING
        assert text_feat.measure_level == omle.MeasureLevel.NOMINAL

    def test_count_vectorizer_vocab_stored(self):
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        docs = self._text_data()
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("cv", CountVectorizer()),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ]).fit(docs, y)
        m = from_sklearn(pipe)
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        cv_node = next(n for n in m.nodes if n.op == "CountVectorizer")
        vocab_attr = next(a for a in cv_node.attributes if a.name == "vocabulary")
        assert _attr_has_tensor(vocab_attr)
        assert len(_tensor_strings(m.tensor_entries, vocab_attr)) > 0

    def test_tfidf_transformer(self):
        import numpy as np
        from sklearn.feature_extraction.text import (
            CountVectorizer,
            TfidfTransformer,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        docs = self._text_data()
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("cv", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ]).fit(docs, y)
        m = from_sklearn(pipe)
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        ops = [n.op for n in m.nodes]
        assert "TfIdfTransformer" in ops

    def test_tfidf_vectorizer(self):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        docs = self._text_data()
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ]).fit(docs, y)
        m = from_sklearn(pipe)
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        ops = [n.op for n in m.nodes]
        assert "RegexTokenizer" in ops
        assert "CountVectorizer" in ops
        assert "TfIdfTransformer" in ops

    def test_hashing_vectorizer(self):
        import numpy as np
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        docs = self._text_data()
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("hv", HashingVectorizer(n_features=512)),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ]).fit(docs, y)
        m = from_sklearn(pipe)
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        ops = [n.op for n in m.nodes]
        assert "RegexTokenizer" in ops
        assert "HashingVectorizer" in ops
        hv_node = next(n for n in m.nodes if n.op == "HashingVectorizer")
        nf_attr = next(a for a in hv_node.attributes if a.name == "num_features")
        assert nf_attr.i == 512

    def test_text_namespace_imported(self):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        docs = self._text_data()
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ]).fit(docs, y)
        m = from_sklearn(pipe)
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        ns = {ni.namespace for ni in m.operator_imports}
        assert "omle.text" in ns

    def test_ngram_range(self):
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        docs = self._text_data()
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ("cv", CountVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(random_state=0, max_iter=200)),
        ]).fit(docs, y)
        m = from_sklearn(pipe)
        assert m.inputs[0].type.dtype == omle.DataType.STRING
        assert m.inputs[0].type.shape == [-1]
        ops = [n.op for n in m.nodes]
        assert "NGram" in ops
        ng_node = next(n for n in m.nodes if n.op == "NGram")
        n_max_attr = next(a for a in ng_node.attributes if a.name == "n_max")
        assert n_max_attr.i == 2


class TestKBinsDiscretizerEncoding:
    def _make_pipe(self, transformer, data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        X, y = data
        pipe = Pipeline([
            ("t", transformer),
            ("clf", LogisticRegression(random_state=0, max_iter=500)),
        ]).fit(X, y)
        return from_sklearn(pipe)

    def test_onehot_dense_single_feature(self, binary_data):
        from sklearn.preprocessing import KBinsDiscretizer
        X, y = binary_data
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        kbd = KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform")
        pipe = Pipeline([("t", kbd), ("clf", LogisticRegression(random_state=0, max_iter=500))]).fit(X[:, :1], y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary", n_features=1)
        ops = [n.op for n in m.nodes]
        assert "Bucketizer" in ops
        assert "OneHotEncoder" in ops
        assert "DenseToSparse" not in ops

    def test_onehot_dense_multi_feature(self, binary_data):
        from sklearn.preprocessing import KBinsDiscretizer
        X, y = binary_data
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        kbd = KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="uniform")
        pipe = Pipeline([("t", kbd), ("clf", LogisticRegression(random_state=0, max_iter=500))]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Bucketizer" in ops
        assert "OneHotEncoder" in ops
        assert "Concat" not in ops  # multi-feature handled natively by Bucketizer boundary_offsets
        assert "DenseToSparse" not in ops

    def test_onehot_sparse_emits_dense_to_sparse(self, binary_data):
        from sklearn.preprocessing import KBinsDiscretizer
        X, y = binary_data
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        kbd = KBinsDiscretizer(n_bins=3, encode="onehot", strategy="uniform")
        pipe = Pipeline([("t", kbd), ("clf", LogisticRegression(random_state=0, max_iter=500))]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Bucketizer" in ops
        assert "OneHotEncoder" in ops
        assert "DenseToSparse" in ops

    def test_ohe_categories_match_n_bins(self, binary_data):
        from sklearn.preprocessing import KBinsDiscretizer
        X, y = binary_data
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        kbd = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="uniform")
        pipe = Pipeline([("t", kbd), ("clf", LogisticRegression(random_state=0, max_iter=500))]).fit(X[:, :1], y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary", n_features=1)
        ohe_node = next(n for n in m.nodes if n.op == "OneHotEncoder")
        cats_attr = next(a for a in ohe_node.attributes if a.name == "categories")
        assert _tensor_strings(m.tensor_entries, cats_attr) == ["0", "1", "2", "3"]


class TestNewFeatureTransformers:
    def _make_pipe(self, transformer, estimator_cls, data):
        from sklearn.pipeline import Pipeline
        X, y = data
        pipe = Pipeline([
            ("t", transformer),
            ("est", estimator_cls(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        return from_sklearn(pipe)

    def test_power_transformer_yeo_johnson(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PowerTransformer
        m = self._make_pipe(PowerTransformer(method="yeo-johnson"), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "PowerTransformer" in ops
        node = next(n for n in m.nodes if n.op == "PowerTransformer")
        method_attr = next(a for a in node.attributes if a.name == "method")
        assert method_attr.s == "yeo_johnson"

    def test_power_transformer_stores_lambdas(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PowerTransformer
        m = self._make_pipe(PowerTransformer(), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        node = next(n for n in m.nodes if n.op == "PowerTransformer")
        lambdas_attr = next((a for a in node.attributes if a.name == "lambdas"), None)
        assert lambdas_attr is not None
        assert _attr_has_tensor(lambdas_attr)

    def test_quantile_transformer(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import QuantileTransformer
        m = self._make_pipe(QuantileTransformer(n_quantiles=50, random_state=0), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "QuantileTransformer" in ops
        node = next(n for n in m.nodes if n.op == "QuantileTransformer")
        attr_names = {a.name for a in node.attributes}
        assert "quantiles" in attr_names
        assert "references" in attr_names
        assert "output_distribution" in attr_names

    def test_spline_transformer(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import SplineTransformer
        m = self._make_pipe(SplineTransformer(n_knots=4, degree=3), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "SplineTransformer" in ops
        node = next(n for n in m.nodes if n.op == "SplineTransformer")
        attr_names = {a.name for a in node.attributes}
        assert "knots" in attr_names
        assert "degree" in attr_names

    def test_polynomial_features(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import PolynomialFeatures
        m = self._make_pipe(PolynomialFeatures(degree=2), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "PolynomialFeatures" in ops
        node = next(n for n in m.nodes if n.op == "PolynomialFeatures")
        max_deg_attr = next(a for a in node.attributes if a.name == "max_degree")
        assert max_deg_attr.i == 2


class TestDimensionalityReduction:
    def _make_pipe(self, transformer, estimator_cls, data):
        from sklearn.pipeline import Pipeline
        X, y = data
        pipe = Pipeline([
            ("t", transformer),
            ("est", estimator_cls(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        return from_sklearn(pipe)

    def test_truncated_svd(self, regression_data):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.ensemble import RandomForestRegressor
        m = self._make_pipe(TruncatedSVD(n_components=3, random_state=0), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "TruncatedSVD" in ops
        node = next(n for n in m.nodes if n.op == "TruncatedSVD")
        assert any(a.name == "components" for a in node.attributes)

    def test_fast_ica(self, regression_data):
        from sklearn.decomposition import FastICA
        from sklearn.ensemble import RandomForestRegressor
        m = self._make_pipe(FastICA(n_components=3, random_state=0, max_iter=500), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "FastICA" in ops
        node = next(n for n in m.nodes if n.op == "FastICA")
        assert any(a.name == "components" for a in node.attributes)

    def test_factor_analysis(self, regression_data):
        from sklearn.decomposition import FactorAnalysis
        from sklearn.ensemble import RandomForestRegressor
        m = self._make_pipe(FactorAnalysis(n_components=3, random_state=0), RandomForestRegressor, regression_data)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "FactorAnalysis" in ops
        node = next(n for n in m.nodes if n.op == "FactorAnalysis")
        attr_names = {a.name for a in node.attributes}
        assert "mean" in attr_names
        assert "components" in attr_names

    def test_nmf(self, regression_data):
        from sklearn.decomposition import NMF
        from sklearn.ensemble import RandomForestRegressor
        X, y = regression_data
        X_pos = np.abs(X)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("t", NMF(n_components=3, random_state=0, max_iter=500)),
            ("est", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X_pos, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "NMF" in ops

    def test_kernel_pca(self, binary_data):
        from sklearn.decomposition import KernelPCA
        from sklearn.ensemble import RandomForestClassifier
        m = self._make_pipe(KernelPCA(n_components=3, kernel="rbf", random_state=0), RandomForestClassifier, binary_data)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "KernelPCA" in ops
        node = next(n for n in m.nodes if n.op == "KernelPCA")
        attr_names = {a.name for a in node.attributes}
        assert "fit_samples" in attr_names
        assert "dual_components" in attr_names
        assert "kernel" in attr_names


class TestImputationExtras:
    def test_missing_indicator_missing_only(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import MissingIndicator
        from sklearn.pipeline import Pipeline
        X, y = binary_data
        X_nan = X.copy().astype(float)
        X_nan[::5, 0] = float("nan")
        mi = MissingIndicator(features="missing-only")
        pipe = Pipeline([
            ("mi", mi),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X_nan, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "MissingIndicator" in ops
        node = next(n for n in m.nodes if n.op == "MissingIndicator")
        mode_attr = next(a for a in node.attributes if a.name == "features_mode")
        assert mode_attr.s == "missing_only"

    def test_knn_imputer(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import KNNImputer
        from sklearn.pipeline import Pipeline
        X, y = binary_data
        X_nan = X.copy().astype(float)
        X_nan[::5, 0] = float("nan")
        pipe = Pipeline([
            ("imp", KNNImputer(n_neighbors=3)),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X_nan, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "KNNImputer" in ops
        node = next(n for n in m.nodes if n.op == "KNNImputer")
        attr_names = {a.name for a in node.attributes}
        assert "train_features" in attr_names
        assert "n_neighbors" in attr_names


class TestAdaBoost:
    def test_adaboost_samme_binary(self, binary_data):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=5,
            random_state=0,
        ).fit(X, y)
        m = from_sklearn(clf, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        ops = [n.op for n in m.nodes]
        assert "Tree" in ops
        # DT supports predict_proba → SAMMEVote emitted for y_prob + y_pred
        assert "SAMMEVote" in ops
        sv_node = next(n for n in m.nodes if n.op == "SAMMEVote")
        assert len(sv_node.inputs) == 5
        w_attr = next(a for a in sv_node.attributes if a.name == "weights")
        assert len(w_attr.float64s) == 5
        out_roles = {o.name for o in sv_node.outputs}
        assert "y_prob" in out_roles
        assert "y_pred" in out_roles

    def test_adaboost_samme_multiclass(self, multiclass_data):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=4,
            random_state=0,
        ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="multiclass", n_classes=3)
        ops = [n.op for n in m.nodes]
        assert "SAMMEVote" in ops
        # y_prob output shape must reflect actual number of classes
        y_prob_spec = next(s for s in m.outputs if s.name == "y_prob")
        assert y_prob_spec.type.shape == [-1, 3]

    def test_adaboost_weights_stored(self, binary_data):
        import numpy as np
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=3,
            random_state=0,
        ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        agg_node = next(n for n in m.nodes if n.op in ("SAMMEVote", "WeightedMajorityVote"))
        w_attr = next(a for a in agg_node.attributes if a.name == "weights")
        np.testing.assert_allclose(w_attr.float64s, clf.estimator_weights_.tolist(), rtol=1e-5)

    def test_adaboost_default_algorithm(self, binary_data):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=3,
                random_state=0,
            ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        agg_ops = {"SAMMEVote", "WeightedMajorityVote"}
        assert any(n.op in agg_ops for n in m.nodes)

    def test_runtime_adaboost_binary_matches_native(self, binary_data):
        import warnings

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=5, random_state=0,
            ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_adaboost_multiclass_matches_native(self, multiclass_data):
        import warnings

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2),
                n_estimators=4, random_state=0,
            ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── Mixed DataFrame ───────────────────────────────────────────────────────────

class TestMixedDataFrame:
    """Pipelines on a DataFrame with int, float, bool, and string columns."""

    def _ct_pipe(self, estimator, df, y, *, num_cols, cat_cols, bool_cols=None,
                 cat_enc="ordinal", ohe_sparse=False):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import (
            OneHotEncoder,
            OrdinalEncoder,
            StandardScaler,
        )

        transformers = [("num", StandardScaler(), num_cols)]
        if cat_enc == "ordinal":
            transformers.append(("cat", OrdinalEncoder(), cat_cols))
        else:
            transformers.append(("cat", OneHotEncoder(sparse_output=ohe_sparse, handle_unknown="ignore"), cat_cols))
        if bool_cols:
            transformers.append(("bool", "passthrough", bool_cols))

        pipe = Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("est",  estimator),
        ]).fit(df, y)
        return from_sklearn(pipe)

    # ── basic structure ───────────────────────────────────────────────────────

    _MIXED_FEAT_NAMES = ["age", "education", "income", "stock", "active", "gender", "segment"]

    def _assert_mixed_io(self, m, *, task, n_classes=2):
        """Assert IO invariants for mixed-df pipelines (7-column df, no X passed)."""
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 7]
        feat_names = [f.expand_names()[0] for f in m.model_schema.features]
        assert feat_names == self._MIXED_FEAT_NAMES
        for feat in m.model_schema.features:
            assert feat.type.dtype == omle.DataType.FLOAT64
            assert feat.type.shape == [-1]
        if task == "binary":
            pred = next(o for o in m.outputs if o.name == "y_pred")
            prob = next(o for o in m.outputs if o.name == "y_prob")
            assert pred.type.dtype == omle.DataType.INT64
            assert pred.type.shape == [-1]
            assert prob.type.dtype == omle.DataType.FLOAT64
            assert prob.type.shape == [-1, 2]
            assert m.model_schema.targets[0].kind == TargetKind.BINARY
        elif task == "multiclass":
            pred = next(o for o in m.outputs if o.name == "y_pred")
            prob = next(o for o in m.outputs if o.name == "y_prob")
            assert pred.type.dtype == omle.DataType.INT64
            assert pred.type.shape == [-1]
            assert prob.type.dtype == omle.DataType.FLOAT64
            assert prob.type.shape == [-1, n_classes]
            assert m.model_schema.targets[0].kind == TargetKind.MULTICLASS
        elif task == "regression":
            pred = next(o for o in m.outputs if o.name == "y_pred")
            assert pred.type.dtype == omle.DataType.FLOAT64
            assert pred.type.shape == [-1]
            assert m.model_schema.targets[0].kind == TargetKind.REGRESSION

    def test_returns_omle_model(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        assert isinstance(m, omle.OMLEModel)
        self._assert_mixed_io(m, task="binary")

    def test_has_standard_scaler_node(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        assert "StandardScaler" in [n.op for n in _all_nodes(m.nodes)]

    def test_has_ordinal_encoder_node(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        assert "OrdinalEncoder" in [n.op for n in _all_nodes(m.nodes)]

    def test_has_concat_node(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        assert "Concat" in [n.op for n in _all_nodes(m.nodes)]

    def test_named_cols_use_take_slots_with_names(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        ops = [n.op for n in _all_nodes(m.nodes)]
        assert "TakeSlots" in ops
        assert "Concat" in ops
        # TakeSlots nodes for named columns carry string names, not integer indices.
        ts_nodes = [n for n in _all_nodes(m.nodes) if n.op == "TakeSlots"]
        assert any(
            any(a.name == "names" for a in ts.attributes)
            for ts in ts_nodes
        )

    # ── OrdinalEncoder categories stored as string tensors ───────────────────

    def test_ordinal_encoder_categories_stored(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        oe_nodes = [n for n in _all_nodes(m.nodes) if n.op == "OrdinalEncoder"]
        assert len(oe_nodes) > 0
        for node in oe_nodes:
            cats_attr = next(a for a in node.attributes if a.name == "categories")
            assert len(_tensor_strings(m.tensor_entries, cats_attr)) > 0

    def test_ordinal_encoder_gender_categories(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender"],
        )
        self._assert_mixed_io(m, task="binary")
        oe_node = next(n for n in _all_nodes(m.nodes) if n.op == "OrdinalEncoder")
        cats_attr = next(a for a in oe_node.attributes if a.name == "categories")
        assert set(_tensor_strings(m.tensor_entries, cats_attr)) == {"F", "M"}

    def test_ordinal_encoder_segment_four_categories(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["segment"],
        )
        self._assert_mixed_io(m, task="binary")
        oe_node = next(n for n in _all_nodes(m.nodes) if n.op == "OrdinalEncoder")
        cats_attr = next(a for a in oe_node.attributes if a.name == "categories")
        assert set(_tensor_strings(m.tensor_entries, cats_attr)) == {"low", "mid", "high", "premium"}

    # ── OneHotEncoder on string columns ──────────────────────────────────────

    def test_ohe_on_string_columns(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
            cat_enc="ohe",
        )
        self._assert_mixed_io(m, task="binary")
        assert "OneHotEncoder" in [n.op for n in _all_nodes(m.nodes)]

    def test_ohe_categories_stored_as_strings(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender"],
            cat_enc="ohe",
        )
        self._assert_mixed_io(m, task="binary")
        ohe_node = next(n for n in _all_nodes(m.nodes) if n.op == "OneHotEncoder")
        cats_attr = next(a for a in ohe_node.attributes if a.name == "categories")
        assert set(_tensor_strings(m.tensor_entries, cats_attr)) == {"F", "M"}

    # ── bool passthrough ─────────────────────────────────────────────────────

    def test_bool_passthrough_uses_take_slots_with_names(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
            bool_cols=["active"],
        )
        self._assert_mixed_io(m, task="binary")
        ops = [n.op for n in _all_nodes(m.nodes)]
        assert "TakeSlots" in ops
        assert "Concat" in ops

    # ── regression target ─────────────────────────────────────────────────────

    def test_regression_target_kind(self, mixed_df):
        from sklearn.ensemble import RandomForestRegressor
        df, y_reg, _, _ = mixed_df
        m = self._ct_pipe(
            RandomForestRegressor(n_estimators=5, random_state=0),
            df, y_reg,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="regression")

    # ── multiclass target ─────────────────────────────────────────────────────

    def test_multiclass_target_kind(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, _, y_multi = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_multi,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="multiclass", n_classes=4)

    # ── gradient boosting on mixed data ──────────────────────────────────────

    def test_gradient_boosting_binary(self, mixed_df):
        from sklearn.ensemble import GradientBoostingClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        te = m.nodes[-1].tree_ensemble
        assert te.post_transform == PostTransform.SIGMOID_BINARY
        assert te.aggregation == TreeAggregation.SUM

    # ── JSON roundtrip ────────────────────────────────────────────────────────

    def test_json_roundtrip(self, mixed_df, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        p = tmp_path / "mixed.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        self._assert_mixed_io(loaded, task="binary")
        ops = [n.op for n in _all_nodes(loaded.nodes)]
        assert "OrdinalEncoder" in ops
        assert "StandardScaler" in ops
        assert "TreeEnsemble" in ops

    def test_json_roundtrip_preserves_categories(self, mixed_df, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender"],
        )
        self._assert_mixed_io(m, task="binary")
        p = tmp_path / "mixed_str.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        self._assert_mixed_io(loaded, task="binary")
        oe_node = next(n for n in _all_nodes(loaded.nodes) if n.op == "OrdinalEncoder")
        cats_attr = next(a for a in oe_node.attributes if a.name == "categories")
        assert set(_tensor_strings(loaded.tensor_entries, cats_attr)) == {"F", "M"}

    # ── namespace imports ─────────────────────────────────────────────────────

    def test_feature_namespace_imported(self, mixed_df):
        from sklearn.ensemble import RandomForestClassifier
        df, _, y_binary, _ = mixed_df
        m = self._ct_pipe(
            RandomForestClassifier(n_estimators=5, random_state=0),
            df, y_binary,
            num_cols=["age", "income"],
            cat_cols=["gender", "segment"],
        )
        self._assert_mixed_io(m, task="binary")
        ns = {ni.namespace for ni in m.operator_imports}
        assert "omle.feature" in ns
        assert "omle.ml" in ns


# ── Auxiliary data (verification / warmup / sample inputs) ───────────────────

class TestSklearnAuxiliaryData:
    def test_no_auxiliary_by_default(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor)
        _assert_std_io(m, task="regression")
        assert m.verification is None
        assert m.warmup is None
        assert m.sample_inputs is None
        # tensor_entries may be non-empty when OMLE_INLINE_TENSOR_LIMIT forces external
        # body-parameter storage; only verify no verification/warmup data was added.

    def test_tensor_entries_populated(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert len(m.tensor_entries) > 0

    def test_verification_populated(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        assert len(m.verification.cases) == 1
        assert m.verification.tolerance is not None

    def test_warmup_populated(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup is not None
        assert m.warmup.cases[0].repeat == 3

    def test_sample_inputs_populated(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.sample_inputs is not None
        assert len(m.sample_inputs.cases) == 3

    def test_n_verify_zero_skips_verification(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X, n_verify=0)
        assert m.verification is None
        assert m.warmup is not None

    def test_n_warmup_zero_skips_warmup(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X, n_warmup=0)
        assert m.warmup is None
        assert m.sample_inputs is not None

    def test_n_sample_zero_skips_sample_inputs(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X, n_sample=0)
        assert m.sample_inputs is None

    def test_custom_n_sample(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X, n_sample=3)
        assert len(m.sample_inputs.cases) == 3

    def test_verify_inputs_reference_tensor_entries(self, rf_binary, binary_data):
        X, _ = binary_data
        m = from_sklearn(rf_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        entry_ids = {te.id for te in m.tensor_entries}
        case = m.verification.cases[0]
        assert all(r.id in entry_ids for r in case.inputs)
        assert all(r.id in entry_ids for r in case.expected_outputs)

    def test_regression_verify_output_name(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_pred" in out_names

    def test_binary_verify_output_name(self, rf_binary, binary_data):
        X, _ = binary_data
        m = from_sklearn(rf_binary, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_multiclass_verify_output_name(self, rf_multiclass, multiclass_data):
        X, _ = multiclass_data
        m = from_sklearn(rf_multiclass, X=X)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        out_names = {te.dense.name for te in m.tensor_entries if te.dense}
        assert "y_prob" in out_names

    def test_warmup_repeat_param(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X, n_warmup_repeat=7)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        assert m.warmup.cases[0].repeat == 7

    def test_tolerance_values(self, dt_regressor, regression_data):
        X, _ = regression_data
        m = from_sklearn(dt_regressor, X=X, verify_atol=1e-3, verify_rtol=2e-3)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT32)
        tol = m.verification.tolerance
        assert abs(tol.atol.float_value - 1e-3) < 1e-10
        assert abs(tol.rtol.float_value - 2e-3) < 1e-10

    def test_json_roundtrip_preserves_auxiliary(self, rf_binary, binary_data, tmp_path):
        X, _ = binary_data
        m = from_sklearn(rf_binary, X=X, n_verify=2, n_warmup=4, n_sample=2)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        p = tmp_path / "with_aux.json"
        omle.save_json(m, p)
        loaded = omle.load_json(p)
        assert loaded.verification is not None
        assert loaded.warmup is not None
        assert loaded.sample_inputs is not None
        assert len(loaded.sample_inputs.cases) == 2
        assert len(loaded.tensor_entries) == len(m.tensor_entries)

    def test_pipeline_auxiliary(self, pipeline_rf, binary_data):
        X, _ = binary_data
        m = from_sklearn(pipeline_rf, X=X)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT32)
        assert m.verification is not None
        assert m.warmup is not None
        assert m.sample_inputs is not None


# ── Linear models ─────────────────────────────────────────────────────────────

class TestLinearModels:
    """All sklearn linear models → Linear node with correct coefficients and post-transform."""

    def _check_linear_node(self, estimator):
        m = from_sklearn(estimator)
        ops = [n.op for n in m.nodes]
        assert "Linear" in ops
        node = next(n for n in m.nodes if n.op == "Linear")
        coef_tv = node.linear.coefficients
        if coef_tv.tensor_ref is not None:
            coef_entry = next(e for e in m.tensor_entries if e.id == coef_tv.tensor_ref.id)
            assert len(coef_entry.dense.float64_data) > 0
        else:
            assert coef_tv.tensor is not None
            assert len(coef_tv.tensor.float64_data) > 0
        return node

    def test_linear_regression(self, regression_data):
        from sklearn.linear_model import LinearRegression
        X, y = regression_data
        node = self._check_linear_node(LinearRegression().fit(X, y))
        assert node.linear.post_transform == PostTransform.POST_TRANSFORM_UNSPECIFIED

    def test_ridge_regression(self, regression_data):
        from sklearn.linear_model import Ridge
        X, y = regression_data
        self._check_linear_node(Ridge(alpha=1.0).fit(X, y))

    def test_lasso(self, regression_data):
        from sklearn.linear_model import Lasso
        X, y = regression_data
        self._check_linear_node(Lasso(alpha=0.01).fit(X, y))

    def test_elastic_net(self, regression_data):
        from sklearn.linear_model import ElasticNet
        X, y = regression_data
        self._check_linear_node(ElasticNet(alpha=0.01).fit(X, y))

    def test_lars(self, regression_data):
        from sklearn.linear_model import Lars
        X, y = regression_data
        self._check_linear_node(Lars().fit(X, y))

    def test_lasso_lars(self, regression_data):
        from sklearn.linear_model import LassoLars
        X, y = regression_data
        self._check_linear_node(LassoLars(alpha=0.01).fit(X, y))

    def test_bayesian_ridge_stores_covariance(self, regression_data):
        from sklearn.linear_model import BayesianRidge
        X, y = regression_data
        est = BayesianRidge().fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        node = next(n for n in m.nodes if n.op == "Linear")
        assert node.linear.weight_covariances is not None
        assert (node.linear.weight_covariances.tensor_ref is not None or
                node.linear.weight_covariances.tensor is not None)
        assert node.linear.noise_precision is not None

    def test_ard_regression(self, regression_data):
        from sklearn.linear_model import ARDRegression
        X, y = regression_data
        est = ARDRegression().fit(X, y)
        m = from_sklearn(est)
        node = next(n for n in m.nodes if n.op == "Linear")
        assert node.linear.weight_covariances is not None
        assert (node.linear.weight_covariances.tensor_ref is not None or
                node.linear.weight_covariances.tensor is not None)

    def test_multitask_lasso(self, regression_data):
        from sklearn.linear_model import MultiTaskLasso
        X, y = regression_data
        # MultiTaskLasso requires 2-D target
        Y = np.stack([y, y * 0.5], axis=1)
        est = MultiTaskLasso(alpha=0.01).fit(X, Y)
        m = from_sklearn(est)
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 6]
        node = next(n for n in m.nodes if n.op == "Linear")
        coef_tv = node.linear.coefficients
        # coef_ shape is (n_targets, n_features) = (2, 6) → 12 values
        if coef_tv.tensor_ref is not None:
            coef_entry = next(e for e in m.tensor_entries if e.id == coef_tv.tensor_ref.id)
            assert len(coef_entry.dense.float64_data) == 2 * X.shape[1]
        else:
            assert len(coef_tv.tensor.float64_data) == 2 * X.shape[1]

    def test_multitask_elastic_net(self, regression_data):
        from sklearn.linear_model import MultiTaskElasticNet
        X, y = regression_data
        Y = np.stack([y, y * 0.5], axis=1)
        self._check_linear_node(MultiTaskElasticNet(alpha=0.01).fit(X, Y))

    def test_huber_regressor(self, regression_data):
        from sklearn.linear_model import HuberRegressor
        X, y = regression_data
        self._check_linear_node(HuberRegressor().fit(X, y))

    def test_theil_sen_regressor(self, regression_data):
        from sklearn.linear_model import TheilSenRegressor
        X, y = regression_data
        self._check_linear_node(TheilSenRegressor(random_state=0).fit(X, y))

    def test_quantile_regressor(self, regression_data):
        from sklearn.linear_model import QuantileRegressor
        X, y = regression_data
        self._check_linear_node(QuantileRegressor(solver="highs").fit(X, y))

    def test_orthogonal_matching_pursuit(self, regression_data):
        from sklearn.linear_model import OrthogonalMatchingPursuit
        X, y = regression_data
        self._check_linear_node(OrthogonalMatchingPursuit().fit(X, y))

    def test_sgd_regressor(self, regression_data):
        from sklearn.linear_model import SGDRegressor
        X, y = regression_data
        self._check_linear_node(SGDRegressor(random_state=0, max_iter=200).fit(X, y))

    def test_passive_aggressive_regressor(self, regression_data):
        from sklearn.linear_model import PassiveAggressiveRegressor
        X, y = regression_data
        self._check_linear_node(PassiveAggressiveRegressor(random_state=0, max_iter=200).fit(X, y))

    def test_logistic_regression_binary_sigmoid(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        node = self._check_linear_node(LogisticRegression(max_iter=200, random_state=0).fit(X, y))
        assert node.linear.post_transform == PostTransform.SIGMOID

    def test_logistic_regression_multiclass_softmax(self, multiclass_data):
        from sklearn.linear_model import LogisticRegression
        X, y = multiclass_data
        node = self._check_linear_node(LogisticRegression(max_iter=200, random_state=0).fit(X, y))
        assert node.linear.post_transform == PostTransform.SOFTMAX

    def test_ridge_classifier(self, binary_data):
        from sklearn.linear_model import RidgeClassifier
        X, y = binary_data
        self._check_linear_node(RidgeClassifier().fit(X, y))

    def test_ridge_classifier_cv(self, binary_data):
        from sklearn.linear_model import RidgeClassifierCV
        X, y = binary_data
        self._check_linear_node(RidgeClassifierCV().fit(X, y))

    def test_sgd_classifier(self, binary_data):
        from sklearn.linear_model import SGDClassifier
        X, y = binary_data
        self._check_linear_node(SGDClassifier(loss="log_loss", random_state=0, max_iter=200).fit(X, y))

    def test_passive_aggressive_classifier(self, binary_data):
        from sklearn.linear_model import PassiveAggressiveClassifier
        X, y = binary_data
        self._check_linear_node(PassiveAggressiveClassifier(random_state=0, max_iter=200).fit(X, y))

    def test_perceptron(self, binary_data):
        from sklearn.linear_model import Perceptron
        X, y = binary_data
        self._check_linear_node(Perceptron(random_state=0, max_iter=200).fit(X, y))

    def test_logistic_regression_cv(self, binary_data):
        from sklearn.linear_model import LogisticRegressionCV
        X, y = binary_data
        self._check_linear_node(LogisticRegressionCV(max_iter=200, random_state=0).fit(X, y))


    def test_runtime_linear_regression_matches_native(self, regression_data):
        from sklearn.linear_model import LinearRegression
        X, y = regression_data
        est = LinearRegression().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_ridge_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        X, y = regression_data
        est = Ridge(alpha=1.0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_lasso_matches_native(self, regression_data):
        from sklearn.linear_model import Lasso
        X, y = regression_data
        est = Lasso(alpha=0.01).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_elastic_net_matches_native(self, regression_data):
        from sklearn.linear_model import ElasticNet
        X, y = regression_data
        est = ElasticNet(alpha=0.01).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_logistic_regression_binary_matches_native(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        X, y = binary_data
        est = LogisticRegression(max_iter=200, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_logistic_regression_multiclass_matches_native(self, multiclass_data):
        from sklearn.linear_model import LogisticRegression
        X, y = multiclass_data
        est = LogisticRegression(max_iter=200, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_sgd_regressor_matches_native(self, regression_data):
        from sklearn.linear_model import SGDRegressor
        X, y = regression_data
        est = SGDRegressor(random_state=0, max_iter=500).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_ridge_classifier_binary_matches_native(self, binary_data):
        from sklearn.linear_model import RidgeClassifier
        X, y = binary_data
        est = RidgeClassifier().fit(X, y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.BINARY
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_huber_regressor_matches_native(self, regression_data):
        from sklearn.linear_model import HuberRegressor
        X, y = regression_data
        est = HuberRegressor().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_lars_matches_native(self, regression_data):
        from sklearn.linear_model import Lars
        X, y = regression_data
        est = Lars().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_bayesian_ridge_matches_native(self, regression_data):
        from sklearn.linear_model import BayesianRidge
        X, y = regression_data
        est = BayesianRidge().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_theil_sen_matches_native(self, regression_data):
        from sklearn.linear_model import TheilSenRegressor
        X, y = regression_data
        est = TheilSenRegressor(random_state=0).fit(X[:100], y[:100])
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X[:100]), m_x.predict(X[:100]), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X[:100]), est.predict(X[:100]), rtol=1e-4, atol=1e-4)

    def test_runtime_quantile_regressor_matches_native(self, regression_data):
        from sklearn.linear_model import QuantileRegressor
        X, y = regression_data
        est = QuantileRegressor(solver="highs").fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_orthogonal_matching_pursuit_matches_native(self, regression_data):
        from sklearn.linear_model import OrthogonalMatchingPursuit
        X, y = regression_data
        est = OrthogonalMatchingPursuit().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_passive_aggressive_regressor_matches_native(self, regression_data):
        from sklearn.linear_model import PassiveAggressiveRegressor
        X, y = regression_data
        est = PassiveAggressiveRegressor(random_state=0, max_iter=500).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_logistic_regression_cv_binary_matches_native(self, binary_data):
        from sklearn.linear_model import LogisticRegressionCV
        X, y = binary_data
        est = LogisticRegressionCV(max_iter=200, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_ridge_classifier_cv_binary_matches_native(self, binary_data):
        from sklearn.linear_model import RidgeClassifierCV
        X, y = binary_data
        est = RidgeClassifierCV().fit(X, y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.BINARY
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_sgd_classifier_binary_matches_native(self, binary_data):
        from sklearn.linear_model import SGDClassifier
        X, y = binary_data
        est = SGDClassifier(loss="log_loss", random_state=0, max_iter=500).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-3, atol=1e-3)

    def test_runtime_passive_aggressive_classifier_matches_native(self, binary_data):
        from sklearn.linear_model import PassiveAggressiveClassifier
        X, y = binary_data
        est = PassiveAggressiveClassifier(random_state=0, max_iter=500).fit(X, y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.BINARY
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_perceptron_matches_native(self, binary_data):
        from sklearn.linear_model import Perceptron
        X, y = binary_data
        est = Perceptron(random_state=0, max_iter=500).fit(X, y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.BINARY
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_linear_svc_multiclass_matches_native(self, multiclass_data):
        from sklearn.svm import LinearSVC
        X, y = multiclass_data
        est = LinearSVC(random_state=0, max_iter=2000).fit(X, y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.MULTICLASS
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))


# ── SVM ───────────────────────────────────────────────────────────────────────

class TestSVM:
    """SVC, SVR, NuSVC, NuSVR, LinearSVC, LinearSVR → SVM node."""

    def _check_svm_node(self, estimator):
        m = from_sklearn(estimator)
        ops = [n.op for n in m.nodes]
        assert "SVM" in ops
        return next(n for n in m.nodes if n.op == "SVM")

    def test_svc_binary(self, binary_data):
        from sklearn.svm import SVC
        X, y = binary_data
        node = self._check_svm_node(SVC(probability=True, kernel="rbf", random_state=0).fit(X, y))
        assert node.svm.kernel.kernel_type.name == "RBF"

    def test_svc_multiclass(self, multiclass_data):
        from sklearn.svm import SVC
        X, y = multiclass_data
        node = self._check_svm_node(SVC(probability=True, random_state=0).fit(X, y))
        assert node.svm.multiclass_strategy.name == "ONE_VS_ONE"

    def test_svr(self, regression_data):
        from sklearn.svm import SVR
        X, y = regression_data
        node = self._check_svm_node(SVR(kernel="rbf").fit(X, y))
        sv = node.svm.kernel.support_vectors
        assert sv.tensor_ref is not None or sv.tensor is not None

    def test_nu_svc(self, binary_data):
        from sklearn.svm import NuSVC
        X, y = binary_data
        self._check_svm_node(NuSVC(probability=True, random_state=0).fit(X, y))

    def test_nu_svr(self, regression_data):
        from sklearn.svm import NuSVR
        X, y = regression_data
        self._check_svm_node(NuSVR().fit(X, y))

    def test_linear_svc(self, binary_data):
        from sklearn.svm import LinearSVC
        X, y = binary_data
        node = self._check_svm_node(LinearSVC(random_state=0, max_iter=1000).fit(X, y))
        coef = node.svm.linear.coefficients
        assert coef.tensor_ref is not None or coef.tensor is not None

    def test_linear_svc_multiclass(self, multiclass_data):
        from sklearn.svm import LinearSVC
        X, y = multiclass_data
        node = self._check_svm_node(LinearSVC(random_state=0, max_iter=1000).fit(X, y))
        coef = node.svm.linear.coefficients
        assert coef.tensor_ref is not None or coef.tensor is not None

    def test_linear_svr(self, regression_data):
        from sklearn.svm import LinearSVR
        X, y = regression_data
        node = self._check_svm_node(LinearSVR(random_state=0, max_iter=1000).fit(X, y))
        coef = node.svm.linear.coefficients
        assert coef.tensor_ref is not None or coef.tensor is not None

    def test_svc_linear_kernel(self, binary_data):
        from sklearn.svm import SVC
        X, y = binary_data
        node = self._check_svm_node(SVC(kernel="linear", probability=True, random_state=0).fit(X, y))
        assert node.svm.kernel.kernel_type.name == "LINEAR"

    def test_svc_poly_kernel(self, binary_data):
        from sklearn.svm import SVC
        X, y = binary_data
        node = self._check_svm_node(SVC(kernel="poly", degree=3, probability=True, random_state=0).fit(X, y))
        assert node.svm.kernel.kernel_type.name == "POLY"
        assert node.svm.kernel.degree == 3

    def test_svm_support_vectors_stored(self, binary_data):
        from sklearn.svm import SVC
        X, y = binary_data
        est = SVC(kernel="rbf", random_state=0).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="binary")
        node = next(n for n in m.nodes if n.op == "SVM")
        sv = node.svm.kernel.support_vectors
        if sv.tensor_ref is not None:
            sv_entry = next(e for e in m.tensor_entries if e.id == sv.tensor_ref.id)
            assert len(sv_entry.dense.float64_data) > 0
        else:
            assert sv.tensor is not None
            assert len(sv.tensor.float64_data) > 0


    def test_runtime_svc_binary_matches_native(self, binary_data):
        from sklearn.svm import SVC
        X, y = binary_data
        est = SVC(kernel="rbf", probability=True, random_state=0).fit(X, y)
        # atol=0.005: libsvm uses iterative multiclass_probability with eps=0.0025
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X, verify_atol=0.005, verify_rtol=0.005)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=0.005, atol=0.005)

    def test_runtime_svc_multiclass_matches_native(self, multiclass_data):
        from sklearn.svm import SVC
        X, y = multiclass_data
        est = SVC(probability=True, random_state=0).fit(X, y)
        # atol=1e-3: libsvm uses iterative multiclass_probability with eps=0.0025
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X, verify_atol=1e-3, verify_rtol=1e-3)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        # Runtime predict uses proba argmax; sklearn uses 1-vs-1 decision votes.
        assert np.array_equal(m.predict(X), np.argmax(est.predict_proba(X), axis=1))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-3, atol=1e-3)

    def test_runtime_svr_matches_native(self, regression_data):
        from sklearn.svm import SVR
        X, y = regression_data
        est = SVR(kernel="rbf").fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_linear_svc_matches_native(self, binary_data):
        from sklearn.svm import LinearSVC
        X, y = binary_data
        est = LinearSVC(random_state=0, max_iter=2000).fit(X, y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.BINARY
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_linear_svr_matches_native(self, regression_data):
        from sklearn.svm import LinearSVR
        X, y = regression_data
        est = LinearSVR(random_state=0, max_iter=2000).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_nu_svc_binary_matches_native(self, binary_data):
        from sklearn.svm import NuSVC
        X, y = binary_data
        est = NuSVC(probability=True, random_state=0).fit(X, y)
        # atol=0.005: libsvm uses iterative multiclass_probability with eps=0.0025
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X, verify_atol=0.005, verify_rtol=0.005)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=0.005, atol=0.005)

    def test_runtime_nu_svr_matches_native(self, regression_data):
        from sklearn.svm import NuSVR
        X, y = regression_data
        est = NuSVR().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)


# ── Neural Network (MLP) ──────────────────────────────────────────────────────

class TestMLP:
    """MLPClassifier, MLPRegressor → NeuralNetwork node."""

    def test_mlp_regressor(self, regression_data):
        from sklearn.neural_network import MLPRegressor
        X, y = regression_data
        est = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=50, random_state=0).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "NeuralNetwork" in ops

    def test_mlp_regressor_layer_count(self, regression_data):
        from sklearn.neural_network import MLPRegressor
        X, y = regression_data
        est = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=50, random_state=0).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        node = next(n for n in m.nodes if n.op == "NeuralNetwork")
        assert len(node.neural_network.layers) == 3  # 2 hidden + 1 output

    def test_mlp_classifier_binary(self, binary_data):
        from sklearn.neural_network import MLPClassifier
        X, y = binary_data
        est = MLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=0).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="binary")
        assert "NeuralNetwork" in [n.op for n in m.nodes]

    def test_mlp_classifier_multiclass(self, multiclass_data):
        from sklearn.neural_network import MLPClassifier
        X, y = multiclass_data
        est = MLPClassifier(hidden_layer_sizes=(16,), max_iter=50, random_state=0).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert "NeuralNetwork" in [n.op for n in m.nodes]

    def test_mlp_weights_stored(self, regression_data):
        from sklearn.neural_network import MLPRegressor
        X, y = regression_data
        est = MLPRegressor(hidden_layer_sizes=(8,), max_iter=50, random_state=0).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        node = next(n for n in m.nodes if n.op == "NeuralNetwork")
        w_tv = node.neural_network.layers[0].weights
        if w_tv.tensor_ref is not None:
            w_entry = next(e for e in m.tensor_entries if e.id == w_tv.tensor_ref.id)
            assert len(w_entry.dense.float64_data) > 0
        else:
            assert w_tv.tensor is not None
            assert len(w_tv.tensor.float64_data) > 0

    def test_mlp_activation_stored(self, regression_data):
        from sklearn.neural_network import MLPRegressor
        X, y = regression_data
        est = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", max_iter=50, random_state=0).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        node = next(n for n in m.nodes if n.op == "NeuralNetwork")
        # Hidden layer should have TANH activation
        assert node.neural_network.layers[0].activation.name == "TANH"


    def test_runtime_regressor_matches_native(self, regression_data):
        from sklearn.neural_network import MLPRegressor
        X, y = regression_data
        est = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=200, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_binary_matches_native(self, binary_data):
        from sklearn.neural_network import MLPClassifier
        X, y = binary_data
        est = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_multiclass_matches_native(self, multiclass_data):
        from sklearn.neural_network import MLPClassifier
        X, y = multiclass_data
        est = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── Neighbors ─────────────────────────────────────────────────────────────────

class TestNeighbors:
    """KNeighbors and RadiusNeighbors classifiers/regressors → KNN node."""

    def _check_knn_node(self, estimator):
        m = from_sklearn(estimator)
        ops = [n.op for n in m.nodes]
        assert "KNN" in ops
        return next(n for n in m.nodes if n.op == "KNN"), m

    def test_kneighbors_classifier_binary(self, binary_data):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = binary_data
        node, m = self._check_knn_node(KNeighborsClassifier(n_neighbors=5).fit(X, y))
        _assert_std_io(m, task="binary")
        n_attr = next(a for a in node.attributes if a.name == "n_neighbors")
        assert n_attr.i == 5

    def test_kneighbors_classifier_multiclass(self, multiclass_data):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = multiclass_data
        _, m = self._check_knn_node(KNeighborsClassifier(n_neighbors=5).fit(X, y))
        _assert_std_io(m, task="multiclass", n_classes=3)

    def test_kneighbors_regressor(self, regression_data):
        from sklearn.neighbors import KNeighborsRegressor
        X, y = regression_data
        node, m = self._check_knn_node(KNeighborsRegressor(n_neighbors=5).fit(X, y))
        _assert_std_io(m, task="regression")
        task_attr = next(a for a in node.attributes if a.name == "task")
        assert task_attr.s == "regression"

    def test_radius_neighbors_classifier(self, binary_data):
        from sklearn.neighbors import RadiusNeighborsClassifier
        X, y = binary_data
        node, m = self._check_knn_node(RadiusNeighborsClassifier(radius=2.0, outlier_label=0).fit(X, y))
        _assert_std_io(m, task="binary")
        mode_attr = next(a for a in node.attributes if a.name == "neighbor_mode")
        assert mode_attr.s == "radius"

    def test_radius_neighbors_regressor(self, regression_data):
        from sklearn.neighbors import RadiusNeighborsRegressor
        X, y = regression_data
        node, m = self._check_knn_node(RadiusNeighborsRegressor(radius=2.0).fit(X, y))
        _assert_std_io(m, task="regression")
        mode_attr = next(a for a in node.attributes if a.name == "neighbor_mode")
        assert mode_attr.s == "radius"

    def test_knn_train_data_stored(self, binary_data):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = binary_data
        est = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        node, m = self._check_knn_node(est)
        _assert_std_io(m, task="binary")
        feat_attr = next(a for a in node.attributes if a.name == "train_features")
        entry = next(e for e in m.tensor_entries if e.id == feat_attr.tensor_ref.id)
        assert len(entry.dense.float64_data) == X.shape[0] * X.shape[1]

    def test_kneighbors_metric_stored(self, binary_data):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = binary_data
        node, m = self._check_knn_node(KNeighborsClassifier(metric="manhattan").fit(X, y))
        _assert_std_io(m, task="binary")
        m_attr = next(a for a in node.attributes if a.name == "metric")
        assert m_attr.s == "manhattan"


    def test_runtime_kneighbors_binary_matches_native(self, binary_data):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = binary_data
        est = KNeighborsClassifier(n_neighbors=5).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_kneighbors_multiclass_matches_native(self, multiclass_data):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = multiclass_data
        est = KNeighborsClassifier(n_neighbors=5).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_kneighbors_regressor_matches_native(self, regression_data):
        from sklearn.neighbors import KNeighborsRegressor
        X, y = regression_data
        est = KNeighborsRegressor(n_neighbors=5).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_kneighbors_manhattan_matches_native(self, binary_data):
        from sklearn.neighbors import KNeighborsClassifier
        X, y = binary_data
        est = KNeighborsClassifier(n_neighbors=5, metric="manhattan").fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_radius_neighbors_regressor_matches_native(self, regression_data):
        from sklearn.neighbors import RadiusNeighborsRegressor
        X, y = regression_data
        est = RadiusNeighborsRegressor(radius=2.0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_radius_neighbors_classifier_matches_native(self, binary_data):
        from sklearn.neighbors import RadiusNeighborsClassifier
        X, y = binary_data
        # weights='distance' avoids uniform-weight ties; separate test set produces outliers
        est = RadiusNeighborsClassifier(radius=2.0, outlier_label=0, weights='distance').fit(X, y)
        X_test = np.random.default_rng(1).standard_normal((100, X.shape[1])).astype(np.float32)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X_test), m_x.predict_proba(X_test), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X_test), est.predict(X_test), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X_test), est.predict_proba(X_test), rtol=1e-4, atol=1e-4)

    def test_runtime_radius_neighbors_classifier_uniform_matches_native(self, binary_data):
        from sklearn.neighbors import RadiusNeighborsClassifier
        X, y = binary_data
        # weights='uniform' on training data exercises tie-breaking (lowest class index wins)
        est = RadiusNeighborsClassifier(radius=2.0, outlier_label=0).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── Clustering ────────────────────────────────────────────────────────────────

class TestClustering:
    """KMeans, MiniBatchKMeans, GaussianMixture → Clustering node."""

    def _check_clustering_node(self, estimator):
        m = from_sklearn(estimator)
        ops = [n.op for n in m.nodes]
        assert "Clustering" in ops
        return next(n for n in m.nodes if n.op == "Clustering"), m

    def test_kmeans(self, regression_data):
        from sklearn.cluster import KMeans
        X, _ = regression_data
        est = KMeans(n_clusters=3, random_state=0, n_init=3).fit(X)
        node, m = self._check_clustering_node(est)
        _assert_std_io(m, task="clustering")
        centers_tv = node.clustering.prototype.centers
        if centers_tv.tensor_ref is not None:
            entry = next(e for e in m.tensor_entries if e.id == centers_tv.tensor_ref.id)
            assert len(entry.dense.float64_data) == 3 * X.shape[1]
        else:
            assert len(centers_tv.tensor.float64_data) == 3 * X.shape[1]

    def test_minibatch_kmeans(self, regression_data):
        from sklearn.cluster import MiniBatchKMeans
        X, _ = regression_data
        node, m = self._check_clustering_node(
            MiniBatchKMeans(n_clusters=3, random_state=0, n_init=3).fit(X)
        )
        _assert_std_io(m, task="clustering")
        assert node.clustering.prototype.distance_measure.name == "EUCLIDEAN"

    def test_gaussian_mixture(self, regression_data):
        from sklearn.mixture import GaussianMixture
        X, _ = regression_data
        node, m = self._check_clustering_node(
            GaussianMixture(n_components=3, random_state=0).fit(X)
        )
        _assert_std_io(m, task="clustering")
        means = node.clustering.gaussian_mixture.means
        covs  = node.clustering.gaussian_mixture.covariances
        assert means.tensor_ref is not None or means.tensor is not None
        assert covs.tensor_ref is not None or covs.tensor is not None

    def test_gaussian_mixture_diagonal(self, regression_data):
        from sklearn.mixture import GaussianMixture
        X, _ = regression_data
        node, m = self._check_clustering_node(
            GaussianMixture(n_components=3, covariance_type="diag", random_state=0).fit(X)
        )
        _assert_std_io(m, task="clustering")
        assert node.clustering.gaussian_mixture.covariance_type.name == "DIAGONAL"

    def test_gaussian_mixture_tied(self, regression_data):
        from sklearn.mixture import GaussianMixture
        X, _ = regression_data
        node, m = self._check_clustering_node(
            GaussianMixture(n_components=2, covariance_type="tied", random_state=0).fit(X)
        )
        _assert_std_io(m, task="clustering")
        assert node.clustering.gaussian_mixture.covariance_type.name == "FULL"

    def test_kmeans_cluster_labels(self, regression_data):
        from sklearn.cluster import KMeans
        X, _ = regression_data
        est = KMeans(n_clusters=4, random_state=0, n_init=3).fit(X)
        node, m = self._check_clustering_node(est)
        _assert_std_io(m, task="clustering")
        assert len(node.clustering.prototype.cluster_labels) == 4


    def test_runtime_kmeans_matches_native(self, regression_data):
        from sklearn.cluster import KMeans
        X, _ = regression_data
        est = KMeans(n_clusters=3, random_state=0, n_init=3).fit(X)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="clustering")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="clustering", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_minibatch_kmeans_matches_native(self, regression_data):
        from sklearn.cluster import MiniBatchKMeans
        X, _ = regression_data
        est = MiniBatchKMeans(n_clusters=3, random_state=0, n_init=3).fit(X)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="clustering")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="clustering", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_gaussian_mixture_matches_native(self, regression_data):
        from sklearn.mixture import GaussianMixture
        X, _ = regression_data
        est = GaussianMixture(n_components=3, random_state=0).fit(X)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="clustering")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="clustering", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))


# ── Naive Bayes ───────────────────────────────────────────────────────────────

class TestNaiveBayes:
    """GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB → NaiveBayes node."""

    def _check_nb_node(self, estimator):
        m = from_sklearn(estimator)
        ops = [n.op for n in m.nodes]
        assert "NaiveBayes" in ops
        return next(n for n in m.nodes if n.op == "NaiveBayes"), m

    def test_gaussian_nb_binary(self, binary_data):
        from sklearn.naive_bayes import GaussianNB
        X, y = binary_data
        node, m = self._check_nb_node(GaussianNB().fit(X, y))
        _assert_std_io(m, task="binary")
        means = node.naive_bayes.gaussian.means
        variances = node.naive_bayes.gaussian.variances
        assert means.tensor_ref is not None or means.tensor is not None
        assert variances.tensor_ref is not None or variances.tensor is not None

    def test_gaussian_nb_multiclass(self, multiclass_data):
        from sklearn.naive_bayes import GaussianNB
        X, y = multiclass_data
        node, m = self._check_nb_node(GaussianNB().fit(X, y))
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert node.naive_bayes.gaussian.variance_epsilon.double_value > 0

    def test_multinomial_nb(self):
        from sklearn.naive_bayes import MultinomialNB
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(100, 8)).astype(float)
        y = rng.integers(0, 3, size=100)
        node, m = self._check_nb_node(MultinomialNB().fit(X, y))
        _assert_std_io(m, task="multiclass", n_classes=3, n_features=8)
        flp = node.naive_bayes.multinomial.feature_log_prob
        assert flp.tensor_ref is not None or flp.tensor is not None

    def test_complement_nb(self):
        from sklearn.naive_bayes import ComplementNB
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(100, 8)).astype(float)
        y = rng.integers(0, 3, size=100)
        node, m = self._check_nb_node(ComplementNB().fit(X, y))
        _assert_std_io(m, task="multiclass", n_classes=3, n_features=8)
        flp = node.naive_bayes.multinomial.feature_log_prob
        assert flp.tensor_ref is not None or flp.tensor is not None

    def test_bernoulli_nb(self, binary_data):
        from sklearn.naive_bayes import BernoulliNB
        X, y = binary_data
        node, m = self._check_nb_node(BernoulliNB().fit(X, y))
        _assert_std_io(m, task="binary")
        flp = node.naive_bayes.bernoulli.feature_log_prob
        assert flp.tensor_ref is not None or flp.tensor is not None

    def test_bernoulli_nb_binarize_threshold(self):
        from sklearn.naive_bayes import BernoulliNB
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 6))
        y = rng.integers(0, 2, size=100)
        node, m = self._check_nb_node(BernoulliNB(binarize=0.5).fit(X, y))
        _assert_std_io(m, task="binary", n_features=6)
        assert abs(node.naive_bayes.bernoulli.binarize_threshold.double_value - 0.5) < 1e-9

    def test_categorical_nb(self):
        from sklearn.naive_bayes import CategoricalNB
        rng = np.random.default_rng(0)
        X = rng.integers(0, 4, size=(100, 5))
        y = rng.integers(0, 2, size=100)
        node, m = self._check_nb_node(CategoricalNB().fit(X, y))
        _assert_std_io(m, task="binary", n_features=5)
        clp = node.naive_bayes.categorical.category_log_prob
        assert clp.tensor_ref is not None or clp.tensor is not None
        assert len(node.naive_bayes.categorical.category_count) == 5

    def test_naive_bayes_priors_stored(self, binary_data):
        from sklearn.naive_bayes import GaussianNB
        X, y = binary_data
        est = GaussianNB().fit(X, y)
        node, m = self._check_nb_node(est)
        _assert_std_io(m, task="binary")
        priors_tv = node.naive_bayes.class_log_priors
        if priors_tv.tensor_ref is not None:
            entry = next(e for e in m.tensor_entries if e.id == priors_tv.tensor_ref.id)
            assert len(entry.dense.float64_data) == 2  # binary
        else:
            assert len(priors_tv.tensor.float64_data) == 2  # binary


    def test_runtime_gaussian_nb_binary_matches_native(self, binary_data):
        from sklearn.naive_bayes import GaussianNB
        X, y = binary_data
        est = GaussianNB().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_gaussian_nb_multiclass_matches_native(self, multiclass_data):
        from sklearn.naive_bayes import GaussianNB
        X, y = multiclass_data
        est = GaussianNB().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_multinomial_nb_matches_native(self):
        from sklearn.naive_bayes import MultinomialNB
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(100, 8)).astype(float)
        y = rng.integers(0, 3, size=100)
        est = MultinomialNB().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3, n_features=8)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, n_features=8)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_bernoulli_nb_matches_native(self, binary_data):
        from sklearn.naive_bayes import BernoulliNB
        X, y = binary_data
        est = BernoulliNB().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), est.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_complement_nb_matches_native(self):
        from sklearn.naive_bayes import ComplementNB
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(100, 8)).astype(float)
        y = rng.integers(0, 3, size=100)
        est = ComplementNB().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3, n_features=8)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, n_features=8)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_categorical_nb_binary_matches_native(self):
        from sklearn.naive_bayes import CategoricalNB
        rng = np.random.default_rng(0)
        X = rng.integers(0, 4, size=(100, 5)).astype(float)
        y = rng.integers(0, 2, size=100)
        est = CategoricalNB().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary", n_features=5)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", n_features=5)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_categorical_nb_multiclass_matches_native(self):
        from sklearn.naive_bayes import CategoricalNB
        rng = np.random.default_rng(0)
        X = rng.integers(0, 4, size=(100, 5)).astype(float)
        y = rng.integers(0, 3, size=100)
        est = CategoricalNB().fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3, n_features=5)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, n_features=5)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── Voting ────────────────────────────────────────────────────────────────────

class TestVoting:
    """VotingClassifier (soft/hard) and VotingRegressor."""

    def test_voting_classifier_soft_binary(self, binary_data):
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        clf = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=200, random_state=0)),
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ],
            voting="soft",
        ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "SoftVote" in ops
        soft_node = next(n for n in m.nodes if n.op == "SoftVote")
        assert len(soft_node.inputs) == 2

    def test_voting_classifier_hard_binary(self, binary_data):
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        clf = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=200, random_state=0)),
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ],
            voting="hard",
        ).fit(X, y)
        m = from_sklearn(clf)
        # Hard voting: no y_prob output, only y_pred
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 6]
        assert m.model_schema.targets[0].kind == TargetKind.BINARY
        ops = [n.op for n in m.nodes]
        assert "MajorityVote" in ops

    def test_voting_classifier_weighted(self, binary_data):
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        clf = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=200, random_state=0)),
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ],
            voting="soft",
            weights=[2.0, 1.0],
        ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        soft_node = next(n for n in m.nodes if n.op == "SoftVote")
        w_attr = next(a for a in soft_node.attributes if a.name == "weights")
        assert len(w_attr.float64s) == 2

    def test_voting_regressor(self, regression_data):
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        reg = VotingRegressor(
            estimators=[
                ("r1", Ridge()),
                ("r2", DecisionTreeRegressor(max_depth=3, random_state=0)),
            ],
        ).fit(X, y)
        m = from_sklearn(reg)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "Average" in ops

    def test_voting_regressor_weighted(self, regression_data):
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        reg = VotingRegressor(
            estimators=[
                ("r1", Ridge()),
                ("r2", DecisionTreeRegressor(max_depth=3, random_state=0)),
            ],
            weights=[3.0, 1.0],
        ).fit(X, y)
        m = from_sklearn(reg)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "WeightedAverage" in ops


    def test_runtime_voting_classifier_soft_binary_matches_native(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        est = VotingClassifier(voting="soft", estimators=[
            ("lr", LogisticRegression(max_iter=200, random_state=0)),
            ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("rf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_voting_regressor_matches_native(self, regression_data):
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        est = VotingRegressor(estimators=[
            ("lr", LinearRegression()),
            ("ridge", Ridge()),
            ("dt", DecisionTreeRegressor(max_depth=3, random_state=0)),
        ]).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_voting_hard_binary_matches_native(self, binary_data):
        from sklearn.ensemble import VotingClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        # Use 3 estimators (odd count) to avoid tie-breaking differences
        est = VotingClassifier(voting="hard", estimators=[
            ("dt1", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("dt2", DecisionTreeClassifier(max_depth=2, random_state=1)),
            ("dt3", DecisionTreeClassifier(max_depth=4, random_state=2)),
        ]).fit(X, y)
        # Hard voting: no y_prob output
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.BINARY
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_voting_hard_multiclass_matches_native(self, multiclass_data):
        from sklearn.ensemble import VotingClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        est = VotingClassifier(voting="hard", estimators=[
            ("dt1", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("dt2", DecisionTreeClassifier(max_depth=2, random_state=1)),
        ]).fit(X, y)
        # Hard voting: no y_prob output
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        assert _ir.model_schema.targets[0].kind == TargetKind.MULTICLASS
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        assert np.array_equal(m.predict(X), m_x.predict(X))
        assert np.array_equal(m.predict(X), est.predict(X))

    def test_runtime_voting_soft_multiclass_matches_native(self, multiclass_data):
        from sklearn.ensemble import VotingClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        est = VotingClassifier(voting="soft", estimators=[
            ("dt1", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ("dt2", DecisionTreeClassifier(max_depth=2, random_state=1)),
        ]).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── Stacking ──────────────────────────────────────────────────────────────────

class TestStacking:
    """StackingClassifier and StackingRegressor → base nodes + Concat + final estimator."""

    def test_stacking_classifier_binary(self, binary_data):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        clf = StackingClassifier(
            estimators=[
                ("dt1", DecisionTreeClassifier(max_depth=3, random_state=0)),
                ("dt2", DecisionTreeClassifier(max_depth=2, random_state=1)),
            ],
            final_estimator=LogisticRegression(max_iter=200),
        ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Concat" not in ops
        assert "Linear" in ops

    def test_stacking_regressor(self, regression_data):
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        reg = StackingRegressor(
            estimators=[
                ("dt1", DecisionTreeRegressor(max_depth=3, random_state=0)),
                ("dt2", DecisionTreeRegressor(max_depth=2, random_state=1)),
            ],
            final_estimator=Ridge(),
        ).fit(X, y)
        m = from_sklearn(reg)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "Concat" not in ops
        assert "Linear" in ops

    def test_stacking_classifier_passthrough(self, binary_data):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        clf = StackingClassifier(
            estimators=[("dt", DecisionTreeClassifier(max_depth=3, random_state=0))],
            final_estimator=LogisticRegression(max_iter=200),
            passthrough=True,
        ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        # passthrough feeds raw X as an additional input to the final estimator directly
        assert "Concat" not in [n.op for n in m.nodes]

    def test_stacking_classifier_multiclass(self, multiclass_data):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        clf = StackingClassifier(
            estimators=[("dt", DecisionTreeClassifier(max_depth=3, random_state=0))],
            final_estimator=LogisticRegression(max_iter=200),
        ).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="multiclass", n_classes=3)
        assert "Concat" not in [n.op for n in m.nodes]


    def test_runtime_stacking_classifier_binary_matches_native(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        est = StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=0)),
                ("rf", RandomForestClassifier(n_estimators=5, random_state=0)),
            ],
            final_estimator=LogisticRegression(max_iter=200, random_state=0),
        ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_stacking_regressor_matches_native(self, regression_data):
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        est = StackingRegressor(
            estimators=[
                ("dt", DecisionTreeRegressor(max_depth=3, random_state=0)),
                ("ridge", Ridge()),
            ],
            final_estimator=LinearRegression(),
        ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_stacking_classifier_binary_proba_matches_native(self, binary_data):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        est = StackingClassifier(
            estimators=[("dt", DecisionTreeClassifier(max_depth=3, random_state=0))],
            final_estimator=LogisticRegression(max_iter=200, random_state=0),
        ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_stacking_classifier_multiclass_matches_native(self, multiclass_data):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        X, y = multiclass_data
        est = StackingClassifier(
            estimators=[("dt", DecisionTreeClassifier(max_depth=3, random_state=0))],
            final_estimator=LogisticRegression(max_iter=200, random_state=0),
        ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── AdaBoost Regressor ────────────────────────────────────────────────────────

class TestAdaBoostRegressor:
    """AdaBoostRegressor → child regressors + WeightedMedian."""

    def test_adaboost_regressor_converts(self, regression_data):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        est = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=5,
            random_state=0,
        ).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "WeightedMedian" in ops

    def test_adaboost_regressor_weights_stored(self, regression_data):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        est = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=4,
            random_state=0,
        ).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        wm_node = next(n for n in m.nodes if n.op == "WeightedMedian")
        w_attr = next(a for a in wm_node.attributes if a.name == "weights")
        assert len(w_attr.float64s) == 4

    def test_adaboost_regressor_n_children(self, regression_data):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        est = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=3,
            random_state=0,
        ).fit(X, y)
        m = from_sklearn(est)
        _assert_std_io(m, task="regression")
        wm_node = next(n for n in m.nodes if n.op == "WeightedMedian")
        assert len(wm_node.inputs) == 3


    def test_runtime_matches_native(self, regression_data):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        est = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=3), n_estimators=5, random_state=0
        ).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="regression")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)


# ── MultiOutput ───────────────────────────────────────────────────────────────

class TestMultiOutput:
    """MultiOutputClassifier, MultiOutputRegressor, ClassifierChain, RegressorChain."""

    def test_multi_output_regressor(self, regression_data):
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        Y = np.stack([y, y * 0.5], axis=1)
        m = from_sklearn(
            MultiOutputRegressor(DecisionTreeRegressor(max_depth=3, random_state=0)).fit(X, Y)
        )
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 6]
        ops = [n.op for n in m.nodes]
        assert "Tree" in ops
        assert "Concat" in ops
        tree_nodes = [n for n in m.nodes if n.op == "Tree"]
        assert len(tree_nodes) == 2

    def test_multi_output_classifier(self, binary_data):
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        Y = np.stack([y, 1 - y], axis=1)
        m = from_sklearn(
            MultiOutputClassifier(DecisionTreeClassifier(max_depth=3, random_state=0)).fit(X, Y)
        )
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 6]
        ops = [n.op for n in m.nodes]
        assert "Tree" in ops
        assert "Concat" in ops

    def test_classifier_chain(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import ClassifierChain
        X, y = binary_data
        Y = np.stack([y, 1 - y], axis=1)
        m = from_sklearn(
            ClassifierChain(LogisticRegression(max_iter=200, random_state=0)).fit(X, Y)
        )
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 6]
        ops = [n.op for n in m.nodes]
        assert "Concat" in ops
        assert "Linear" in ops

    def test_regressor_chain(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import RegressorChain
        X, y = regression_data
        Y = np.stack([y, y * 0.5], axis=1)
        m = from_sklearn(RegressorChain(Ridge()).fit(X, Y))
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 6]
        ops = [n.op for n in m.nodes]
        assert "Concat" in ops
        assert "Linear" in ops

    def test_multi_output_regressor_linear_base(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        X, y = regression_data
        Y = np.stack([y, y * 2], axis=1)
        m = from_sklearn(MultiOutputRegressor(Ridge()).fit(X, Y))
        assert m.inputs[0].name == "X"
        assert m.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert m.inputs[0].type.shape == [-1, 6]
        linear_nodes = [n for n in m.nodes if n.op == "Linear"]
        assert len(linear_nodes) == 2


    def test_runtime_multi_output_regressor_matches_native(self, regression_data):
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.tree import DecisionTreeRegressor
        X, y = regression_data
        Y = np.stack([y, y * 0.5], axis=1)
        est = MultiOutputRegressor(DecisionTreeRegressor(max_depth=3, random_state=0)).fit(X, Y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)

    def test_runtime_multi_output_classifier_matches_native(self, binary_data):
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.tree import DecisionTreeClassifier
        X, y = binary_data
        Y = np.stack([y, 1 - y], axis=1)
        est = MultiOutputClassifier(DecisionTreeClassifier(max_depth=3, random_state=0)).fit(X, Y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        expected_prob = np.column_stack([p[:, 1] for p in est.predict_proba(X)])
        np.testing.assert_allclose(m.predict_proba(X), expected_prob, rtol=1e-4, atol=1e-4)

    def test_runtime_classifier_chain_matches_native(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import ClassifierChain
        X, y = binary_data
        Y = np.stack([y, 1 - y], axis=1)
        est = ClassifierChain(LogisticRegression(max_iter=200, random_state=0)).fit(X, Y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_regressor_chain_matches_native(self, regression_data):
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import RegressorChain
        X, y = regression_data
        Y = np.stack([y, y * 0.5], axis=1)
        est = RegressorChain(Ridge()).fit(X, Y)
        _ir = from_sklearn(est)
        assert _ir.inputs[0].name == "X"
        assert _ir.inputs[0].type.dtype == omle.DataType.FLOAT64
        assert _ir.inputs[0].type.shape == [-1, 6]
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        assert _ir_x.inputs[0].name == "X"
        assert _ir_x.inputs[0].type.dtype == omle.DataType.FLOAT32
        assert _ir_x.inputs[0].type.shape == [-1, 6]
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)


# ── OneVsRest ─────────────────────────────────────────────────────────────────

class TestOneVsRest:
    """OneVsRestClassifier → binary child nodes + TakeSlots + SoftVote (no Concat)."""

    def test_ovr_binary_converts(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        X, y = binary_data
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200, random_state=0)).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "SoftVote" in ops
        assert "Concat" not in ops

    def test_ovr_multiclass_converts(self, multiclass_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        X, y = multiclass_data
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200, random_state=0)).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="multiclass", n_classes=3)
        ops = [n.op for n in m.nodes]
        assert "SoftVote" in ops
        assert "Concat" not in ops
        # One TakeSlots per class to extract positive-class prob
        ts_nodes = [n for n in m.nodes if n.op == "TakeSlots"]
        assert len(ts_nodes) == 3  # 3 classes

    def test_ovr_one_vs_one_raises(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsOneClassifier
        X, y = binary_data
        clf = OneVsOneClassifier(LogisticRegression(max_iter=200)).fit(X, y)
        with pytest.raises(NotImplementedError):
            from_sklearn(clf)

    def test_ovr_linear_base(self, binary_data):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC
        X, y = binary_data
        clf = OneVsRestClassifier(LinearSVC(max_iter=1000, random_state=0)).fit(X, y)
        m = from_sklearn(clf)
        _assert_std_io(m, task="binary")
        assert "SoftVote" in [n.op for n in m.nodes]


    def test_runtime_ovr_binary_matches_native(self, binary_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        X, y = binary_data
        est = OneVsRestClassifier(LogisticRegression(max_iter=200, random_state=0)).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="binary")
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)

    def test_runtime_ovr_multiclass_matches_native(self, multiclass_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        X, y = multiclass_data
        est = OneVsRestClassifier(LogisticRegression(max_iter=200, random_state=0)).fit(X, y)
        _ir = from_sklearn(est)
        _assert_std_io(_ir, task="multiclass", n_classes=3)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(est, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), est.predict(X), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(m.predict_proba(X), est.predict_proba(X), rtol=1e-4, atol=1e-4)


# ── FunctionTransformer ───────────────────────────────────────────────────────

class TestFunctionTransformer:
    """FunctionTransformer with numpy functions → Derive nodes."""

    def _check_derive(self, func, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        X, y = regression_data
        pipe = Pipeline([
            ("ft", FunctionTransformer(func)),
            ("clf", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "Derive" in ops

    def test_log(self, regression_data):
        X, y = regression_data
        # Use abs values to avoid log(negative)
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        X_pos = np.abs(X) + 1e-6
        pipe = Pipeline([
            ("ft", FunctionTransformer(np.log)),
            ("clf", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X_pos, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        assert "Derive" in [n.op for n in m.nodes]

    def test_exp(self, regression_data):
        self._check_derive(np.exp, regression_data)

    def test_sqrt(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        X, y = regression_data
        X_pos = np.abs(X)
        pipe = Pipeline([
            ("ft", FunctionTransformer(np.sqrt)),
            ("clf", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X_pos, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        assert "Derive" in [n.op for n in m.nodes]

    def test_log1p(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        X, y = regression_data
        X_pos = np.abs(X)
        pipe = Pipeline([
            ("ft", FunctionTransformer(np.log1p)),
            ("clf", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X_pos, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        assert "Derive" in [n.op for n in m.nodes]

    def test_passthrough(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        X, y = regression_data
        pipe = Pipeline([
            ("ft", FunctionTransformer(None)),
            ("clf", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        # passthrough FunctionTransformer emits no Derive node
        assert isinstance(m, omle.OMLEModel)

    def test_unsupported_func_raises(self, regression_data):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        X, y = regression_data
        pipe = Pipeline([
            ("ft", FunctionTransformer(lambda x: x + 1)),
            ("clf", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        with pytest.raises(NotImplementedError):
            from_sklearn(pipe)


# ── LabelEncoder ──────────────────────────────────────────────────────────────

class TestLabelEncoder:
    """LabelEncoder → LabelEncoder node."""

    def test_label_encoder_op_emitted(self):
        from sklearn.preprocessing import LabelEncoder

        from omle_convert.sklearn._builder import Builder
        from omle_convert.sklearn._preprocessing import _label_encoder
        le = LabelEncoder().fit(["cat", "dog", "bird"])
        builder = Builder()
        _label_encoder(le, "X", "le", builder)
        ops = [n.op for n in builder.nodes]
        assert "LabelEncoder" in ops

    def test_label_encoder_classes_stored(self):
        from sklearn.preprocessing import LabelEncoder

        from omle_convert.sklearn._builder import Builder
        from omle_convert.sklearn._preprocessing import _label_encoder
        le = LabelEncoder().fit(["cat", "dog", "bird"])
        builder = Builder()
        _label_encoder(le, "X", "le", builder)
        node = next(n for n in builder.nodes if n.op == "LabelEncoder")
        labels_attr = next(a for a in node.attributes if a.name == "labels")
        assert set(_tensor_strings(builder.tensor_entries, labels_attr)) == {"cat", "dog", "bird"}

    def test_label_encoder_in_pipeline(self, binary_data):
        from sklearn.preprocessing import LabelEncoder
        X, y = binary_data
        # LabelEncoder used inside a pipeline as a transformer
        from omle_convert.sklearn._builder import Builder
        from omle_convert.sklearn._preprocessing import _label_encoder
        le = LabelEncoder().fit([0, 1])
        builder = Builder()
        _label_encoder(le, "X", "le", builder)
        assert len(builder.nodes) == 1


# ── FeatureUnion ──────────────────────────────────────────────────────────────

class TestFeatureUnion:
    """FeatureUnion → parallel transformer branches + Concat."""

    def test_feature_union_emits_concat(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        X, y = binary_data
        pipe = Pipeline([
            ("union", FeatureUnion([
                ("std", StandardScaler()),
                ("mm", MinMaxScaler()),
            ])),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Concat" in ops

    def test_feature_union_two_branches(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        X, y = binary_data
        pipe = Pipeline([
            ("union", FeatureUnion([
                ("std", StandardScaler()),
                ("mm", MinMaxScaler()),
            ])),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        concat_node = next(n for n in m.nodes if n.op == "Concat")
        assert len(concat_node.inputs) == 2

    def test_feature_union_with_pca(self, binary_data):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import StandardScaler
        X, y = binary_data
        pipe = Pipeline([
            ("union", FeatureUnion([
                ("std", StandardScaler()),
                ("pca", PCA(n_components=3)),
            ])),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "Linear" in ops  # PCA → Linear node
        assert "Concat" in ops


# ── Additional feature selectors ──────────────────────────────────────────────

class TestFeatureSelectionExtra:
    """SelectFdr, SelectFpr, SelectFwe, RFECV, SequentialFeatureSelector → TakeSlots."""

    def _check(self, selector, data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        X, y = data
        pipe = Pipeline([
            ("sel", selector),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]).fit(X, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="binary")
        ops = [n.op for n in m.nodes]
        assert "TakeSlots" in ops
        ts = next(n for n in m.nodes if n.op == "TakeSlots")
        idx_attr = next(a for a in ts.attributes if a.name == "indices")
        expected = selector.get_support(indices=True).tolist()
        assert idx_attr.ints == expected

    def test_select_fdr(self, binary_data):
        from sklearn.feature_selection import SelectFdr, f_classif
        self._check(SelectFdr(f_classif, alpha=0.5), binary_data)

    def test_select_fpr(self, binary_data):
        from sklearn.feature_selection import SelectFpr, f_classif
        self._check(SelectFpr(f_classif, alpha=0.5), binary_data)

    def test_select_fwe(self, binary_data):
        from sklearn.feature_selection import SelectFwe, f_classif
        self._check(SelectFwe(f_classif, alpha=0.5), binary_data)

    def test_rfecv(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFECV
        self._check(
            RFECV(RandomForestClassifier(n_estimators=5, random_state=0), cv=3),
            binary_data,
        )

    def test_sequential_feature_selector(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SequentialFeatureSelector
        self._check(
            SequentialFeatureSelector(
                RandomForestClassifier(n_estimators=5, random_state=0),
                n_features_to_select=4, cv=3,
            ),
            binary_data,
        )


# ── Additional dimensionality reduction ───────────────────────────────────────

class TestDimensionalityReductionExtra:
    """SparsePCA, LatentDirichletAllocation."""

    def _make_pipe(self, transformer, estimator_cls, data):
        from sklearn.pipeline import Pipeline
        X, y = data
        return Pipeline([
            ("t", transformer),
            ("m", estimator_cls(n_estimators=5, random_state=0) if hasattr(estimator_cls, "n_estimators") else estimator_cls()),
        ]).fit(X, y)

    def test_sparse_pca(self, regression_data):
        from sklearn.decomposition import SparsePCA
        from sklearn.ensemble import RandomForestRegressor
        X, y = regression_data
        pipe = self._make_pipe(
            SparsePCA(n_components=3, random_state=0, max_iter=10),
            RandomForestRegressor, regression_data,
        )
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression")
        ops = [n.op for n in m.nodes]
        assert "SparsePCA" in ops
        node = next(n for n in m.nodes if n.op == "SparsePCA")
        comp_attr = next(a for a in node.attributes if a.name == "components")
        assert _attr_has_tensor(comp_attr)

    def test_lda_transformer(self, regression_data):
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.ensemble import RandomForestRegressor
        rng = np.random.default_rng(0)
        X_count = rng.integers(0, 10, size=(200, 8)).astype(float)
        y = rng.standard_normal(200)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("lda", LatentDirichletAllocation(n_components=3, random_state=0, max_iter=5)),
            ("m", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]).fit(X_count, y)
        m = from_sklearn(pipe)
        _assert_std_io(m, task="regression", n_features=8)
        ops = [n.op for n in m.nodes]
        assert "LatentDirichletAllocation" in ops




# ── Float64 input data ────────────────────────────────────────────────────────

class TestSklearnFloat64:
    def test_dt_regression_converts(self, dt_regressor_f64):
        m = from_sklearn(dt_regressor_f64)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT64)
        assert isinstance(m, omle.OMLEModel)

    def test_dt_regression_uses_tree_op(self, dt_regressor_f64):
        m = from_sklearn(dt_regressor_f64)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT64)
        assert m.nodes[0].op == "Tree"
        assert m.nodes[0].tree is not None

    def test_dt_regression_no_post_transform(self, dt_regressor_f64):
        # Tree op has no post_transform; verify the node uses Tree not TreeEnsemble
        m = from_sklearn(dt_regressor_f64)
        _assert_std_io(m, task="regression", input_dtype=omle.DataType.FLOAT64)
        assert m.nodes[0].op == "Tree"

    def test_rf_binary_converts(self, rf_binary_f64):
        m = from_sklearn(rf_binary_f64)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT64)
        assert isinstance(m, omle.OMLEModel)

    def test_rf_binary_tree_count(self, rf_binary_f64):
        m = from_sklearn(rf_binary_f64)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT64)
        assert len(m.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_rf_binary_aggregation(self, rf_binary_f64):
        m = from_sklearn(rf_binary_f64)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT64)
        assert m.nodes[0].tree_ensemble.aggregation == TreeAggregation.AVERAGE

    def test_rf_binary_n_features(self, rf_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        m = from_sklearn(rf_binary_f64)
        _assert_std_io(m, task="binary", input_dtype=omle.DataType.FLOAT64)
        assert m.inputs[0].type.shape[1] == X.shape[1]

    def test_gb_multiclass_converts(self, gb_multiclass_f64):
        m = from_sklearn(gb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT64)
        assert isinstance(m, omle.OMLEModel)

    def test_gb_multiclass_softmax(self, gb_multiclass_f64):
        m = from_sklearn(gb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT64)
        assert m.nodes[0].tree_ensemble.post_transform == PostTransform.SOFTMAX

    def test_gb_multiclass_tree_group(self, gb_multiclass_f64):
        m = from_sklearn(gb_multiclass_f64)
        _assert_std_io(m, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT64)
        tg = m.nodes[0].tree_ensemble.tree_group
        for i, g in enumerate(tg):
            assert g == i % 3

    def test_runtime_dt_regression_matches_native(self, dt_regressor_f64, regression_data_f64):
        X, _ = regression_data_f64
        _ir = from_sklearn(dt_regressor_f64)
        _assert_std_io(_ir, task="regression", input_dtype=omle.DataType.FLOAT64)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(dt_regressor_f64, X=X)
        _assert_std_io(_ir_x, task="regression", input_dtype=omle.DataType.FLOAT64)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict(X), m_x.predict(X), rtol=1e-6)
        np.testing.assert_allclose(m.predict(X), dt_regressor_f64.predict(X), rtol=1e-5)

    def test_runtime_rf_binary_matches_native(self, rf_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        _ir = from_sklearn(rf_binary_f64)
        _assert_std_io(_ir, task="binary", input_dtype=omle.DataType.FLOAT64)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(rf_binary_f64, X=X)
        _assert_std_io(_ir_x, task="binary", input_dtype=omle.DataType.FLOAT64)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), rf_binary_f64.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), rf_binary_f64.predict_proba(X), rtol=1e-5)

    def test_runtime_gb_multiclass_matches_native(self, gb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        _ir = from_sklearn(gb_multiclass_f64)
        _assert_std_io(_ir, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT64)
        m = _load_runtime(_ir)
        _ir_x = from_sklearn(gb_multiclass_f64, X=X)
        _assert_std_io(_ir_x, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT64)
        m_x = _load_runtime(_ir_x)
        np.testing.assert_allclose(m.predict_proba(X), m_x.predict_proba(X), rtol=1e-6)
        assert np.array_equal(m.predict(X), gb_multiclass_f64.predict(X))
        np.testing.assert_allclose(m.predict_proba(X), gb_multiclass_f64.predict_proba(X), rtol=1e-4, atol=1e-4)



# ── DataFrame (float32-only) pipeline tests ───────────────────────────────────

class TestSklearnDataFramePipelines:
    """Runtime tests for sklearn Pipelines fitted on float32-only DataFrames.

    These exercise named-column handling: feature_names_in_, ColumnTransformer
    column-name dispatch, and the float32 DataFrame code path in the converter.
    """

    def test_scaler_ridge_regression(self, regression_df_f32):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        df, y = regression_df_f32
        p = Pipeline([("sc", StandardScaler()), ("m", Ridge())]).fit(df, y)
        _ir = from_sklearn(p, X=df)
        _assert_std_io(_ir, task="regression", input_dtype=omle.DataType.FLOAT32)
        m = _load_runtime(_ir)
        np.testing.assert_allclose(m.predict(df), p.predict(df), rtol=1e-4, atol=1e-4)

    def test_scaler_lr_binary(self, binary_df_f32):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        df, y = binary_df_f32
        p = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=200, random_state=0)),
        ]).fit(df, y)
        _ir = from_sklearn(p, X=df)
        _assert_std_io(_ir, task="binary", input_dtype=omle.DataType.FLOAT32)
        m = _load_runtime(_ir)
        assert np.array_equal(m.predict(df), p.predict(df))
        np.testing.assert_allclose(m.predict_proba(df), p.predict_proba(df), rtol=1e-4, atol=1e-4)

    def test_scaler_lr_multiclass(self, multiclass_df_f32):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        df, y = multiclass_df_f32
        p = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=200, random_state=0)),
        ]).fit(df, y)
        _ir = from_sklearn(p, X=df)
        _assert_std_io(_ir, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m = _load_runtime(_ir)
        assert np.array_equal(m.predict(df), p.predict(df))
        np.testing.assert_allclose(m.predict_proba(df), p.predict_proba(df), rtol=1e-4, atol=1e-4)

    def test_column_transformer_rf_binary(self, binary_df_f32):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        df, y = binary_df_f32
        p = Pipeline([
            ("ct", ColumnTransformer([
                ("sc", StandardScaler(), ["feat_a", "feat_b", "feat_c"]),
                ("mm", MinMaxScaler(),   ["feat_d", "feat_e", "feat_f"]),
            ])),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]).fit(df, y)
        _ir = from_sklearn(p, X=df)
        _assert_std_io(_ir, task="binary", input_dtype=omle.DataType.FLOAT32)
        m = _load_runtime(_ir)
        assert np.array_equal(m.predict(df), p.predict(df))
        np.testing.assert_allclose(m.predict_proba(df), p.predict_proba(df), rtol=1e-4, atol=1e-4)

    def test_column_transformer_rf_multiclass(self, multiclass_df_f32):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        df, y = multiclass_df_f32
        p = Pipeline([
            ("ct", ColumnTransformer([
                ("sc", StandardScaler(), ["feat_a", "feat_b", "feat_c"]),
                ("pass", "passthrough",  ["feat_d", "feat_e", "feat_f"]),
            ])),
            ("rf", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]).fit(df, y)
        _ir = from_sklearn(p, X=df)
        _assert_std_io(_ir, task="multiclass", n_classes=3, input_dtype=omle.DataType.FLOAT32)
        m = _load_runtime(_ir)
        assert np.array_equal(m.predict(df), p.predict(df))
        np.testing.assert_allclose(m.predict_proba(df), p.predict_proba(df), rtol=1e-4, atol=1e-4)

    def test_pca_gb_regression(self, regression_df_f32):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        df, y = regression_df_f32
        p = Pipeline([
            ("pca", PCA(n_components=4, random_state=0)),
            ("gb",  GradientBoostingRegressor(n_estimators=50, random_state=0)),
        ]).fit(df, y)
        _ir = from_sklearn(p, X=df)
        _assert_std_io(_ir, task="regression", input_dtype=omle.DataType.FLOAT32)
        m = _load_runtime(_ir)
        np.testing.assert_allclose(m.predict(df), p.predict(df), rtol=1e-4, atol=1e-4)

    def test_feature_union_binary(self, binary_df_f32):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        df, y = binary_df_f32
        p = Pipeline([
            ("union", FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])),
            ("lr",    LogisticRegression(max_iter=200, random_state=0)),
        ]).fit(df, y)
        _ir = from_sklearn(p, X=df)
        _assert_std_io(_ir, task="binary", input_dtype=omle.DataType.FLOAT32)
        m = _load_runtime(_ir)
        assert np.array_equal(m.predict(df), p.predict(df))
        np.testing.assert_allclose(m.predict_proba(df), p.predict_proba(df), rtol=1e-4, atol=1e-4)


class TestOneClassSVMAnomaly:
    """OneClassSVM and SGDOneClassSVM → AnomalyDetection node."""

    def _node(self, estimator, X):
        m = from_sklearn(estimator, X=X)
        node = next(n for n in m.nodes if n.op == "AnomalyDetection")
        assert node.domain == "omle.ml"
        return node, m

    def _tensor_values(self, m, tv):
        if tv.tensor_ref is not None:
            entry = next(e for e in m.tensor_entries if e.id == tv.tensor_ref.id)
            return list(entry.dense.float64_data)
        return list(tv.tensor.float64_data)

    def test_one_class_svm_rbf(self, regression_data):
        from sklearn.svm import OneClassSVM
        X, _ = regression_data
        est = OneClassSVM(kernel="rbf", gamma=0.3, nu=0.2).fit(X)
        node, m = self._node(est, X)

        ocs = node.anomaly_detection.one_class_svm
        assert ocs is not None
        assert ocs.kernel_svm.kernel_type.name == "RBF"
        assert ocs.kernel_svm.gamma.double_value == pytest.approx(est._gamma)
        # score_samples is the plain kernel sum, so no intercept is stored; the
        # offset carries it and decision_function is score - offset.
        assert ocs.kernel_svm.intercept is None
        assert ocs.offset.double_value == pytest.approx(float(np.ravel(est.offset_)[0]))

        sv = self._tensor_values(m, ocs.kernel_svm.support_vectors)
        assert len(sv) == est.support_vectors_.size
        dual = self._tensor_values(m, ocs.kernel_svm.dual_coefficients)
        assert dual == pytest.approx(est.dual_coef_.ravel().tolist())

    @pytest.mark.parametrize("kernel", ["linear", "poly", "sigmoid"])
    def test_one_class_svm_kernels(self, regression_data, kernel):
        from sklearn.svm import OneClassSVM
        X, _ = regression_data
        est = OneClassSVM(kernel=kernel, nu=0.2, degree=2, coef0=1.0).fit(X)
        node, _ = self._node(est, X)
        ocs = node.anomaly_detection.one_class_svm
        assert ocs.kernel_svm.kernel_type.name == kernel.upper()
        assert ocs.kernel_svm.degree == est.degree
        assert ocs.kernel_svm.coef0.double_value == pytest.approx(est.coef0)

    def test_one_class_svm_rejects_precomputed_kernel(self, regression_data):
        from sklearn.svm import OneClassSVM
        X, _ = regression_data
        est = OneClassSVM(kernel="precomputed", nu=0.2)
        est.fit(X @ X.T)
        with pytest.raises(NotImplementedError, match="precomputed"):
            from_sklearn(est, X=X)

    def test_sgd_one_class_svm(self, regression_data):
        from sklearn.linear_model import SGDOneClassSVM
        X, _ = regression_data
        est = SGDOneClassSVM(nu=0.2, random_state=0).fit(X)
        node, m = self._node(est, X)

        lin = node.anomaly_detection.linear_one_class_svm
        assert lin is not None
        # score_samples(X) = X @ coef_, with offset_ applied separately.
        assert lin.intercept is None
        assert lin.offset.double_value == pytest.approx(float(est.offset_.ravel()[0]))
        coef = self._tensor_values(m, lin.coefficients)
        assert coef == pytest.approx(est.coef_.ravel().tolist())

    def test_anomaly_metadata_is_set(self, regression_data):
        from sklearn.svm import OneClassSVM
        X, _ = regression_data
        node, _ = self._node(OneClassSVM(nu=0.2).fit(X), X)
        ad = node.anomaly_detection
        assert ad.task_type.name == "ANOMALY_DETECTION"
        assert ad.mode.name == "NOVELTY_DETECTION"
        assert ad.raw_score_polarity.name == "LOWER_MORE_ABNORMAL"
        assert ad.threshold is not None
