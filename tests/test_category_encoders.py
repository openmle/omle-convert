"""Tests for category_encoders → OMLE converter."""

import numpy as np
import pandas as pd
import pytest

ce = pytest.importorskip("category_encoders", reason="category_encoders not installed")
sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import omle
from omle.ir.enums import OutputRole, TargetKind
from omle_convert.sklearn import from_sklearn


def _assert_cat_io(m, *, target_name="y"):
    """Assert IO invariants for pipelines using cat_df (color/size STRING inputs, regression)."""
    input_names = {i.name for i in m.inputs}
    assert "color" in input_names
    assert "size" in input_names
    for inp in m.inputs:
        assert inp.type.dtype == omle.DataType.STRING
        assert inp.type.shape == [-1]
    assert len(m.outputs) == 1
    assert m.outputs[0].name == "y_pred"
    assert m.outputs[0].role == OutputRole.PREDICTION
    assert m.outputs[0].type.dtype == omle.DataType.FLOAT64
    assert m.outputs[0].type.shape == [-1]
    assert m.model_schema.targets[0].kind == TargetKind.REGRESSION
    assert m.model_schema.targets[0].name == target_name
    feat_names = {f.name for f in m.model_schema.features}
    assert "color" in feat_names
    assert "size" in feat_names
    for feat in m.model_schema.features:
        assert feat.type.dtype == omle.DataType.STRING
        assert feat.type.shape == [-1]


# ── Compat patch ──────────────────────────────────────────────────────────────
# category_encoders 2.6.x doesn't implement __sklearn_tags__ (sklearn 1.6+ API).
# Patch BaseEstimator._get_tags so calls from inside sklearn don't crash.
try:
    import sklearn.base as _sb
    _orig_get_tags = _sb.BaseEstimator._get_tags

    def _safe_get_tags(self):
        try:
            return _orig_get_tags(self)
        except AttributeError:
            return {}

    _sb.BaseEstimator._get_tags = _safe_get_tags
except Exception:
    pass


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cat_df():
    """Small DataFrame with categorical string columns and a continuous target."""
    rng = np.random.default_rng(42)
    n = 120
    colors = ["red", "green", "blue", "yellow"]
    sizes = ["small", "medium", "large"]
    df = pd.DataFrame({
        "color": rng.choice(colors, n),
        "size":  rng.choice(sizes, n),
    })
    y = (
        (df["color"] == "red").astype(float) * 2.0
        + (df["size"] == "large").astype(float) * 1.5
        + rng.standard_normal(n) * 0.3
    )
    return df, y.values


# ── Converter tests ───────────────────────────────────────────────────────────

class TestTargetEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import TargetEncoder
        X, y = cat_df
        enc = TargetEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)

    def test_node_emitted(self, cat_df):
        from category_encoders import TargetEncoder
        X, y = cat_df
        enc = TargetEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        model = from_sklearn(pipe, X=X)
        node_ops = [n.op for n in model.nodes]
        assert "TargetEncoder" in node_ops
        _assert_cat_io(model)


class TestCountEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import CountEncoder
        X, y = cat_df
        enc = CountEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)

    def test_node_emitted(self, cat_df):
        from category_encoders import CountEncoder
        X, y = cat_df
        enc = CountEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        model = from_sklearn(pipe, X=X)
        node_ops = [n.op for n in model.nodes]
        assert "OrdinalEncoder" in node_ops
        _assert_cat_io(model)


class TestWOEEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import WOEEncoder
        X, y = cat_df
        y_bin = (y > y.mean()).astype(int)
        enc = WOEEncoder(cols=["color", "size"]).fit(X, y_bin)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y_bin))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)


class TestMEstimateEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import MEstimateEncoder
        X, y = cat_df
        enc = MEstimateEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)


class TestJamesSteinEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import JamesSteinEncoder
        X, y = cat_df
        enc = JamesSteinEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)


class TestQuantileEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import QuantileEncoder
        X, y = cat_df
        enc = QuantileEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)


class TestCatBoostEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import CatBoostEncoder
        X, y = cat_df
        enc = CatBoostEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)


class TestLeaveOneOutEncoderConverter:
    def test_converts_without_error(self, cat_df):
        from category_encoders import LeaveOneOutEncoder
        X, y = cat_df
        enc = LeaveOneOutEncoder(cols=["color", "size"]).fit(X, y)
        pipe = Pipeline([("enc", enc), ("reg", Ridge().fit(enc.transform(X), y))])
        m = from_sklearn(pipe, X=X)
        assert m is not None
        _assert_cat_io(m)


# ── Runtime tests ─────────────────────────────────────────────────────────────

omr = pytest.importorskip("omleruntime", reason="omleruntime not installed")


def _load_runtime(ir_model):
    from omle.proto.convert import ir_to_proto
    data = ir_to_proto(ir_model).SerializeToString()
    return omr.load_bytes(data)


def _pipe_predict(enc_cls, enc_kwargs, cat_df, y_override=None):
    """Fit Pipeline(encoder → Ridge) and return (native_predict, runtime_predict, X)."""
    X, y = cat_df
    if y_override is not None:
        y = y_override
    enc = enc_cls(cols=["color", "size"], **enc_kwargs).fit(X, y)
    X_enc = enc.transform(X)
    reg = Ridge().fit(X_enc, y)
    pipe = Pipeline([("enc", enc), ("reg", reg)])
    native = pipe.predict(X)
    model = from_sklearn(pipe, X=X)
    rt = _load_runtime(model)
    rt_pred = rt.predict(X)
    return native, rt_pred


class TestTargetEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import TargetEncoder
        native, rt_pred = _pipe_predict(TargetEncoder, {}, cat_df)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-4, atol=1e-4)


class TestCountEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import CountEncoder
        native, rt_pred = _pipe_predict(CountEncoder, {}, cat_df)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-4, atol=1e-4)


class TestWOEEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import WOEEncoder
        X, y = cat_df
        y_bin = (y > y.mean()).astype(int)
        native, rt_pred = _pipe_predict(WOEEncoder, {}, cat_df, y_override=y_bin)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-4, atol=1e-4)


class TestMEstimateEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import MEstimateEncoder
        native, rt_pred = _pipe_predict(MEstimateEncoder, {}, cat_df)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-4, atol=1e-4)


class TestJamesSteinEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import JamesSteinEncoder
        native, rt_pred = _pipe_predict(JamesSteinEncoder, {}, cat_df)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-4, atol=1e-4)


class TestQuantileEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import QuantileEncoder
        native, rt_pred = _pipe_predict(QuantileEncoder, {}, cat_df)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-4, atol=1e-4)


class TestCatBoostEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import CatBoostEncoder
        native, rt_pred = _pipe_predict(CatBoostEncoder, {}, cat_df)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-3, atol=1e-3)


class TestLeaveOneOutEncoderRuntime:
    def test_predict_matches_native(self, cat_df):
        from category_encoders import LeaveOneOutEncoder
        native, rt_pred = _pipe_predict(LeaveOneOutEncoder, {}, cat_df)
        np.testing.assert_allclose(rt_pred, native, rtol=1e-4, atol=1e-4)
