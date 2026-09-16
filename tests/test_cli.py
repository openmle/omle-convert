"""Tests for the omle-convert CLI."""

import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")

import omle
from omle_convert.cli import main

N_ESTIMATORS = 5


@pytest.fixture(scope="module")
def xgb_model_json(tmp_path_factory, binary_data):
    p = tmp_path_factory.mktemp("xgb") / "clf.json"
    X, y = binary_data
    model = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                               random_state=0, verbosity=0, eval_metric="logloss")
    model.fit(X, y)
    model.get_booster().save_model(str(p))
    return p


@pytest.fixture(scope="module")
def lgb_model_txt(tmp_path_factory, binary_data):
    p = tmp_path_factory.mktemp("lgb") / "clf.txt"
    X, y = binary_data
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                random_state=0, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(str(p))
    return p



# ── XGBoost ───────────────────────────────────────────────────────────────────

class TestCLIXGBoost:
    def test_convert_to_json(self, xgb_model_json, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(xgb_model_json), str(out)])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert len(loaded.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_feature_names_option(self, xgb_model_json, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(xgb_model_json), str(out), "--feature-names", "a,b,c,d,e,f"])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert [f.name for f in loaded.model_schema.features] == list("abcdef")

    def test_target_name_option(self, xgb_model_json, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(xgb_model_json), str(out), "--target-name", "label"])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert loaded.model_schema.targets[0].name == "label"

    def test_model_name_option(self, xgb_model_json, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(xgb_model_json), str(out), "--model-name", "my_xgb"])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert loaded.metadata.name == "my_xgb"


# ── LightGBM ──────────────────────────────────────────────────────────────────

class TestCLILightGBM:
    def test_convert_to_json(self, lgb_model_txt, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(lgb_model_txt), str(out)])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert len(loaded.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_feature_names_option(self, lgb_model_txt, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(lgb_model_txt), str(out), "--feature-names", "a,b,c,d,e,f"])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert [f.name for f in loaded.model_schema.features] == list("abcdef")

    def test_model_name_option(self, lgb_model_txt, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(lgb_model_txt), str(out), "--model-name", "my_lgb"])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert loaded.metadata.name == "my_lgb"



# ── sklearn ───────────────────────────────────────────────────────────────────

class TestCLISklearn:
    @pytest.fixture
    def sklearn_joblib(self, binary_data, tmp_path):
        joblib = pytest.importorskip("joblib", reason="joblib not installed")
        from sklearn.ensemble import RandomForestClassifier
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=0).fit(X, y)
        p = tmp_path / "clf.joblib"
        joblib.dump(model, str(p))
        return p

    def test_convert_to_json(self, sklearn_joblib, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(sklearn_joblib), str(out)])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert len(loaded.nodes[0].tree_ensemble.trees) == N_ESTIMATORS

    def test_model_name_option(self, sklearn_joblib, tmp_path):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(sklearn_joblib), str(out), "--model-name", "my_rf"])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert loaded.metadata.name == "my_rf"


# ── CatBoost ──────────────────────────────────────────────────────────────────

class TestCLICatBoost:
    """CatBoost writes both `.cbm` and `.json`.

    The `.json` case shares its extension with XGBoost, so the CLI picks the
    converter by scanning the file for `"oblivious_trees"`. A prefix peek is not
    enough — CatBoost emits that key tens of kilobytes in — so this test guards
    against a regression that silently routes the file to the XGBoost loader.
    """

    @pytest.fixture(scope="class")
    def cb_models(self, binary_data, tmp_path_factory):
        cb = pytest.importorskip("catboost", reason="catboost not installed")
        d = tmp_path_factory.mktemp("cb")
        X, y = binary_data
        model = cb.CatBoostClassifier(iterations=N_ESTIMATORS, depth=3,
                                      random_seed=0, verbose=0).fit(X, y)
        model.save_model(str(d / "clf.cbm"))
        model.save_model(str(d / "clf.json"), format="json")
        return d / "clf.cbm", d / "clf.json"

    @pytest.mark.parametrize("index", [0, 1], ids=["cbm", "json"])
    def test_convert(self, cb_models, tmp_path, index):
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(cb_models[index]), str(out)])
        assert exc.value.code == 0
        loaded = omle.load_json(out)
        assert loaded.metadata.source_frameworks[0].name == "catboost"


# ── Extension dispatch ────────────────────────────────────────────────────────

class TestCLIExtensions:
    """The extensions advertised in the README all reach a converter."""

    def test_xgboost_ubj(self, binary_data, tmp_path):
        X, y = binary_data
        model = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                  random_state=0, verbosity=0,
                                  eval_metric="logloss").fit(X, y)
        src = tmp_path / "clf.ubj"
        model.get_booster().save_model(str(src))
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(src), str(out)])
        assert exc.value.code == 0
        assert omle.load_json(out).metadata.source_frameworks[0].name == "xgboost"

    def test_lightgbm_bin(self, binary_data, tmp_path):
        """`.bin` is dispatched to the LightGBM text loader alongside `.txt`."""
        X, y = binary_data
        model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=3,
                                   random_state=0, verbose=-1).fit(X, y)
        src = tmp_path / "clf.bin"
        model.booster_.save_model(str(src))
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(src), str(out)])
        assert exc.value.code == 0
        assert omle.load_json(out).metadata.source_frameworks[0].name == "lightgbm"

    def test_pickled_non_sklearn_estimator(self, binary_data, tmp_path):
        """A pickle dispatches on the unpickled object, not the extension."""
        joblib = pytest.importorskip("joblib", reason="joblib not installed")
        X, y = binary_data
        src = tmp_path / "clf.joblib"
        joblib.dump(lgb.LGBMClassifier(n_estimators=N_ESTIMATORS,
                                       verbose=-1).fit(X, y), str(src))
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit) as exc:
            main([str(src), str(out)])
        assert exc.value.code == 0
        names = [f.name for f in omle.load_json(out).metadata.source_frameworks]
        assert "lightgbm" in names


# ── Error handling ────────────────────────────────────────────────────────────

class TestCLIErrors:
    def test_missing_args(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0

    def test_nonexistent_input(self, tmp_path):
        out = tmp_path / "out.omle"
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path / "ghost.json"), str(out)])
        assert exc.value.code != 0
