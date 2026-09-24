"""Shared fixtures for omle-convert tests."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omle import OMLEModel

# ── Shared SparkSession fixture ────────────────────────────────────────────────
# A single session-scoped `spark` fixture shared by test_spark.py and
# test_spark_jvm.py.  It must be created here (conftest.py) so that
# SparkSession.builder is called exactly once — PySpark's SparkContext is a JVM
# singleton and cannot be restarted after stop() in the same process.
# If the omle-spark JAR is found, the session is configured with it so that
# TestPredictionMatch tests in test_spark_jvm.py can use it.

try:
    from pyspark.sql import SparkSession as _SparkSession
    _HAS_PYSPARK = True
except ImportError:
    _HAS_PYSPARK = False

if _HAS_PYSPARK:
    # The omle-spark wheel ships every JAR the JVM side needs — the Scala
    # transformer for both Scala versions, the omle-runtime JAR carrying the
    # native library per platform, and JNA — and picks the pair matching the
    # installed PySpark. Asking it beats rediscovering them here, which is what
    # this used to do and got wrong three ways: the runtime JAR was pinned to a
    # literal `omle-runtime-0.1.0.jar` that no build produces, the JNA search
    # covered only macOS cache layouts, and the availability check tested for
    # `spark/python/omle/spark/ml.py` — the path before the package was renamed
    # to omle_spark — so it had been silently False ever since.
    try:
        import omle_spark
        _OMLE_SPARK_JARS = [Path(j) for j in omle_spark.jars()]
    except Exception:
        _OMLE_SPARK_JARS = []
    _HAS_OMLE_SPARK = bool(_OMLE_SPARK_JARS)

    def _native_lib_fallback():
        """Directory holding a locally built libomleruntime, or None.

        Only needed when the bundled omle-runtime JAR carries no native library
        — a wheel built from a plain `mvn package`. A released wheel has them
        and JNA extracts the matching one from the classpath.

        The check matters. Setting jna.library.path unconditionally silently
        pairs a locally built native with whatever omle-runtime version
        omle-spark pins, and the two need not agree on the C ABI. That is not
        hypothetical: OMLE_COL_STRING moved from 1 to 2 when FLOAT64 support
        landed, so an rc9 JAR sending 1 for a string column had its char*
        array read as double* by a freshly built native. Every row decoded to
        category index 0, and a OneHotEncoder pipeline silently predicted as
        though every row held the first category — no error, just wrong
        numbers, visible only where a decision boundary happened to sit.
        """
        try:
            import omle_spark as _osp
        except ImportError:
            return None
        import zipfile
        for _jar in _OMLE_SPARK_JARS:
            if not os.path.basename(_jar).startswith("omle-runtime-"):
                continue
            try:
                _names = zipfile.ZipFile(_jar).namelist()
            except Exception:
                continue
            if any(n.endswith((".so", ".dylib", ".dll")) for n in _names):
                return None  # the JAR is self-sufficient; do not override it
        try:
            import omle_runtime as _omr
            return Path(_omr.__file__).parent
        except ImportError:
            return None

    _NATIVE_LIB = _native_lib_fallback()

    def _find_xgboost4j_spark_jar():
        """Locate the xgboost4j-spark JAR matching the installed xgboost Python version.

        Search order: XGBOOST4J_SPARK_JAR env var → local Maven repo → Coursier cache → ivy2.
        """
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

    def _synapseml_maven_coords(package_version):
        """Return (packages, repositories) for the running PySpark, or None.

        SynapseML publishes a separate build per Spark line, and they differ in
        Scala version, so the coordinate cannot be hardcoded:

            Spark 3.5.x   Scala 2.12   synapseml_2.12:<ver>
            Spark 4.0.1+  Scala 2.13   synapseml_2.13:<ver>-spark4.0
            Spark 4.1.x   Scala 2.13   synapseml_2.13:<ver>-spark4.1

        See https://github.com/microsoft/SynapseML for the current table.

        Spark lines newer than the last published build fall back to the newest
        one known here, so they fail loudly at Maven resolution rather than
        skipping silently. Update the mapping when SynapseML publishes a build
        for a new line.
        """
        try:
            import pyspark as _pyspark
            parts = _pyspark.__version__.split(".")
            major, minor = int(parts[0]), int(parts[1])
        except Exception:
            return None

        if major == 3 and minor >= 5:
            scala, pkg = "2.12", package_version
        elif major == 4 and minor == 0:
            scala, pkg = "2.13", f"{package_version}-spark4.0"
        elif major == 4 and minor >= 1:
            scala, pkg = "2.13", f"{package_version}-spark4.1"
        else:
            return None  # Spark <3.5 is not covered by current SynapseML builds

        avro = f"org.apache.spark:spark-avro_{scala}:{_pyspark.__version__}"
        return (
            f"com.microsoft.azure:synapseml_{scala}:{pkg},{avro}",
            "https://mmlspark.blob.core.windows.net/maven",
        )

    def _find_synapseml_jar():
        """Locate the SynapseML JAR or detect Maven-based resolution.

        Search order: SYNAPSEML_JAR env var → local JAR in the package directory →
        Maven resolution (synapseml installs as 'synapse' namespace and has no bundled
        JARs; returns the sentinel string "maven" so the Spark session can be configured
        with spark.jars.packages instead).
        Returns None when PySpark >= 4.x makes synapseml 1.x JARs (Scala 2.12) incompatible.
        """
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
            # No bundled JAR — synapseml resolves via Maven at Spark startup.
            import synapse.ml.core as _smc
            if _synapseml_maven_coords(_smc.__spark_package_version__) is None:
                return None
            return "maven"
        except ImportError:
            pass
        return None

    _XGBOOST4J_JAR = _find_xgboost4j_spark_jar()
    _SYNAPSEML_JAR = _find_synapseml_jar()
    _HAS_XGBOOST4J_SPARK = _XGBOOST4J_JAR is not None
    _HAS_SYNAPSEML = _SYNAPSEML_JAR is not None

    @pytest.fixture(scope="session")
    def binary_df(spark):
        """100-row binary classification DataFrame (3-dim assembled feature vector + label)."""
        from pyspark.ml.linalg import Vectors
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        rows = [(Vectors.dense(*X[i].tolist()), float(y[i])) for i in range(n)]
        return spark.createDataFrame(rows, ["features", "label"])

    @pytest.fixture(scope="session")
    def regression_df(spark):
        """100-row regression DataFrame (2-dim assembled feature vector + label).

        Labels are strictly positive so the DataFrame is also usable for
        Poisson / log-link GLMs.
        """
        from pyspark.ml.linalg import Vectors
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = np.exp(X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.1)
        rows = [(Vectors.dense(*X[i].tolist()), float(y[i])) for i in range(n)]
        return spark.createDataFrame(rows, ["features", "label"])

    @pytest.fixture(scope="session")
    def multiclass_df(spark):
        """120-row 3-class classification DataFrame (2-dim assembled feature vector + label)."""
        from pyspark.ml.linalg import Vectors
        rng = np.random.default_rng(42)
        n = 120
        X = rng.standard_normal((n, 2))
        y = (np.abs(X[:, 0]) * 3).astype(int).clip(0, 2).astype(float)
        rows = [(Vectors.dense(*X[i].tolist()), float(y[i])) for i in range(n)]
        return spark.createDataFrame(rows, ["features", "label"])

    @pytest.fixture(scope="session")
    def nonneg_df(spark):
        """40-row non-negative count DataFrame (3-dim) for NaiveBayes/ChiSqSelector."""
        from pyspark.ml.linalg import Vectors
        rng = np.random.default_rng(0)
        n = 40
        X = rng.integers(0, 5, size=(n, 3)).astype(float)
        y = (X[:, 0] + X[:, 1] > 3).astype(float)
        rows = [(Vectors.dense(*X[i].tolist()), float(y[i])) for i in range(n)]
        return spark.createDataFrame(rows, ["features", "label"])

    @pytest.fixture(scope="session")
    def binary_feat_df(spark):
        """30-row binary {0, 1} feature DataFrame (3-dim) for NaiveBayes bernoulli."""
        from pyspark.ml.linalg import Vectors
        rng = np.random.default_rng(0)
        n = 30
        X = rng.integers(0, 2, size=(n, 3)).astype(float)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        rows = [(Vectors.dense(*X[i].tolist()), float(y[i])) for i in range(n)]
        return spark.createDataFrame(rows, ["features", "label"])

    @pytest.fixture(scope="session")
    def spark_mixed_df(spark):
        """120-row Spark DataFrame with mixed column types: 1 string category (3 levels),
        2 numeric floats, 1 binary int, and a binary label.  Designed for testing full
        pipelines that combine StringIndexer / OneHotEncoder / VectorAssembler with a
        downstream model."""
        rng = np.random.default_rng(99)
        n = 120
        colors = rng.choice(["red", "green", "blue"], n)
        sizes = rng.standard_normal(n)
        weights = rng.standard_normal(n)
        is_premium = rng.integers(0, 2, size=n)
        labels = (sizes + weights > 0).astype(float)
        rows = [
            (str(colors[i]), float(sizes[i]), float(weights[i]), int(is_premium[i]), float(labels[i]))
            for i in range(n)
        ]
        return spark.createDataFrame(rows, ["color", "size", "weight", "is_premium", "label"])

    @pytest.fixture(scope="session")
    def text_df(spark):
        """Small text DataFrame for NLP pipeline tests."""
        rows = [
            ("hello world foo",), ("spark ml pipeline",), ("hello spark world",),
            ("foo bar baz",), ("machine learning model",), ("hello foo bar",),
            ("spark pipeline model",), ("world hello bar",), ("foo spark hello",),
            ("bar baz ml model",),
        ]
        return spark.createDataFrame(rows, ["text"])

    @pytest.fixture(scope="session")
    def spark():
        builder = _SparkSession.builder.master("local[2]").appName("omle-convert-test")
        # Collect all extra JARs that are present.
        _extra_jars = [
            j for j in [
                *(_OMLE_SPARK_JARS if _HAS_OMLE_SPARK else []),
                _XGBOOST4J_JAR,
                _SYNAPSEML_JAR if isinstance(_SYNAPSEML_JAR, Path) else None,
            ] if j is not None
        ]
        if _extra_jars:
            builder = builder.config("spark.jars", ",".join(str(j) for j in _extra_jars))
        if _SYNAPSEML_JAR == "maven":
            import synapse.ml.core as _smc
            _packages, _repos = _synapseml_maven_coords(_smc.__spark_package_version__)
            builder = (builder
                .config("spark.jars.packages", _packages)
                .config("spark.jars.repositories", _repos))
        if _HAS_OMLE_SPARK:
            _open_mods = (
                " --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
                " --add-opens=java.base/java.nio=ALL-UNNAMED"
                " --add-opens=java.base/java.lang=ALL-UNNAMED"
                " --add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
                " --add-opens=java.base/java.util=ALL-UNNAMED"
            )
            _jna_path = f"-Djna.library.path={_NATIVE_LIB}" if _NATIVE_LIB else ""
            builder = (builder
                .config("spark.driver.extraJavaOptions", _jna_path + _open_mods)
                .config("spark.executor.extraJavaOptions", _jna_path + _open_mods))
        session = builder.getOrCreate()
        session.sparkContext.setLogLevel("ERROR")
        yield session
        session.stop()

_CONVERTERS = [
    ("omle_convert.xgboost",   ["from_xgboost", "from_xgboost_json"]),
    ("omle_convert.lightgbm",  ["from_lightgbm", "from_lightgbm_text"]),
    ("omle_convert.catboost",  ["from_catboost", "from_catboost_file"]),
    ("omle_convert.sklearn",   ["from_sklearn"]),
    ("omle_convert.pmml",      ["from_pmml", "from_pmml_string"]),
    ("omle_convert.spark",     ["from_spark", "from_spark_live"]),
    # omle_convert/__init__.py does `from omle_convert.sklearn import from_sklearn`
    # at import time, and _convert_object calls those bare names. Patching only
    # the defining modules above leaves those copies untouched, so every
    # to_omle / export_omle conversion bypassed the callbacks entirely. Wrap the
    # re-exported copies too; the depth counter keeps the export to one per call.
    ("omle_convert", [
        "from_catboost", "from_catboost_file",
        "from_lightgbm", "from_lightgbm_text",
        "from_sklearn",
        "from_spark", "from_spark_live",
        "from_xgboost", "from_xgboost_json",
    ]),
]

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EXPORT_DIR = os.path.join(_TESTS_DIR, "models")


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _print_proto_json(model, func_name: str) -> None:
    try:
        from google.protobuf import json_format

        from omle.proto.convert import ir_to_proto
        msg = ir_to_proto(model)
        text = json_format.MessageToJson(
            msg,
            preserving_proto_field_name=True,
            indent=2,
            always_print_fields_with_no_presence=False,
        )
        print(f"\n{'─' * 60}")
        print(f"  {func_name}")
        print('─' * 60)
        print(text)
    except ImportError:
        pass


def _sanitize_test_name(test_id: str) -> str:
    """Convert a PYTEST_CURRENT_TEST value to a safe filename stem.

    "tests/test_sklearn.py::TestClass::test_func[0] (call)"
    → "test_sklearn__TestClass__test_func_0"
    """
    node = test_id.split(" ")[0]           # strip " (call)" / " (setup)" etc.
    parts = node.split("::")               # ["tests/test_sklearn.py", "TestClass", "test_func[0]"]
    file_stem = parts[0].replace("\\", "/").split("/")[-1]
    if file_stem.endswith(".py"):
        file_stem = file_stem[:-3]
    combined = "__".join([file_stem] + parts[1:])
    combined = re.sub(r"[^a-zA-Z0-9_]", "_", combined)
    return combined.strip("_")


def _make_export_callback(export_dir: str):
    """Return a callback that saves each converted model to *export_dir*."""
    os.makedirs(export_dir, exist_ok=True)
    _counts: dict[str, int] = {}

    def _export(model: OMLEModel, func_name: str) -> None:
        try:
            import omle
            test_id = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
            name = _sanitize_test_name(test_id)
            count = _counts.get(name, 0)
            suffix = f"_{count}" if count > 0 else ""
            path = os.path.join(export_dir, f"{name}{suffix}.omle")
            if model.verification:
                omle.save(model, path)
                _counts[name] = count + 1
        except Exception:
            pass

    return _export


# ── Patching ──────────────────────────────────────────────────────────────────

# Depth counter: when a wrapped converter calls another wrapped converter
# internally (e.g. from_xgboost delegates to from_sklearn), only the
# outermost call should trigger the callbacks.
_converter_depth = 0


def _wrap(original, func_name: str, callbacks: list):
    def wrapper(*args, **kwargs):
        global _converter_depth
        _converter_depth += 1
        try:
            model = original(*args, **kwargs)
            is_outermost = _converter_depth == 1
        finally:
            _converter_depth -= 1
        if is_outermost:
            for cb in callbacks:
                cb(model, func_name)
        return model
    return wrapper


def pytest_addoption(parser):
    parser.addoption(
        "--proto-json",
        action="store_true",
        default=True,
        help="Print proto JSON of every converted model (default: True)",
    )
    parser.addoption(
        "--export-models",
        metavar="DIR",
        nargs="?",
        const=_DEFAULT_EXPORT_DIR,
        default=_DEFAULT_EXPORT_DIR,
        help="Export each converted model as <test_name>.omle into DIR "
             f"(default: {_DEFAULT_EXPORT_DIR})",
    )


def pytest_configure(config):
    callbacks = []

    try:
        if config.getoption("--proto-json"):
            callbacks.append(_print_proto_json)
    except ValueError:
        pass

    try:
        export_dir = config.getoption("--export-models")
        if export_dir:
            callbacks.append(_make_export_callback(export_dir))
    except ValueError:
        pass

    if not callbacks:
        return

    import importlib
    for module_name, func_names in _CONVERTERS:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        for func_name in func_names:
            original = getattr(mod, func_name, None)
            if original is not None:
                setattr(mod, func_name, _wrap(original, f"{module_name}.{func_name}", callbacks))


@pytest.fixture(scope="session")
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    y = X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + rng.standard_normal(200).astype(np.float32)
    return X, y


@pytest.fixture(scope="session")
def binary_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture(scope="session")
def multiclass_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    y = (np.abs(X[:, 0]) * 3).astype(int).clip(0, 2)
    return X, y


@pytest.fixture(scope="session")
def regression_data_f64():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6))  # float64
    y = X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + rng.standard_normal(200)
    return X, y


@pytest.fixture(scope="session")
def binary_data_f64():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6))  # float64
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture(scope="session")
def multiclass_data_f64():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 6))  # float64
    y = (np.abs(X[:, 0]) * 3).astype(int).clip(0, 2)
    return X, y


@pytest.fixture(scope="session")
def numeric_mixed_df():
    """DataFrame with int64, float32, float64, and bool columns — no string columns.

    Both XGBoost and LightGBM accept this directly without a Pipeline.
    Returns (df, y_binary).
    """
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "a_int":     rng.integers(0, 10, size=n),                    # int64
        "b_float32": rng.standard_normal(n).astype(np.float32),      # float32
        "c_float64": rng.standard_normal(n),                         # float64
        "d_bool":    rng.integers(0, 2, size=n).astype(bool),        # bool
    })
    y = (df["a_int"].to_numpy() + df["b_float32"].to_numpy() +
         df["c_float64"].to_numpy() > 0.5).astype(int)
    return df, y



@pytest.fixture(scope="session")
def cat_mixed_df(mixed_df):
    """mixed_df with string columns (gender, segment) converted to CategoricalDtype.

    Both XGBoost (enable_categorical=True) and LightGBM accept this directly.
    Returns (df_cat, y_regression, y_binary, y_multiclass).
    """
    df, y_regression, y_binary, y_multiclass = mixed_df
    df_cat = df.copy()
    df_cat["gender"]  = df_cat["gender"].astype("category")
    df_cat["segment"] = df_cat["segment"].astype("category")
    return df_cat, y_regression, y_binary, y_multiclass


@pytest.fixture(scope="session")
def regression_df_f32():
    """DataFrame with float32-only columns for regression."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "feat_a": rng.standard_normal(n).astype(np.float32),
        "feat_b": rng.standard_normal(n).astype(np.float32),
        "feat_c": rng.standard_normal(n).astype(np.float32),
        "feat_d": rng.standard_normal(n).astype(np.float32),
        "feat_e": rng.standard_normal(n).astype(np.float32),
        "feat_f": rng.standard_normal(n).astype(np.float32),
    })
    y = (df["feat_a"].to_numpy() * 2.0 + df["feat_b"].to_numpy()
         - df["feat_c"].to_numpy()).astype(np.float32)
    return df, y


@pytest.fixture(scope="session")
def binary_df_f32():
    """DataFrame with float32-only columns for binary classification."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "feat_a": rng.standard_normal(n).astype(np.float32),
        "feat_b": rng.standard_normal(n).astype(np.float32),
        "feat_c": rng.standard_normal(n).astype(np.float32),
        "feat_d": rng.standard_normal(n).astype(np.float32),
        "feat_e": rng.standard_normal(n).astype(np.float32),
        "feat_f": rng.standard_normal(n).astype(np.float32),
    })
    y = (df["feat_a"].to_numpy() + df["feat_b"].to_numpy() > 0).astype(int)
    return df, y


@pytest.fixture(scope="session")
def multiclass_df_f32():
    """DataFrame with float32-only columns for multiclass classification."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "feat_a": rng.standard_normal(n).astype(np.float32),
        "feat_b": rng.standard_normal(n).astype(np.float32),
        "feat_c": rng.standard_normal(n).astype(np.float32),
        "feat_d": rng.standard_normal(n).astype(np.float32),
        "feat_e": rng.standard_normal(n).astype(np.float32),
        "feat_f": rng.standard_normal(n).astype(np.float32),
    })
    y = (np.abs(df["feat_a"].to_numpy()) * 3).astype(int).clip(0, 2)
    return df, y


@pytest.fixture(scope="session")
def mixed_df():
    """DataFrame with mixed dtypes: int, float, bool, binary string, multiclass string.

    Returns (df, y_regression, y_binary, y_multiclass).
    """
    rng = np.random.default_rng(0)
    n = 300

    df = pd.DataFrame({
        "age":       rng.integers(18, 80, size=n),
        "education": rng.integers(6, 22, size=n).astype(np.int32),
        "income":    rng.uniform(20_000, 150_000, size=n),  # float64
        "stock":     rng.uniform(0.0, 10_000.0, size=n).astype(np.float32),
        "active":    rng.integers(0, 2, size=n).astype(bool),
        "gender":    rng.choice(["M", "F"], size=n),
        "segment":   rng.choice(["low", "mid", "high", "premium"], size=n),
    })

    y_regression  = (df["income"].to_numpy() / 1000 + rng.standard_normal(n)).astype(np.float32)
    y_binary      = (df["age"].to_numpy() > 40).astype(int)
    y_multiclass  = np.where(df["segment"] == "low", 0,
                    np.where(df["segment"] == "mid", 1,
                    np.where(df["segment"] == "high", 2, 3)))

    return df, y_regression, y_binary, y_multiclass
