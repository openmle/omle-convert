# Contributing to omle-convert

Thanks for your interest. This document covers how to work on this repository —
setup, the layout, and the checks a change needs to pass.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## What lives here

Converters that read a trained model from a framework and emit the OMLE
format. The format itself — the proto schema, the operator and function
registries, validation — lives in [omle](https://github.com/openmle/omle),
and this package depends on it.

If your change adds an operator or alters an operator's contract, it belongs in
`omle` first; this repository then emits it.

| Repository | Owns |
|---|---|
| [omle](https://github.com/openmle/omle) | Format, registries, validation, Python SDK |
| [omle-runtime](https://github.com/openmle/omle-runtime) | C++ inference runtime |
| [omle-viewer](https://github.com/openmle/omle-viewer) | Interactive DAG viewer |
| [omle.js](https://github.com/openmle/omle.js) | TypeScript loader and execution engine |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate

# Upgrade pip first: the build needs PEP 517 support, and older pip (macOS
# system Python ships 21.x) falls back to a legacy path that fails.
python -m pip install --upgrade pip

pip install -e ".[dev]"
pytest tests/
```

That runs 1151 tests in about 90 seconds. Install a JDK (17 is what CI uses)
before running them: the `dev` extra pulls in PySpark, and every PySpark test
starts a JVM.

### The Spark suites

`tests/test_spark.py` exercises the **no-Spark** reader — it loads a saved Spark
model directory with pyarrow, without going through a SparkSession.

`tests/test_spark_jvm.py` fits models through a live SparkSession. Both run in
the default command above and both run in CI, on every matrix row.

On Apple Silicon the two SynapseML LightGBM cases fail with
`UnsatisfiedLinkError: lib_lightgbm.dylib ... incompatible architecture`:
SynapseML's macOS JAR carries an x86_64 native library. This is upstream
packaging, not a bug here, and does not occur on the x86_64 Linux runners.

XGBoost on Spark has two saved formats. Reading either needs only the ordinary
`xgboost` package; the difference is in how a test *fixture* gets produced.

| Produced by | Saved as | Covered? |
|---|---|---|
| `xgboost.spark` — native to xgboost since 1.7 | `SparkXGBClassifierModel` (`model/part-*`) | yes, with the `dev` extra |
| XGBoost4J (Scala) | `XGBoostClassificationModel` (`data/part-*`) | **not currently** |

The Scala format is supported by the converter but has no test. Fitting one from
Python used to go through `sparkxgb`, an anonymous PyPI package with no source
repository, no license and `unk <unk@gmail.com>` as its author, so it was dropped
rather than recommended. The `xgboost4j-spark` JAR itself is official (`ml.dmlc`
on Maven Central) — only that Python wrapper was not.

To restore coverage, check a small saved `XGBoostClassificationModel` directory
into `tests/models/` and read it with `from_spark()`. The converter only ever
reads this format, so such a test needs no JVM, no JAR and no wrapper.

### Running the SynapseML LightGBM tests

`pip install synapseml` is all that is needed — the JARs resolve from Maven when
the SparkSession starts, and `conftest.py` picks the coordinate that matches the
PySpark you have installed:

| PySpark | Scala | Maven coordinate |
|---|---|---|
| 3.5.x | 2.12 | `com.microsoft.azure:synapseml_2.12:<ver>` |
| 4.0.x | 2.13 | `com.microsoft.azure:synapseml_2.13:<ver>-spark4.0` |
| 4.1.x | 2.13 | `com.microsoft.azure:synapseml_2.13:<ver>-spark4.1` |

Resolved from `https://mmlspark.blob.core.windows.net/maven`. SynapseML supports
Spark 4 as well as 3.5 — earlier versions of this file claimed otherwise and
skipped the tests on Spark 4, which was wrong.

Set `SYNAPSEML_JAR` to a JAR path to bypass Maven resolution entirely.

**On Apple Silicon these tests fail, and it is not a configuration problem.**
SynapseML bundles `lib_lightgbm.dylib` built for x86_64 only, so the JVM raises
`UnsatisfiedLinkError: ... incompatible architecture (have 'x86_64', need
'arm64')` once it tries to load the native library. Everything up to that point
works. Run them on x86_64 Linux.

The fixtures auto-detect JARs in the usual local caches (`~/.m2`, `~/.ivy2`,
Coursier) and in the `synapseml` install directory.

Note that `tests/conftest.py` imports pandas and numpy unconditionally, so those
are required for any test run.

## Layout

```
src/omle_convert/
  __init__.py        to_omle/export_omle — framework auto-detection
  _common.py         shared model/schema/output construction, verification cases
  sklearn/           pipeline walker, one module per estimator family
  spark/             saved-format reader (pyarrow) and live PySpark path
  xgboost.py         sklearn wrappers, Booster, .json
  lightgbm.py        sklearn wrappers, Booster, .txt
  catboost.py        sklearn wrappers, .cbm / .json
  cli.py             omle-convert entry point
tests/               pytest suite; tests/models/ holds .omle fixtures
```

## Adding a converter

1. Write the conversion in the right module. Build nodes through the family's
   `Builder`, and produce outputs with `make_node_outputs` so roles and dtypes
   stay consistent with what the runtimes expect.
2. Register it for auto-detection in `__init__.py` — `_convert_object` for live
   model objects (dispatch on `type(model).__module__`), `_convert_from_path`
   for files (dispatch on extension, or content when extensions collide, as
   XGBoost and CatBoost JSON do).
3. Validate the output in the test: `omle.validate(model)` must pass. A
   converter that emits an invalid model is a bug even if the numbers are right.
4. Where the framework can score, add a verification case comparing runtime
   output against the framework's own prediction.
5. **Update `README.md`.** It documents every supported class, and is the first
   thing users read.

## Code style

Python follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
enforced by `ruff check .` (configured in `pyproject.toml`; CI runs it).
`ruff check --fix .` handles most findings.

Two deliberate notes:

- **No autoformatter.** `ruff format` would rewrite the aligned dict literals
  and `# ── section ──` dividers this codebase uses.
- **`X` and `N` are not renamed.** `N803`/`N806` are disabled because `X` for a
  design matrix and `N` for a row count are the universal convention in ML code
  — scikit-learn's own public API uses them, and these converters mirror the
  APIs they read. Same for `F`/`C`/`R` aliases of `pyspark.ml` submodules.

Line length (Google's 80 columns) and docstring conventions are not yet enforced
— the codebase is not clean against them. New code should follow both.

## Pre-commit hooks (optional)

```bash
pip install pre-commit
pre-commit install
```

Runs the hygiene hooks plus `ruff`. CI enforces the same things.

## Tests and CI

`.github/workflows/test.yml` runs lint and the non-JVM suite on Python 3.10
through 3.14 for every push and pull request.

Please add a test with any behaviour change. For a bug fix, a test that fails
before the fix is the most useful thing you can include.

## Reporting bugs

Include the converter and framework versions, and the smallest script that
reproduces. If the produced model is wrong rather than absent, attach the output
of `omle inspect model.omle` — it is usually enough to spot the problem.

## License

Contributions are accepted under the [Apache License 2.0](LICENSE), in
accordance with section 5 of that license. There is no separate CLA.
