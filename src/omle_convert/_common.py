"""Shared utilities for omle-convert converters."""

from __future__ import annotations

import re as _re
from typing import Optional, Union

import omle
from omle.ir.enums import TaskType


def _cls_snake(cls) -> str:
    """Convert a class (or class name string) to snake_case."""
    name = cls if isinstance(cls, str) else type(cls).__name__
    s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    return _re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s).lower()


# ── DataFrame column info ──────────────────────────────────────────────────────

def _scalar_for_value(v) -> omle.Scalar:
    """Create a Scalar preserving the Python/NumPy type of *v*."""
    try:
        import numpy as np
        if isinstance(v, (bool, np.bool_)):
            return omle.Scalar.bool(bool(v))
        if isinstance(v, (int, np.integer)):
            return omle.Scalar.int(int(v))
        if isinstance(v, (float, np.floating)):
            return omle.Scalar.float(float(v))
    except ImportError:
        if isinstance(v, bool):
            return omle.Scalar.bool(v)
        if isinstance(v, int):
            return omle.Scalar.int(v)
        if isinstance(v, float):
            return omle.Scalar.float(v)
    return omle.Scalar.string(str(v))


def _make_discrete_domain(categories) -> omle.ValueDomain:
    """Build a ValueDomain from a list of category values of any scalar type."""
    values = [omle.DomainValue(value=_scalar_for_value(c)) for c in categories]
    return omle.ValueDomain(discrete=omle.DiscreteDomain(values=values))


def _cat_data_type(cat_dtype) -> omle.DataType:
    """Map a pandas categories Index dtype to an omle DataType."""
    import pandas as pd
    if pd.api.types.is_bool_dtype(cat_dtype):
        return omle.DataType.BOOL
    if pd.api.types.is_integer_dtype(cat_dtype):
        return omle.DataType.INT64
    if cat_dtype == "float32":
        return omle.DataType.FLOAT32
    if pd.api.types.is_float_dtype(cat_dtype):
        return omle.DataType.FLOAT64
    return omle.DataType.STRING


def _is_str_dtype(dtype) -> bool:
    """Return True if *dtype* holds strings.

    ``dtype == object`` is not sufficient: pandas 3.0 stores string columns in a
    dedicated (Arrow-backed) ``StringDtype`` rather than ``object``, so the plain
    equality check silently reports False there and a string column gets treated
    as numeric. pandas 3.0 requires Python >= 3.11, which is why this only shows
    up on the newer interpreters in the matrix.
    """
    import pandas as pd
    return pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)


def _is_string_categorical(series) -> bool:
    """Return True if *series* holds string/object categories that need integer encoding."""
    import pandas as pd
    dtype = series.dtype
    if pd.api.types.is_object_dtype(dtype):
        return True
    if isinstance(dtype, pd.CategoricalDtype):
        if dtype.categories is None:
            return True  # unknown; assume strings
        return (pd.api.types.is_object_dtype(dtype.categories.dtype) or
                pd.api.types.is_string_dtype(dtype.categories.dtype))
    return False


_NOMINAL_DTYPES = frozenset({omle.DataType.STRING, omle.DataType.BYTES})


def _measure_level_for_dtype(dtype: omle.DataType) -> omle.MeasureLevel:
    if dtype == omle.DataType.BOOL:
        return omle.MeasureLevel.FLAG
    if dtype in _NOMINAL_DTYPES:
        return omle.MeasureLevel.NOMINAL
    return omle.MeasureLevel.CONTINUOUS


def _col_series_info(series) -> tuple[omle.DataType, omle.MeasureLevel, Optional[omle.ValueDomain]]:
    """Return (DataType, MeasureLevel, ValueDomain | None) for a pandas Series.

    String/object categoricals stay as STRING; numeric categoricals use the real
    category dtype with typed scalars in the ValueDomain.
    """
    import pandas as pd
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return omle.DataType.BOOL, omle.MeasureLevel.FLAG, None
    if pd.api.types.is_integer_dtype(dtype):
        return omle.DataType.INT64, omle.MeasureLevel.CONTINUOUS, None
    if dtype == "float32":
        return omle.DataType.FLOAT32, omle.MeasureLevel.CONTINUOUS, None
    if pd.api.types.is_float_dtype(dtype):
        return omle.DataType.FLOAT64, omle.MeasureLevel.CONTINUOUS, None
    if isinstance(dtype, pd.CategoricalDtype):
        measure = omle.MeasureLevel.ORDINAL if dtype.ordered else omle.MeasureLevel.NOMINAL
        if dtype.categories is not None:
            cats = list(dtype.categories)
            data_type = _cat_data_type(dtype.categories.dtype)
        else:
            cats = sorted(series.dropna().unique(), key=str)
            data_type = omle.DataType.STRING
        return data_type, measure, _make_discrete_domain(cats)
    if pd.api.types.is_object_dtype(dtype):
        cats = sorted(str(v) for v in series.dropna().unique())
        return omle.DataType.STRING, omle.MeasureLevel.NOMINAL, _make_discrete_domain(cats)
    return omle.DataType.FLOAT64, omle.MeasureLevel.CONTINUOUS, None


def _try_name_range(names: list[str]) -> Optional[omle.NameRange]:
    """Return a NameRange if *names* follow the pattern ``prefix+str(i)`` for i in 0..n-1, else None."""
    if not names:
        return None
    import re
    m = re.match(r'^(.*?)(\d+)$', names[0])
    if not m:
        return None
    prefix = m.group(1)
    n = len(names)
    if all(name == f"{prefix}{i}" for i, name in enumerate(names)):
        return omle.NameRange(prefix=prefix, start=0, end=n)
    return None


def make_body_tensor_value(
    data,
    base_name: str,
    tensor_entries: list,
    dtype: "omle.DataType" = None,
) -> "omle.TensorValue":
    """Create a TensorValue for a model-body parameter following OMLE_INLINE_TENSOR_LIMIT.

    Used by converters that do not have a Builder (e.g. XGBoost, LightGBM).
    - data            : array-like, already in the correct dtype
    - base_name       : unique name stem for the TensorEntry id
    - tensor_entries  : model's tensor_entries list — appended to when external ref is used
    - dtype           : omle.DataType (FLOAT32 or FLOAT64); inferred when None
    """
    import numpy as np
    try:
        import numpy as _np
        arr = _np.asarray(data)
    except Exception:
        arr = data
    flat = arr.ravel()
    n    = len(flat)

    if dtype is None:
        dtype = omle.DataType.FLOAT32 if (hasattr(arr, 'dtype') and arr.dtype == np.float32) \
                else omle.DataType.FLOAT64

    shape = list(arr.shape) if arr.ndim > 1 else [n]
    ttype = omle.TensorType(dtype=dtype, shape=shape)

    def _make_tensor(name=""):
        if dtype == omle.DataType.FLOAT32:
            return omle.Tensor(name=name, type=ttype,
                                  float32_data=[float(v) for v in flat])
        return omle.Tensor(name=name, type=ttype,
                              float64_data=[float(v) for v in flat])

    if n <= _inline_threshold():
        return omle.TensorValue(tensor=_make_tensor())

    import uuid
    tid = f"{base_name}_{uuid.uuid4().hex[:8]}"
    tensor_entries.append(omle.TensorEntry(id=tid, dense=_make_tensor(tid)))
    return omle.TensorValue(tensor_ref=omle.TensorRef(id=tid))


def _inline_threshold() -> int:
    """Return the max tensor element count for inline-in-attribute storage.

    Tensors with more elements are promoted to named tensor entries and
    referenced via TensorRef.  Override with the OMLE_INLINE_TENSOR_LIMIT
    env var (default 100).  Set to -1 to always inline.
    """
    import os
    val = os.environ.get("OMLE_INLINE_TENSOR_LIMIT")
    if val is None:
        return 100
    try:
        v = int(val)
        return v if v >= 0 else 10 ** 18  # -1 → always inline
    except (ValueError, TypeError):
        return 100


def _str_attr_with_entry(
    attr_name: str,
    base_id: str,
    strings: list[str],
) -> tuple[omle.Attribute, Optional[omle.TensorEntry]]:
    if len(strings) <= _inline_threshold():
        return omle.Attribute(name=attr_name, tensor=omle.Tensor(
            name=attr_name,
            type=omle.TensorType(dtype=omle.DataType.STRING, shape=[len(strings)]),
            string_data=list(strings),
        )), None
    entry = omle.TensorEntry(id=base_id, dense=omle.Tensor(
        name=base_id,
        type=omle.TensorType(dtype=omle.DataType.STRING, shape=[len(strings)]),
        string_data=list(strings),
    ))
    return omle.Attribute(name=attr_name, tensor_ref=omle.TensorRef(id=base_id)), entry


def _int_attr_with_entry(
    attr_name: str,
    base_id: str,
    ints: list[int],
) -> tuple[omle.Attribute, Optional[omle.TensorEntry]]:
    if len(ints) <= _inline_threshold():
        return omle.Attribute(name=attr_name, tensor=omle.Tensor(
            name=attr_name,
            type=omle.TensorType(dtype=omle.DataType.INT64, shape=[len(ints)]),
            int64_data=ints,
        )), None
    entry = omle.TensorEntry(id=base_id, dense=omle.Tensor(
        name=base_id,
        type=omle.TensorType(dtype=omle.DataType.INT64, shape=[len(ints)]),
        int64_data=ints,
    ))
    return omle.Attribute(name=attr_name, tensor_ref=omle.TensorRef(id=base_id)), entry


def _encoded_name(col: str) -> str:
    """Tensor name produced by the LabelEncode node for *col*."""
    return f"{col}_encoded"


def _ct_stub_df(ct, numeric_dtype=None):
    """Build a 1-row stub DataFrame from a fitted ColumnTransformer.

    OrdinalEncoder columns get CategoricalDtype with the fitted categories.
    All other transformer/passthrough columns get *numeric_dtype* (float32 if None).
    Column order follows ct.feature_names_in_ (original fit order).
    Returns None when pandas is not available.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return None

    if numeric_dtype is None:
        numeric_dtype = np.float32

    col_dtypes: dict = {}
    col_index: Optional[list] = list(ct.feature_names_in_) if hasattr(ct, "feature_names_in_") else None

    for step_name, transformer, columns in ct.transformers_:
        if step_name == "remainder":
            if transformer == "drop":
                continue
            for c in columns:
                col = col_index[c] if (isinstance(c, int) and col_index is not None) else str(c)
                col_dtypes[col] = numeric_dtype
        else:
            col_names: list = []
            for c in columns:
                col_names.append(col_index[c] if (isinstance(c, int) and col_index is not None) else str(c))
            if hasattr(transformer, "categories_"):
                for i, col in enumerate(col_names):
                    cats = [str(cat) for cat in transformer.categories_[i]]
                    col_dtypes[col] = pd.CategoricalDtype(categories=cats, ordered=False)
            else:
                for col in col_names:
                    col_dtypes[col] = numeric_dtype

    ordered_cols = (
        [c for c in col_index if c in col_dtypes]
        if col_index is not None
        else list(col_dtypes.keys())
    )

    data: dict = {}
    for col in ordered_cols:
        dtype = col_dtypes[col]
        if isinstance(dtype, pd.CategoricalDtype):
            cats = list(dtype.categories) if dtype.categories is not None else [""]
            data[col] = pd.Categorical([cats[0]], dtype=dtype)
        else:
            data[col] = np.array([0.0], dtype=dtype)

    return pd.DataFrame(data)


def _extract_pipeline_steps(model):
    """If *model* is a sklearn Pipeline return (ColumnTransformer_or_None, final_estimator).

    Walks the pipeline steps (all but last) looking for the first ColumnTransformer.
    Returns (None, model) when *model* is not a Pipeline.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
    except ImportError:
        return None, model
    if not isinstance(model, Pipeline):
        return None, model
    ct = None
    for _, step in model.steps[:-1]:
        if isinstance(step, ColumnTransformer):
            ct = step
            break
    return ct, model.steps[-1][1]


def make_label_encode_nodes_from_ct(
    ct,
    df=None,
) -> tuple[list[omle.Node], list[tuple[str, str]], list[omle.NodeInput], list[omle.TensorEntry]]:
    """Build LabelEncode nodes from a fitted ColumnTransformer.

    Returns ``(label_encode_nodes, ct_output_cols, ensemble_inputs, tensor_entries)`` where:
    - *ct_output_cols* is individual ``(col_name, col_name)`` pairs in CT output order,
      used to derive ``feat_names`` for tree feature-index mapping and schema.
    - *ensemble_inputs* is the model node's inputs: one ``NodeInput`` per OE step (the
      encoded matrix) plus one per passthrough column — no TakeSlots needed.
    """
    label_encode_nodes: list[omle.Node] = []
    ct_output_cols: list[tuple[str, str]] = []
    ensemble_inputs: list[omle.NodeInput] = []
    tensor_entries: list[omle.TensorEntry] = []

    col_index: Optional[list[str]] = None
    if hasattr(ct, "feature_names_in_"):
        col_index = list(ct.feature_names_in_)
    elif df is not None:
        col_index = list(df.columns)

    for step_name, transformer, columns in ct.transformers_:
        if step_name == "remainder":
            if transformer == "drop":
                continue
            for c in columns:
                # sklearn < 1.7 stores remainder cols as int indices; >= 1.7 as names
                if isinstance(c, int):
                    col = col_index[c] if col_index is not None else str(c)
                else:
                    col = str(c)
                ct_output_cols.append((col, col))
                ensemble_inputs.append(omle.NodeInput(name=col))
        else:
            col_names: list[str] = []
            for c in columns:
                if isinstance(c, int):
                    col_names.append(col_index[c] if col_index is not None else str(c))
                else:
                    col_names.append(str(c))

            if hasattr(transformer, "categories_"):
                all_cats: list[str] = []
                offsets: list[int] = [0]
                for i, _col in enumerate(col_names):
                    cats = [str(c) for c in transformer.categories_[i]]
                    all_cats.extend(cats)
                    offsets.append(offsets[-1] + len(cats))
                cats_attr, cats_entry = _str_attr_with_entry(
                    "categories", f"{step_name}_cats", all_cats)
                off_attr, off_entry = _int_attr_with_entry(
                    "category_offsets", f"{step_name}_cat_offsets", offsets)
                if cats_entry:
                    tensor_entries.append(cats_entry)
                if off_entry:
                    tensor_entries.append(off_entry)
                n_cols = len(col_names)
                encoded_matrix = f"{step_name}_encoded"
                matrix_type = omle.TensorType(dtype=omle.DataType.INT64, shape=[-1, n_cols])
                node_name = f"{step_name}_{_cls_snake(transformer)}"
                label_encode_nodes.append(omle.Node(
                    name=node_name,
                    domain="omle.feature",
                    op="OrdinalEncoder",
                    inputs=[omle.NodeInput(name=col) for col in col_names],
                    outputs=[omle.NodeOutput(name=encoded_matrix, type=matrix_type)],
                    attributes=[cats_attr, off_attr],
                ))
                for col in col_names:
                    ct_output_cols.append((col, col))
                ensemble_inputs.append(omle.NodeInput(name=encoded_matrix))
            else:
                for col in col_names:
                    ct_output_cols.append((col, col))
                    ensemble_inputs.append(omle.NodeInput(name=col))

    return label_encode_nodes, ct_output_cols, ensemble_inputs, tensor_entries


def make_label_encode_nodes(df) -> tuple[list[omle.Node], list[omle.TensorEntry]]:
    """Return one LabelEncoder node for all string-categorical columns in *df*.

    The node accepts all string-cat columns as inputs and produces a single
    ``[N, F]`` encoded matrix ``label_encoded``.  Downstream model operators
    receive the matrix directly — no TakeSlots splitting needed.
    Returns empty lists when *df* is None or has no string-categorical columns.
    """
    if df is None:
        return [], []
    import pandas as pd
    col_list: list[str] = []
    all_labels: list[str] = []
    offsets: list[int] = [0]
    for col in df.columns:
        series = df[col]
        if not _is_string_categorical(series):
            continue
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype) and dtype.categories is not None:
            cats = [str(c) for c in dtype.categories]
        else:
            cats = sorted(str(v) for v in series.dropna().unique())
        col_list.append(str(col))
        all_labels.extend(cats)
        offsets.append(offsets[-1] + len(cats))

    if not col_list:
        return [], []

    labels_attr, labels_entry = _str_attr_with_entry(
        "labels", "label_encode_labels", all_labels)
    off_attr, off_entry = _int_attr_with_entry(
        "label_offsets", "label_encode_offsets", offsets)
    tensor_entries: list[omle.TensorEntry] = []
    if labels_entry:
        tensor_entries.append(labels_entry)
    if off_entry:
        tensor_entries.append(off_entry)

    n_cols = len(col_list)
    encoded_matrix = "label_encoded"
    matrix_type = omle.TensorType(dtype=omle.DataType.INT64, shape=[-1, n_cols])
    nodes: list[omle.Node] = [omle.Node(
        name="label_encoder",
        domain="omle.feature",
        op="LabelEncoder",
        inputs=[omle.NodeInput(name=col) for col in col_list],
        outputs=[omle.NodeOutput(name=encoded_matrix, type=matrix_type)],
        attributes=[labels_attr, off_attr],
    )]
    return nodes, tensor_entries


def df_column_info(df) -> list[tuple[str, omle.DataType, omle.MeasureLevel, Optional[omle.ValueDomain]]]:
    """Return [(col_name, DataType, MeasureLevel, ValueDomain|None)] for each column of *df*."""
    return [(str(col), *_col_series_info(df[col])) for col in df.columns]


def _df_uniform_dtype(df) -> Optional[omle.DataType]:
    """Return the shared DataType if all df columns share the same non-STRING dtype, else None."""
    col_info = df_column_info(df)
    if not col_info:
        return None
    dtypes = {dtype for _, dtype, _, _ in col_info}
    if len(dtypes) != 1:
        return None
    dtype = next(iter(dtypes))
    if dtype == omle.DataType.STRING:
        return None
    return dtype


# ── Input / schema builders ───────────────────────────────────────────────────

def make_feature_index(feature_names: list[str]) -> dict[str, int]:
    """Map feature name → 0-based column index."""
    return {name: i for i, name in enumerate(feature_names)}


def make_metadata(
    framework: str,
    framework_version: str,
    model_name: str = "",
    doc_string: str = "",
    version: str = "",
    timestamp: str = "",
    copyright: str = "",
) -> omle.ModelMetadata:
    from datetime import datetime, timezone
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sf = omle.SourceFramework(name=framework, version=framework_version)
    return omle.ModelMetadata(
        format_version="0.1.0",
        name=model_name or framework.lower().replace(" ", "_") + "_model",
        version=version,
        timestamp=ts,
        producer="omle-convert",
        producer_version="0.1.0",
        source_frameworks=[sf],
        doc_string=doc_string,
        copyright=copyright,
    )


def numpy_input_dtype(X) -> Optional[omle.DataType]:
    """Return the OMLE DataType matching a numpy array's dtype; None for non-numpy."""
    try:
        import numpy as np
        if isinstance(X, np.ndarray):
            if X.dtype == np.float32:
                return omle.DataType.FLOAT32
            if X.dtype == np.float64:
                return omle.DataType.FLOAT64
    except ImportError:
        pass
    return None


def make_input_spec(
    n_features: int,
    feature_names: list[str],
    feature_dtype: omle.DataType = omle.DataType.FLOAT32,
    input_dtype: Optional[omle.DataType] = None,
) -> omle.InputSpec:
    """Single-tensor input spec (no DataFrame)."""
    return make_input_specs(feature_names, feature_dtype=feature_dtype,
                            input_dtype=input_dtype)[0]


def make_input_specs(
    feature_names: list[str],
    df=None,
    feature_dtype: omle.DataType = omle.DataType.FLOAT32,
    input_dtype: Optional[omle.DataType] = None,
) -> list[omle.InputSpec]:
    """Return per-column InputSpecs when *df* is a DataFrame, else a single 'X' tensor.

    InputSpec.type.dtype reflects the actual training-data dtype:
    - DataFrame path: actual column dtype (may differ per column).
    - Numpy path: *input_dtype* (the array's real dtype).
    - No training data: *feature_dtype* (framework internal precision).

    Feature.type.dtype (set separately via make_model_schema) always reflects the
    framework's internal *feature_dtype* for float features.
    """
    if df is not None:
        uniform = _df_uniform_dtype(df)
        if uniform is not None:
            return [omle.InputSpec(
                name="X",
                type=omle.TensorType(dtype=uniform, shape=[-1, len(df.columns)]),
            )]
        col_info = df_column_info(df)
        return [
            omle.InputSpec(
                name=name,
                type=omle.TensorType(dtype=dtype, shape=[-1]),
            )
            for name, dtype, _, _domain in col_info
        ]
    n = len(feature_names)
    spec_dtype = input_dtype if input_dtype is not None else feature_dtype
    return [omle.InputSpec(
        name="X",
        type=omle.TensorType(dtype=spec_dtype, shape=[-1, n]),
    )]


def make_node_inputs(feature_names: list[str], df=None) -> list[omle.NodeInput]:
    """Return per-column NodeInputs when *df* is a DataFrame, else a single 'X'.

    All string-categorical columns are encoded together into a single ``label_encoded``
    matrix by the upstream LabelEncoder node; one ``NodeInput`` is emitted at the
    position of the first string-cat column, and subsequent string-cat columns are
    skipped (covered by the matrix columns).  Non-string-cat columns use their name.
    """
    if df is not None:
        if _df_uniform_dtype(df) is not None:
            return [omle.NodeInput(name="X")]
        inputs = []
        matrix_added = False
        for col in df.columns:
            if _is_string_categorical(df[col]):
                if not matrix_added:
                    inputs.append(omle.NodeInput(name="label_encoded"))
                    matrix_added = True
            else:
                inputs.append(omle.NodeInput(name=str(col)))
        return inputs
    return [omle.NodeInput(name="X")]


def make_output_specs(
    task: TaskType,
    n_classes: int = 2,
    dtype: omle.DataType = omle.DataType.FLOAT64,
) -> list[omle.OutputSpec]:
    if task == TaskType.REGRESSION:
        return [omle.OutputSpec(
            name="y_pred",
            role=omle.OutputRole.PREDICTION,
            type=omle.TensorType(dtype=dtype, shape=[-1]),
        )]
    if task == TaskType.CLUSTERING:
        return [omle.OutputSpec(
            name="cluster_id",
            role=omle.OutputRole.ENTITY_ID,
            type=omle.TensorType(dtype=omle.DataType.INT32, shape=[-1]),
        )]
    if task == TaskType.ANOMALY_DETECTION:
        return [omle.OutputSpec(
            name="anomaly_score",
            role=omle.OutputRole.SCORE,
            type=omle.TensorType(dtype=dtype, shape=[-1]),
        )]
    prob_shape = [-1, 2] if task == TaskType.BINARY else [-1, n_classes]
    return [
        omle.OutputSpec(
            name="y_pred",
            role=omle.OutputRole.PREDICTION,
            type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1]),
        ),
        omle.OutputSpec(
            name="y_prob",
            role=omle.OutputRole.PROBABILITY,
            type=omle.TensorType(dtype=dtype, shape=prob_shape),
        ),
    ]


def _dtype_for_class_labels(class_labels: list) -> omle.DataType:
    """Return the omle DataType that matches the Python type of the class labels."""
    if not class_labels:
        return omle.DataType.INT64
    first = class_labels[0]
    try:
        import numpy as np
        if isinstance(first, (bool, np.bool_)):
            return omle.DataType.BOOL
        if isinstance(first, (int, np.integer)):
            return omle.DataType.INT64
        if isinstance(first, (float, np.floating)):
            return omle.DataType.FLOAT64
    except ImportError:
        if isinstance(first, bool):
            return omle.DataType.BOOL
        if isinstance(first, int):
            return omle.DataType.INT64
        if isinstance(first, float):
            return omle.DataType.FLOAT64
    return omle.DataType.STRING


_FLOAT_DTYPES = frozenset({omle.DataType.FLOAT32, omle.DataType.FLOAT64})


def make_model_schema(
    feature_names: list[str],
    task: TaskType,
    target_name: str = "y",
    class_labels: Optional[list] = None,
    df=None,
    target_dtype: omle.DataType = omle.DataType.FLOAT64,
    feature_dtype: omle.DataType = omle.DataType.FLOAT32,
    per_feature_inputs: bool = False,
    no_targets: bool = False,
) -> omle.ModelSchema:
    """Build ModelSchema.

    Feature.type.dtype always reflects the framework's internal *feature_dtype*
    for float columns.  InputSpec.type.dtype (set separately) reflects the actual
    training-data dtype when a DataFrame is provided.  Non-float columns (string,
    int, bool) keep their own dtype in Feature.type.
    """
    if df is not None:
        col_info = df_column_info(df)
        uniform = _df_uniform_dtype(df)
        if uniform is not None:
            col_names = [name for name, *_ in col_info]
            nr = _try_name_range(col_names)
            eff_dtype = feature_dtype
            measure = _measure_level_for_dtype(eff_dtype)
            if nr is not None:
                features = [omle.Feature(
                    range=nr,
                    source="X",
                    index=0,
                    type=omle.TensorType(dtype=eff_dtype, shape=[-1]),
                    measure_level=measure,
                )]
            else:
                features = [
                    omle.Feature(
                        name=name,
                        source="X",
                        index=i,
                        type=omle.TensorType(dtype=eff_dtype, shape=[-1]),
                        measure_level=measure,
                    )
                    for i, (name, *_) in enumerate(col_info)
                ]
        else:
            features = [
                omle.Feature(
                    name=name,
                    source=name,
                    index=0,
                    type=omle.TensorType(
                        dtype=feature_dtype if dtype in _FLOAT_DTYPES else dtype,
                        shape=[-1],
                    ),
                    measure_level=measure,
                    domain=domain,
                )
                for name, dtype, measure, domain in col_info
            ]
    else:
        measure = _measure_level_for_dtype(feature_dtype)
        if per_feature_inputs:
            features = [
                omle.Feature(
                    name=name,
                    source=name,
                    index=0,
                    type=omle.TensorType(dtype=feature_dtype, shape=[-1]),
                    measure_level=measure,
                )
                for name in feature_names
            ]
        else:
            nr = _try_name_range(feature_names)
            if nr is not None:
                features = [omle.Feature(
                    range=nr,
                    source="X",
                    index=0,
                    type=omle.TensorType(dtype=feature_dtype, shape=[-1]),
                    measure_level=measure,
                )]
            else:
                features = [
                    omle.Feature(
                        name=name,
                        source="X",
                        index=i,
                        type=omle.TensorType(dtype=feature_dtype, shape=[-1]),
                        measure_level=measure,
                    )
                    for i, name in enumerate(feature_names)
                ]

    if no_targets:
        targets = []
    elif task in (TaskType.ANOMALY_DETECTION, TaskType.CLUSTERING):
        targets = []  # unsupervised — no label column
    elif task == TaskType.REGRESSION:
        targets = [omle.Target(
            name=target_name,
            kind=omle.TargetKind.REGRESSION,
            type=omle.TensorType(dtype=target_dtype, shape=[-1]),
        )]
    elif task == TaskType.BINARY:
        raw = class_labels or [0, 1]
        targets = [omle.Target(
            name=target_name,
            kind=omle.TargetKind.BINARY,
            type=omle.TensorType(dtype=_dtype_for_class_labels(raw), shape=[-1]),
            class_labels=[_scalar_for_value(c) for c in raw],
        )]
    else:
        raw = class_labels or list(range(2))
        targets = [omle.Target(
            name=target_name,
            kind=omle.TargetKind.MULTICLASS,
            type=omle.TensorType(dtype=_dtype_for_class_labels(raw), shape=[-1]),
            class_labels=[_scalar_for_value(c) for c in raw],
        )]

    return omle.ModelSchema(features=features, targets=targets)


_NAMESPACE_VERSION: dict[str, str] = {
    "omle.ml":      "0.1",
    "omle.feature": "0.1",
    "omle.core":    "0.1",
    "omle.text":    "0.1",
}


def make_operator_imports(nodes: list[omle.Node]) -> list[omle.NamespaceImport]:
    """Return one NamespaceImport per distinct node domain, in stable order."""
    seen: dict[str, str] = {}
    for node in nodes:
        ns = node.domain
        if ns and ns not in seen:
            seen[ns] = _NAMESPACE_VERSION.get(ns, "0.1")
    return [omle.NamespaceImport(namespace=ns, version=ver) for ns, ver in seen.items()]


def make_node_outputs(
    task: TaskType,
    n_classes: int = 0,
    dtype: omle.DataType = omle.DataType.FLOAT64,
) -> list[omle.NodeOutput]:
    """Return typed NodeOutput list for an estimator node.

    ``n_classes=0`` means unknown; the probability shape dimension is set to
    ``-1`` (dynamic).  Pass the actual class count for a precise annotation.
    """
    if task == TaskType.REGRESSION:
        return [omle.NodeOutput(
            name="y_pred", role=omle.OutputRole.PREDICTION,
            type=omle.TensorType(dtype=dtype, shape=[-1]),
        )]
    if task == TaskType.CLUSTERING:
        return [omle.NodeOutput(
            name="cluster_id", role=omle.OutputRole.ENTITY_ID,
            type=omle.TensorType(dtype=omle.DataType.INT32, shape=[-1]),
        )]
    if task == TaskType.ANOMALY_DETECTION:
        return [omle.NodeOutput(
            name="anomaly_score", role=omle.OutputRole.SCORE,
            type=omle.TensorType(dtype=dtype, shape=[-1]),
        )]
    # Classification (binary or multiclass)
    n = n_classes if n_classes > 0 else -1
    prob_shape = [-1, 2] if task == TaskType.BINARY else [-1, n]
    return [
        omle.NodeOutput(
            name="y_pred", role=omle.OutputRole.PREDICTION,
            type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1]),
        ),
        omle.NodeOutput(
            name="y_prob", role=omle.OutputRole.PROBABILITY,
            type=omle.TensorType(dtype=dtype, shape=prob_shape),
        ),
    ]


# ── Auxiliary data (verification / warmup / sample inputs) ────────────────────

def _make_input_entries(
    prefix: str,
    model: omle.OMLEModel,
    X,
    indices,
) -> tuple[list[omle.TensorEntry], list[omle.TensorRef]]:
    """Build TensorEntry+TensorRef for each model input for the given row indices."""
    import numpy as np
    indices = list(indices)
    try:
        import pandas as pd
        _is_df = isinstance(X, pd.DataFrame)
    except ImportError:
        _is_df = False

    n_rows = len(indices)
    entries: list[omle.TensorEntry] = []
    refs: list[omle.TensorRef] = []

    def _is_single_flat(spec) -> bool:
        """True when the spec should be stored as a single tensor (2-D matrix or 1-D string).

        STRING inputs are always flat (one value or sequence per row) regardless
        of shape rank, so they share the single-tensor branch where _to_str handles
        list-of-tokens → space-joined and shape is [n_rows], not [n_rows, 1].
        """
        if spec.type is None:
            return True
        if spec.type.dtype == omle.DataType.STRING:
            return True
        return len(spec.type.shape) != 1

    def _spec_dtype(spec) -> omle.DataType:
        """Return the declared numeric dtype of a spec, defaulting to FLOAT32."""
        if spec.type is not None and spec.type.dtype in (
            omle.DataType.FLOAT64, omle.DataType.INT64, omle.DataType.INT32,
        ):
            return spec.type.dtype
        return omle.DataType.FLOAT32

    def _numeric_tensor(name: str, arr: "np.ndarray", dtype: omle.DataType) -> "omle.Tensor":
        """Build a numeric Tensor using the field that matches dtype."""
        shape = list(arr.shape)
        if dtype == omle.DataType.FLOAT64:
            return omle.Tensor(name=name, float64_data=arr.astype(np.float64).ravel().tolist(),
                                  type=omle.TensorType(dtype=dtype, shape=shape))
        if dtype == omle.DataType.INT64:
            return omle.Tensor(name=name, int64_data=arr.astype(np.int64).ravel().tolist(),
                                  type=omle.TensorType(dtype=dtype, shape=shape))
        if dtype == omle.DataType.INT32:
            return omle.Tensor(name=name, int32_data=arr.astype(np.int32).ravel().tolist(),
                                  type=omle.TensorType(dtype=dtype, shape=shape))
        return omle.Tensor(name=name, float32_data=arr.astype(np.float32).ravel().tolist(),
                              type=omle.TensorType(dtype=omle.DataType.FLOAT32, shape=shape))

    if _is_df:
        chunk = X.iloc[indices]
        single_flat = len(model.inputs) == 1 and _is_single_flat(model.inputs[0])
        if single_flat:
            in_name = model.inputs[0].name
            # String/object column (e.g. StringIndexer input) → STRING tensor
            if in_name in chunk.columns and _is_str_dtype(chunk[in_name].dtype):
                series = chunk[in_name]
                te_id = f"{prefix}_{in_name}"
                rows_in = series.tolist()
                first_in = rows_in[0] if rows_in else None
                if isinstance(first_in, list):
                    # Token list input (e.g. Word2Vec): pad to max length → 2-D STRING tensor.
                    max_len = max((len(r) for r in rows_in), default=0)
                    flat = [str(t) for row in rows_in
                            for t in (row + [""] * (max_len - len(row)))]
                    in_shape = [n_rows, max_len]
                else:
                    # Scalar string input (e.g. Tokenizer sentence column).
                    flat = [str(v) for v in rows_in]
                    in_shape = [n_rows]
                tensor = omle.Tensor(
                    name=in_name,
                    string_data=flat,
                    type=omle.TensorType(dtype=omle.DataType.STRING, shape=in_shape),
                )
                entries.append(omle.TensorEntry(id=te_id, dense=tensor))
                refs.append(omle.TensorRef(id=te_id))
            else:
                dt  = _spec_dtype(model.inputs[0])
                arr = chunk.to_numpy(dtype=np.float64 if dt == omle.DataType.FLOAT64 else np.float32)
                te_id = f"{prefix}_{in_name}"
                entries.append(omle.TensorEntry(id=te_id, dense=_numeric_tensor(in_name, arr, dt)))
                refs.append(omle.TensorRef(id=te_id))
        else:
            for spec in model.inputs:
                col = spec.name
                if col not in chunk.columns:
                    continue
                te_id = f"{prefix}_{col}"
                series = chunk[col]
                is_str = (spec.type is not None and spec.type.dtype == omle.DataType.STRING) \
                         or _is_str_dtype(series.dtype)
                if is_str:
                    tensor = omle.Tensor(
                        name=col,
                        string_data=[str(v) for v in series.tolist()],
                        type=omle.TensorType(dtype=omle.DataType.STRING, shape=[n_rows, 1]),
                    )
                else:
                    dt     = _spec_dtype(spec)
                    tensor = _numeric_tensor(col, series.to_numpy()[:, np.newaxis], dt)
                entries.append(omle.TensorEntry(id=te_id, dense=tensor))
                refs.append(omle.TensorRef(id=te_id))
    else:
        dt0 = _spec_dtype(model.inputs[0]) if model.inputs else omle.DataType.FLOAT32
        np_dtype = np.float64 if dt0 == omle.DataType.FLOAT64 else np.float32
        arr = np.asarray(X, dtype=np_dtype)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        chunk = arr[indices]
        single_flat = len(model.inputs) == 1 and _is_single_flat(model.inputs[0])
        if single_flat:
            in_name = model.inputs[0].name
            te_id   = f"{prefix}_{in_name}"
            entries.append(omle.TensorEntry(id=te_id, dense=_numeric_tensor(in_name, chunk, dt0)))
            refs.append(omle.TensorRef(id=te_id))
        else:
            n_cols = chunk.shape[1]
            for i, spec in enumerate(model.inputs):
                col   = spec.name
                te_id = f"{prefix}_{col}"
                dt    = _spec_dtype(spec)
                col_arr = chunk[:, i:i+1] if i < n_cols else np.zeros((n_rows, 1), dtype=np_dtype)
                entries.append(omle.TensorEntry(id=te_id, dense=_numeric_tensor(col, col_arr, dt)))
                refs.append(omle.TensorRef(id=te_id))

    return entries, refs


def _make_output_entries(
    prefix: str,
    model: omle.OMLEModel,
    output_map: dict,
    indices,
) -> tuple[list[omle.TensorEntry], list[omle.TensorRef]]:
    """Build TensorEntry+TensorRef for expected outputs for the given row indices."""
    import numpy as np
    indices = list(indices)
    entries: list[omle.TensorEntry] = []
    refs: list[omle.TensorRef] = []
    for spec in model.outputs:
        name = spec.name
        if name not in output_map:
            continue
        spec_dtype = spec.type.dtype if spec.type else omle.DataType.FLOAT64
        raw = output_map[name]
        te_id = f"{prefix}_{name}"
        # String / sequence outputs: each element is a str or list-of-str.
        if spec_dtype == omle.DataType.STRING or (
            hasattr(raw, "dtype") and _is_str_dtype(raw.dtype)
        ) or (
            isinstance(raw, list) and raw and isinstance(raw[0], (str, list))
        ):
            rows = [raw[i] for i in indices]
            n_r = len(rows)
            if rows and isinstance(rows[0], list):
                # Array output (e.g. Tokenizer tokens): pad to max length → 2-D STRING tensor.
                max_len = max((len(r) for r in rows), default=0)
                flat = [str(t) for row in rows for t in (row + [""] * (max_len - len(row)))]
                shape = [n_r, max_len]
            else:
                # Scalar string output → 1-D STRING tensor.
                flat = [str(row) for row in rows]
                shape = [n_r]
            tensor = omle.Tensor(
                name=name,
                string_data=flat,
                type=omle.TensorType(dtype=omle.DataType.STRING, shape=shape),
            )
            entries.append(omle.TensorEntry(id=te_id, dense=tensor))
            refs.append(omle.TensorRef(id=te_id))
            continue
        arr = np.asarray(raw)
        arr = arr[indices]
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        n_r, n_c = arr.shape
        if spec_dtype == omle.DataType.FLOAT32:
            tensor = omle.Tensor(
                name=name,
                float32_data=arr.astype(np.float32).ravel().tolist(),
                type=omle.TensorType(dtype=omle.DataType.FLOAT32, shape=[n_r, n_c]),
            )
        elif spec_dtype == omle.DataType.INT32:
            tensor = omle.Tensor(
                name=name,
                int32_data=arr.astype(np.int32).ravel().tolist(),
                type=omle.TensorType(dtype=omle.DataType.INT32, shape=[n_r, n_c]),
            )
        elif spec_dtype == omle.DataType.INT64:
            tensor = omle.Tensor(
                name=name,
                int64_data=arr.astype(np.int64).ravel().tolist(),
                type=omle.TensorType(dtype=omle.DataType.INT64, shape=[n_r, n_c]),
            )
        else:
            tensor = omle.Tensor(
                name=name,
                float64_data=arr.astype(np.float64).ravel().tolist(),
                type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[n_r, n_c]),
            )
        entries.append(omle.TensorEntry(id=te_id, dense=tensor))
        refs.append(omle.TensorRef(id=te_id))
    return entries, refs


def make_auxiliary_data(
    model: omle.OMLEModel,
    X,
    output_map: Optional[dict],
    n_verify: Union[int, float] = 10,
    n_warmup: Union[int, float] = 10,
    n_warmup_repeat: int = 3,
    n_sample: int = 3,
    verify_atol: float = 1e-4,
    verify_rtol: float = 1e-4,
    random_state: Optional[int] = None,
) -> None:
    """Populate model.tensor_entries, .verification, .warmup, .sample_inputs in-place.

    ``n_verify`` and ``n_warmup`` accept either an ``int`` (absolute row count) or a
    ``float`` in ``(0.0, 1.0]`` (fraction of ``len(X)``).  ``0`` / ``0.0`` skips the
    corresponding section.  ``n_sample`` is always an absolute int count.

    When ``n_verify == n_warmup`` both sections share the same batch TensorEntry,
    avoiding duplicate storage.  ``random_state`` controls row selection: ``None``
    uses the first N rows; an integer seeds a reproducible random draw.
    """
    import numpy as np

    from omle.ir.verification import (
        ModelVerification,
        NumericTolerance,
        RuntimeWarmup,
        SampleInputCase,
        SampleInputSet,
        VerificationCase,
        WarmupCase,
    )
    N = len(X)
    all_entries: list[omle.TensorEntry] = []

    def _resolve(n_need: Union[int, float]) -> int:
        if isinstance(n_need, float):
            return max(1, round(N * n_need)) if n_need > 0.0 else 0
        return int(n_need)

    def _pick(n_need: Union[int, float]) -> list[int]:
        n = min(_resolve(n_need), N)
        if n <= 0:
            return []
        if random_state is None:
            return list(range(n))
        rng = np.random.default_rng(random_state)
        return sorted(int(i) for i in rng.choice(N, n, replace=False))

    verification = None
    warmup = None
    share_batch = (n_verify == n_warmup and n_verify > 0)

    if share_batch:
        batch_idx = _pick(n_verify)
        n = len(batch_idx)
        in_e, in_r = _make_input_entries("batch", model, X, batch_idx)
        all_entries.extend(in_e)
        if output_map is not None:
            out_e, out_r = _make_output_entries("batch", model, output_map, batch_idx)
            all_entries.extend(out_e)
            verification = ModelVerification(
                cases=[VerificationCase(
                    inputs=in_r,
                    expected_outputs=out_r,
                    description=f"{n} rows",
                )],
                tolerance=NumericTolerance(
                    atol=omle.Scalar(float_value=float(verify_atol)),
                    rtol=omle.Scalar(float_value=float(verify_rtol)),
                ),
            )
        warmup = RuntimeWarmup(cases=[WarmupCase(
            inputs=in_r,
            repeat=n_warmup_repeat,
            description=f"{n} rows × {n_warmup_repeat}",
        )])
    else:
        if n_verify > 0 and output_map is not None:
            verify_idx = _pick(n_verify)
            n = len(verify_idx)
            in_e, in_r = _make_input_entries("verify", model, X, verify_idx)
            out_e, out_r = _make_output_entries("verify", model, output_map, verify_idx)
            all_entries.extend(in_e)
            all_entries.extend(out_e)
            verification = ModelVerification(
                cases=[VerificationCase(
                    inputs=in_r,
                    expected_outputs=out_r,
                    description=f"{n} rows",
                )],
                tolerance=NumericTolerance(
                    atol=omle.Scalar(float_value=float(verify_atol)),
                    rtol=omle.Scalar(float_value=float(verify_rtol)),
                ),
            )
        if n_warmup > 0:
            warmup_idx = _pick(n_warmup)
            n = len(warmup_idx)
            in_e, in_r = _make_input_entries("warmup", model, X, warmup_idx)
            all_entries.extend(in_e)
            warmup = RuntimeWarmup(cases=[WarmupCase(
                inputs=in_r,
                repeat=n_warmup_repeat,
                description=f"{n} rows × {n_warmup_repeat}",
            )])

    sample_inputs = None
    if n_sample > 0:
        sample_idx = _pick(n_sample)
        cases = []
        for i, row_i in enumerate(sample_idx):
            in_e, in_r = _make_input_entries(f"sample_{i}", model, X, [row_i])
            all_entries.extend(in_e)
            cases.append(SampleInputCase(inputs=in_r, description=f"row {row_i}"))
        sample_inputs = SampleInputSet(cases=cases)

    model.tensor_entries = list(model.tensor_entries or []) + all_entries
    model.verification = verification
    model.warmup = warmup
    model.sample_inputs = sample_inputs
