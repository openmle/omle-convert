"""Converter: CatBoost → OMLE.

Supports:
  - CatBoostClassifier (binary and multiclass)
  - CatBoostRegressor
  - CatBoost native API (Pool)

Usage::

    import catboost as cb
    from omle_convert.catboost import from_catboost
    import omle

    clf = cb.CatBoostClassifier(n_estimators=50).fit(X_train, y_train)
    model = from_catboost(clf)
    omle.save(model, "model.omle")

Notes
-----
Only ``FloatFeature`` splits are supported.  Models trained with categorical
features that use CatBoost's internal CTR (Counter Target Regularization)
encoding raise ``NotImplementedError``.  If you need categorical feature
support, encode them as numbers before training (e.g. target-encode or
ordinal-encode externally).

CatBoost uses *oblivious* (symmetric) trees where every node at the same
depth applies the same split.  Each such tree is expanded into a full
balanced binary tree for the OMLE representation.
"""

from __future__ import annotations

import json
import os
import re as _re
import tempfile
from typing import Optional, Union

import omle
from omle.ir.bodies import Tree, TreeEnsemble
from omle.ir.enums import (
    PostTransform,
    TreeAggregation,
    TreeNodeKind,
    TreeSplitOp,
)
from omle.ir.tensor import Tensor
from omle.ir.types import TensorValue

from ._common import (
    TaskType,
    make_auxiliary_data,
    make_input_specs,
    make_metadata,
    make_model_schema,
    make_node_inputs,
    make_node_outputs,
    make_operator_imports,
    make_output_specs,
    numpy_input_dtype,
)

# ── Public API ────────────────────────────────────────────────────────────────

def from_catboost_file(
    path,
    **kwargs,
) -> omle.OMLEModel:
    """Convert a saved CatBoost model file to an OMLEModel.

    Parameters
    ----------
    path:
        Path to a CatBoost model file.  Both the native binary format
        (``.cbm``) and the JSON export (``.json``) are accepted.
    **kwargs:
        Forwarded to :func:`from_catboost`.
    """
    try:
        import catboost as cb
    except ImportError as e:
        raise ImportError("catboost is required: pip install catboost") from e

    path = str(path)
    fmt = "json" if path.lower().endswith(".json") else "cbm"
    # Load as a generic CatBoost model first, then decide the subtype.
    model = cb.CatBoost()
    model.load_model(path, format=fmt)
    return from_catboost(model, **kwargs)


def from_catboost(
    model,
    *,
    X=None,
    feature_names: Optional[list[str]] = None,
    target_name: str = "y",
    class_labels: Optional[list] = None,
    model_name: str = "",
    model_version: str = "",
    copyright: str = "",
    n_verify: Union[int, float] = 10,
    n_warmup: Union[int, float] = 10,
    n_warmup_repeat: int = 3,
    n_sample: int = 3,
    verify_atol: float = 1e-4,
    verify_rtol: float = 1e-4,
    random_state: Optional[int] = None,
) -> omle.OMLEModel:
    """Convert a CatBoost model to an OMLEModel.

    Parameters
    ----------
    model:
        A ``CatBoostClassifier``, ``CatBoostRegressor``, or bare ``CatBoost``
        object.
    X:
        Optional data (numpy array or DataFrame) used for schema inference
        (column names, per-column dtypes) and to populate verification cases,
        warmup inputs, and sample inputs.  When ``None`` (the default), those
        sections are omitted.
    feature_names:
        Column names for the input tensor.  Inferred from the model when
        possible (``feature_names_``).
    target_name:
        Name for the target variable in ``ModelSchema``.
    class_labels:
        Class label strings for classification models.  Inferred from
        ``model.classes_`` when available.
    model_name:
        Human-readable name stored in ``ModelMetadata.name``.
    model_version:
        Version string for the model artifact (e.g. ``"1.0.0"``), stored in
        ``ModelMetadata.version``.  Distinct from the framework version.
    copyright:
        Copyright notice stored in ``ModelMetadata.copyright``
        (e.g. ``"© 2025 Acme Corp"``).
    n_verify:
        Rows for verification — ``int`` (absolute) or ``float`` fraction of
        ``len(X)`` (e.g. ``0.1`` = 10 %).  ``0`` / ``0.0`` skips the section.
    n_warmup:
        Rows for warmup — same int-or-float semantics as ``n_verify``.
    n_warmup_repeat:
        How many times the runtime should repeat the warmup pass.
    n_sample:
        Number of single-row sample inputs to store (0 = skip).
    verify_atol:
        Absolute tolerance for verification comparisons.
    verify_rtol:
        Relative tolerance for verification comparisons.
    random_state:
        Seed for random row selection from ``X``.  ``None`` uses the first N
        rows; an integer seeds a reproducible draw.
    """
    try:
        import catboost as cb  # noqa: F401
    except ImportError as e:
        raise ImportError("catboost is required: pip install catboost") from e

    model_json = _load_model_json(model)

    task = _task_type(model)
    n_classes = _n_classes(model, task)
    feat_names = feature_names or _feature_names(model)
    df = _as_dataframe(X)

    if class_labels is None:
        class_labels = _class_labels(model, n_classes)

    scale, bias = _scale_and_bias(model_json)
    feat_index = _float_feature_index_map(model_json)

    trees, tree_group, tensor_entries = _convert_all_trees(
        model_json, feat_index, scale, bias, task, n_classes
    )

    ensemble = TreeEnsemble(
        task_type=task,
        trees=trees,
        aggregation=TreeAggregation.SUM,
        post_transform=_post_transform(task),
        tree_group=tree_group,
    )

    ensemble_node = omle.Node(
        name=_estimator_node_name(model),
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=make_node_inputs(feat_names, df),
        outputs=make_node_outputs(task, n_classes),
        tree_ensemble=ensemble,
    )

    cb_version = _catboost_version()
    omle_model = omle.OMLEModel(
        metadata=make_metadata("catboost", cb_version, model_name, version=model_version, copyright=copyright),
        operator_imports=make_operator_imports([ensemble_node]),
        inputs=make_input_specs(feat_names, df, feature_dtype=omle.DataType.FLOAT32,
                                input_dtype=numpy_input_dtype(X)),
        outputs=make_output_specs(task, n_classes),
        model_schema=make_model_schema(feat_names, task, target_name, class_labels, df,
                                       feature_dtype=omle.DataType.FLOAT32),
        nodes=[ensemble_node],
        tensor_entries=tensor_entries,
    )

    if X is not None:
        output_map = _get_catboost_outputs(model, X, task, n_classes) if n_verify > 0 else None
        make_auxiliary_data(
            omle_model, X, output_map,
            n_verify=n_verify, n_warmup=n_warmup, n_warmup_repeat=n_warmup_repeat,
            n_sample=n_sample, verify_atol=verify_atol, verify_rtol=verify_rtol,
            random_state=random_state,
        )

    return omle_model


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimator_node_name(model) -> str:
    cls = type(model).__name__
    s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', cls)
    return _re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s).lower()


def _as_dataframe(X):
    try:
        import pandas as pd
        return X if isinstance(X, pd.DataFrame) else None
    except ImportError:
        return None


def _load_model_json(model) -> dict:
    """Export model to JSON and parse it, regardless of model subclass."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name
    try:
        model.save_model(fname, format="json")
        with open(fname) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def _task_type(model) -> TaskType:
    try:
        import catboost as cb
        if isinstance(model, cb.CatBoostRegressor):
            return TaskType.REGRESSION
        if isinstance(model, cb.CatBoostClassifier):
            n = getattr(model, "classes_", None)
            if n is not None and len(n) > 2:
                return TaskType.MULTICLASS
            return TaskType.BINARY
    except ImportError:
        pass
    # Bare CatBoost object: use class_params if available
    try:
        params = model.get_params()
        loss = params.get("loss_function", "") or ""
        if "MultiClass" in loss or "MultiLogloss" in loss:
            return TaskType.MULTICLASS
        if "Logloss" in loss or "CrossEntropy" in loss:
            return TaskType.BINARY
    except Exception:
        pass
    return TaskType.REGRESSION


def _n_classes(model, task: TaskType) -> int:
    if task == TaskType.REGRESSION:
        return 1
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return len(classes)
    return 2


def _feature_names(model) -> list[str]:
    fn = getattr(model, "feature_names_", None)
    if fn:
        return [str(f) for f in fn]
    n = getattr(model, "n_features_in_", 0) or 0
    return [f"f{i}" for i in range(n)]


def _class_labels(model, n_classes: int) -> list:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return list(classes)
    return list(range(n_classes))


def _catboost_version() -> str:
    try:
        import catboost
        return catboost.__version__
    except Exception:
        return "unknown"


def _scale_and_bias(model_json: dict) -> tuple[float, list[float]]:
    """Return (scale, bias_list) from scale_and_bias field."""
    sb = model_json.get("scale_and_bias", [1, [0]])
    scale = float(sb[0]) if sb else 1.0
    bias_raw = sb[1] if len(sb) > 1 else [0]
    if isinstance(bias_raw, list):
        bias = [float(b) for b in bias_raw]
    else:
        bias = [float(bias_raw)]
    return scale, bias


def _float_feature_index_map(model_json: dict) -> dict[int, int]:
    """Map float_feature_index → flat_feature_index (position in input vector)."""
    ff_list = model_json.get("features_info", {}).get("float_features", [])
    return {ff["feature_index"]: ff["flat_feature_index"] for ff in ff_list}


def _post_transform(task: TaskType) -> PostTransform:
    return {
        TaskType.REGRESSION:  PostTransform.POST_TRANSFORM_UNSPECIFIED,
        TaskType.BINARY:      PostTransform.SIGMOID_BINARY,
        TaskType.MULTICLASS:  PostTransform.SOFTMAX,
    }[task]


# ── Tree conversion ───────────────────────────────────────────────────────────

def _convert_all_trees(
    model_json: dict,
    feat_index: dict[int, int],
    scale: float,
    bias: list[float],
    task: TaskType,
    n_classes: int,
) -> tuple[list[Tree], list[int], list[omle.TensorEntry]]:
    """Convert all oblivious trees and prepend bias trees for non-zero bias."""
    raw_trees = model_json.get("oblivious_trees", [])
    trees: list[Tree] = []
    tree_group: list[int] = []
    tensor_entries: list[omle.TensorEntry] = []

    # Bias trees (single-leaf trees encoding the intercept per class)
    if task == TaskType.MULTICLASS:
        for c, b in enumerate(bias):
            if b != 0.0:
                bias_tree, te = _make_bias_tree(b, c, n_classes)
                trees.append(bias_tree)
                tree_group.append(c)
                tensor_entries.extend(te)
    else:
        b = bias[0] if bias else 0.0
        if b != 0.0:
            bias_tree, te = _make_bias_tree(b, 0, 1)
            trees.append(bias_tree)
            tensor_entries.extend(te)

    for ob_tree in raw_trees:
        splits = ob_tree["splits"]
        _check_splits(splits)
        depth = len(splits)

        if task == TaskType.MULTICLASS:
            leaf_vals_flat = ob_tree["leaf_values"]
            for c in range(n_classes):
                # interleaved: leaf_vals_flat[leaf_idx * n_classes + c]
                n_leaves = 1 << depth
                class_leaves = [leaf_vals_flat[li * n_classes + c] for li in range(n_leaves)]
                trees.append(_expand_oblivious_tree(splits, class_leaves, scale, feat_index))
                tree_group.append(c)
        else:
            leaf_vals = ob_tree["leaf_values"]
            trees.append(_expand_oblivious_tree(splits, leaf_vals, scale, feat_index))

    return trees, tree_group, tensor_entries


def _check_splits(splits: list[dict]) -> None:
    for s in splits:
        if s.get("split_type") not in ("FloatFeature", "BinarizedFloatFeature"):
            raise NotImplementedError(
                f"CatBoost split type '{s.get('split_type')}' is not supported. "
                "Only FloatFeature splits are supported; models with categorical "
                "features (OnlineCtr splits) are not convertible."
            )


def _expand_oblivious_tree(
    splits: list[dict],
    leaf_values_catboost: list[float],
    scale: float,
    feat_index: dict[int, int],
) -> Tree:
    """Expand one CatBoost oblivious tree into a BFS-ordered OMLE Tree.

    CatBoost's symmetric tree has depth D, with 2^D leaves indexed by:
        leaf_idx = sum(bit_d * 2^d for d in 0..D-1)
    where bit_d = 1 if x[splits[d].feature] > splits[d].border.

    In BFS order the same leaf is reached by path position p where the
    BFS position and CatBoost index are related by bit-reversal of D bits.
    """
    depth = len(splits)
    n_internal = (1 << depth) - 1   # 2^D - 1
    n_nodes = (1 << (depth + 1)) - 1 # 2^{D+1} - 1

    node_kind:       list[TreeNodeKind] = []
    split_feature:   list[int]         = []
    split_threshold: list[float]       = []
    split_op:        list[TreeSplitOp] = []
    children_index:  list[int]         = []
    children_offset: list[int]         = []
    children_count:  list[int]         = []
    default_child:   list[int]         = []
    leaf_value:      list[float]       = []

    child_ptr = 0
    for i in range(n_nodes):
        children_offset.append(child_ptr)

        if i < n_internal:
            # Depth of node i in 0-indexed BFS: floor(log2(i+1))
            d = (i + 1).bit_length() - 1
            sp = splits[d]
            flat_idx = feat_index.get(sp["float_feature_index"], sp["float_feature_index"])

            node_kind.append(TreeNodeKind.BRANCH)
            split_feature.append(flat_idx)
            split_threshold.append(float(sp["border"]))
            split_op.append(TreeSplitOp.LESS_OR_EQUAL)

            left_child  = 2 * i + 1
            right_child = 2 * i + 2
            children_index.extend([left_child, right_child])
            children_count.append(2)
            child_ptr += 2

            # CatBoost: NaN treated as > border → goes right
            nan_treat = sp.get("nan_value_treatment", "AsIs")
            default_child.append(left_child if nan_treat == "AsFalse" else right_child)
            leaf_value.append(0.0)
        else:
            # Leaf node
            bfs_leaf_pos = i - n_internal   # position within leaf level, 0-indexed
            cb_leaf_idx = _bit_reverse(bfs_leaf_pos, depth)
            lv = float(leaf_values_catboost[cb_leaf_idx]) * scale

            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            children_count.append(0)
            default_child.append(0)
            leaf_value.append(lv)

    return Tree(
        num_nodes=n_nodes,
        node_kind=node_kind,
        split_feature=split_feature,
        split_threshold=TensorValue.of_tensor(Tensor(float32_data=split_threshold)),
        split_op=split_op,
        children_index=children_index,
        children_offset=children_offset,
        children_count=children_count,
        default_child=default_child,
        leaf_value=TensorValue.of_tensor(Tensor(float64_data=leaf_value)),
    )


def _bit_reverse(value: int, n_bits: int) -> int:
    """Reverse the bits of `value` over `n_bits` bits."""
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _make_bias_tree(
    bias_value: float,
    class_idx: int,
    n_output_classes: int,
) -> tuple[Tree, list[omle.TensorEntry]]:
    """Single-leaf bias tree contributing `bias_value` as a constant base score."""
    tree = Tree(
        num_nodes=1,
        node_kind=[TreeNodeKind.LEAF],
        split_feature=[0],
        split_threshold=TensorValue.of_tensor(Tensor(float32_data=[0.0])),
        split_op=[TreeSplitOp.SPLIT_OP_UNSPECIFIED],
        children_index=[],
        children_offset=[0],
        children_count=[0],
        default_child=[0],
        leaf_value=TensorValue.of_tensor(Tensor(float64_data=[bias_value])),
    )
    return tree, []


# ── Prediction helpers ────────────────────────────────────────────────────────

def _get_catboost_outputs(model, X, task: TaskType, n_classes: int) -> dict:
    """Run native CatBoost predictions and return output_name → np.ndarray."""
    import numpy as np

    if task == TaskType.REGRESSION:
        pred = np.asarray(model.predict(X)).ravel()
        return {"y_pred": pred}

    proba = np.asarray(model.predict_proba(X))
    if task == TaskType.BINARY:
        return {
            "y_pred": (proba[:, 1] >= 0.5).astype(np.int64),
            "y_prob": proba,
        }
    return {
        "y_pred": np.argmax(proba, axis=1).astype(np.int64),
        "y_prob": proba,
    }
