"""Converter: XGBoost → OMLE.

Supports:
  - XGBClassifier (binary and multiclass)
  - XGBRegressor
  - XGBRFClassifier / XGBRFRegressor (random forest variants)
  - XGBRanker (learning-to-rank; produces raw scores)
  - xgb.Booster (native API)

Usage::

    import xgboost as xgb
    from omle_convert.xgboost import from_xgboost
    import omle

    clf = xgb.XGBClassifier(n_estimators=50).fit(X_train, y_train)
    model = from_xgboost(clf)
    omle.save(model, "model.omle")
"""

from __future__ import annotations

import json
import math
import re as _re
from typing import Optional, Union

import numpy as np

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
    _ct_stub_df,
    _extract_pipeline_steps,
    make_auxiliary_data,
    make_body_tensor_value,
    make_feature_index,
    make_input_specs,
    make_label_encode_nodes,
    make_label_encode_nodes_from_ct,
    make_metadata,
    make_model_schema,
    make_node_inputs,
    make_node_outputs,
    make_operator_imports,
    make_output_specs,
    numpy_input_dtype,
)

# ── Public API ────────────────────────────────────────────────────────────────

def from_xgboost_json(
    path,
    **kwargs,
) -> omle.OMLEModel:
    """Convert an XGBoost JSON model file to an OMLEModel.

    Parameters
    ----------
    path:
        Path to a JSON file saved with ``booster.save_model("model.json")``.
    **kwargs:
        Forwarded to :func:`from_xgboost` (``feature_names``, ``target_name``,
        ``class_labels``, ``model_name``).
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    booster = xgb.Booster()
    booster.load_model(str(path))
    return from_xgboost(booster, **kwargs)


def from_xgboost(
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
    """Convert an XGBoost model to an OMLEModel.

    Parameters
    ----------
    model:
        An ``XGBClassifier``, ``XGBRegressor``, ``XGBRFClassifier``,
        ``XGBRFRegressor``, ``XGBRanker``, or native ``xgb.Booster``.
    X:
        Optional data (numpy array or DataFrame) used for schema inference
        (column names, dtypes, categorical domains) and to populate
        verification cases, warmup inputs, and sample inputs.  When a
        DataFrame is provided, per-column InputSpecs are generated.  When
        ``None`` (the default), those sections are omitted.
    feature_names:
        Column names for the input tensor.  Inferred from the model when
        possible (``feature_names_in_`` / ``booster.feature_names``).
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
        import xgboost as xgb  # noqa: F401
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    # sklearn-compatible bare estimators (XGBClassifier, XGBRegressor) delegate
    # to from_sklearn so that to_omle() and from_xgboost() produce identical
    # models.  Pipelines and models with XGBoost categorical features keep their
    # own handling path (LabelEncode nodes, booster feature_types detection).
    try:
        from sklearn.base import BaseEstimator
        from sklearn.pipeline import Pipeline as _Pipeline
        if isinstance(model, BaseEstimator) and not isinstance(model, _Pipeline):
            _df = _as_dataframe(X)
            _booster = _get_booster(model)
            _has_cats = (
                (_df is not None and _has_string_categorical_columns(_df))
                or (_df is None and _booster_has_categorical_features(_booster))
            )
            if not _has_cats:
                from omle_convert.sklearn import from_sklearn
                converted = from_sklearn(
                    model,
                    X=X,
                    feature_names=feature_names,
                    target_name=target_name,
                    class_labels=class_labels,
                    model_name=model_name,
                    model_version=model_version,
                    n_verify=n_verify,
                    n_warmup=n_warmup,
                    n_warmup_repeat=n_warmup_repeat,
                    n_sample=n_sample,
                    verify_atol=verify_atol,
                    verify_rtol=verify_rtol,
                    random_state=random_state,
                )
                # XGBoost's internal precision is float32; override the default float64.
                for out in converted.outputs:
                    if out.type is not None and out.type.dtype == omle.DataType.FLOAT64:
                        out.type = omle.TensorType(
                            dtype=omle.DataType.FLOAT32, shape=out.type.shape
                        )
                for tgt in converted.model_schema.targets:
                    if tgt.type is not None and tgt.type.dtype == omle.DataType.FLOAT64:
                        tgt.type = omle.TensorType(
                            dtype=omle.DataType.FLOAT32, shape=tgt.type.shape
                        )
                # The model is XGBoost; sklearn is just the scikit-learn-compatible API wrapper.
                converted.metadata.source_frameworks[:] = [
                    omle.SourceFramework(name="xgboost", version=_xgb_version())
                ]
                return converted
    except ImportError:
        pass

    ct, tree_model = _extract_pipeline_steps(model)
    _df = _as_dataframe(X)
    booster = _get_booster(tree_model)
    task = _task_type(tree_model, booster)
    n_classes = _n_classes(tree_model, booster, task)

    if ct is not None:
        label_encode_nodes, ct_output_cols, ensemble_inputs, tensor_entries = make_label_encode_nodes_from_ct(ct, _df)
        feat_names = feature_names or [name for name, _ in ct_output_cols]
        if _df is None:
            import numpy as _np
            _df = _ct_stub_df(ct, numeric_dtype=_np.float32)
    else:
        feat_names = feature_names or (_df.columns.tolist() if _df is not None else None) or _feature_names(tree_model, booster)
        if _df is None and booster.feature_types and any(t == "c" for t in booster.feature_types):
            _df = _xgb_cat_stub_df(feat_names, booster.feature_types)
        label_encode_nodes, tensor_entries = make_label_encode_nodes(_df)
        ensemble_inputs = make_node_inputs(feat_names, _df)

    feat_index = make_feature_index(feat_names)

    if class_labels is None:
        class_labels = _class_labels(tree_model, n_classes)

    xgb_version = _xgb_version()
    trees, tree_group = _convert_trees(booster, feat_index, n_classes, task, tensor_entries)
    base_scores = _base_scores(tree_model, booster)
    ensemble = _make_ensemble(trees, tree_group, task, base_scores)

    ensemble_node = omle.Node(
        name=_estimator_node_name(tree_model),
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=ensemble_inputs,
        outputs=make_node_outputs(task, n_classes),
        tree_ensemble=ensemble,
    )
    nodes = label_encode_nodes + [ensemble_node]

    metadata = make_metadata("xgboost", xgb_version, model_name, version=model_version, copyright=copyright)
    if ct is not None:
        try:
            from omle_convert.sklearn import _sk_version
            metadata.source_frameworks.append(
                omle.SourceFramework(name="sklearn", version=_sk_version(), role="preprocessing")
            )
        except ImportError:
            pass

    omle_model = omle.OMLEModel(
        metadata=metadata,
        operator_imports=make_operator_imports(nodes),
        inputs=make_input_specs(feat_names, _df, feature_dtype=omle.DataType.FLOAT32,
                               input_dtype=numpy_input_dtype(X)),
        outputs=make_output_specs(task, n_classes, dtype=omle.DataType.FLOAT32),
        model_schema=make_model_schema(feat_names, task, target_name, class_labels, _df,
                                       target_dtype=omle.DataType.FLOAT32,
                                       feature_dtype=omle.DataType.FLOAT32),
        nodes=nodes,
        tensor_entries=tensor_entries,
    )

    if X is not None:
        X_arr = _prepare_X_arr(X)
        output_map = _get_xgb_outputs(model, booster, X, X_arr, task, n_classes) if n_verify > 0 else None
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
    if cls == "Booster":
        return "xgb_booster"
    s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', cls)
    return _re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s).lower()


def _as_dataframe(X):
    """Return X if it is a pandas DataFrame, else None."""
    try:
        import pandas as pd
        return X if isinstance(X, pd.DataFrame) else None
    except ImportError:
        return None


def _has_string_categorical_columns(df) -> bool:
    """True if any column of *df* is a string/object/categorical dtype."""
    from omle_convert._common import _is_string_categorical
    return any(_is_string_categorical(df[col]) for col in df.columns)


def _booster_has_categorical_features(booster) -> bool:
    """True if the XGBoost booster reports any categorical feature types."""
    ft = getattr(booster, "feature_types", None)
    return ft is not None and any(t == "c" for t in ft)



def _get_booster(model):
    if hasattr(model, "get_booster"):
        return model.get_booster()
    return model  # already a Booster


def _task_type(model, booster) -> TaskType:
    objective = (
        getattr(model, "objective", None)
        or _booster_objective(booster)
        or "reg:squarederror"
    )
    if isinstance(objective, str):
        obj = objective.lower()
        if "binary" in obj or "logistic" in obj:
            return TaskType.BINARY
        if "multi" in obj or "softmax" in obj or "softprob" in obj:
            return TaskType.MULTICLASS
    return TaskType.REGRESSION


def _booster_objective(booster) -> Optional[str]:
    try:
        cfg = json.loads(booster.save_config())
        return (cfg.get("learner", {})
                   .get("objective", {})
                   .get("name"))
    except Exception:
        return None


def _n_classes(model, booster, task: TaskType) -> int:
    if task == TaskType.REGRESSION:
        return 1
    n = getattr(model, "n_classes_", None)
    if n is not None:
        return int(n)
    try:
        cfg = json.loads(booster.save_config())
        num_class = int(
            cfg.get("learner", {})
               .get("learner_model_param", {})
               .get("num_class", 2)
        )
        return num_class if num_class > 1 else 2
    except Exception:
        return 2


def _feature_names(model, booster) -> list[str]:
    # sklearn API
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None:
        return [str(f) for f in fn]
    # native booster
    fn = getattr(booster, "feature_names", None)
    if fn:
        return list(fn)
    # fallback: numbered
    try:
        n = booster.num_features()
    except Exception:
        n = 0
    return [f"f{i}" for i in range(n)]


def _xgb_cat_stub_df(feature_names: list[str], feature_types: list[str]):
    """Build a one-row stub DataFrame when no X is provided but the booster has
    categorical features (feature_types contains 'c').

    Categorical columns get dtype int64; others get float32. This drives
    make_input_specs / make_node_inputs to emit per-column InputSpecs with the
    correct dtypes instead of a single FLOAT32 matrix 'X'.
    """
    import numpy as np
    import pandas as pd

    data = {}
    for name, ftype in zip(feature_names, feature_types, strict=True):
        if ftype == "c":
            data[name] = np.array([0], dtype=np.int64)
        else:
            data[name] = np.array([0.0], dtype=np.float32)
    return pd.DataFrame(data)


def _class_labels(model, n_classes: int) -> list:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return list(classes)
    return list(range(n_classes))


def _xgb_version() -> str:
    try:
        import xgboost
        return xgboost.__version__
    except Exception:
        return "unknown"


def _base_scores(model, booster) -> Optional[list[float]]:
    """Intercepts fitted by the booster, one per output.

    Returns a single-element list for regression and binary, and one entry per
    class for multiclass. None when the booster reports no intercept at all.
    """
    # sklearn API. Set explicitly by the caller, so always a single value.
    bs = getattr(model, "base_score", None)
    if bs is not None:
        try:
            return [float(bs)]
        except (TypeError, ValueError):
            pass
    # Otherwise read what the booster actually fitted.
    try:
        cfg = json.loads(booster.save_config())
        raw = (cfg.get("learner", {})
                  .get("learner_model_param", {})
                  .get("base_score"))
        if raw is not None:
            return _parse_base_scores(raw)
    except Exception:
        pass
    return None


def _parse_base_scores(raw) -> Optional[list[float]]:
    """Parse the base_score reported by booster.save_config().

    XGBoost < 2.0 reported a bare scalar ("5E-1"). Since 2.0 the intercept is
    fitted rather than fixed, and save_config() renders it as a JSON array in a
    string: "[2.0719469E0]" for a single output, or one entry per class for
    multiclass. float() rejects the bracketed form, and because the caller
    wraps this in a bare except the failure was silent — the model converted
    cleanly and simply predicted every row low by the intercept.
    """
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            return [float(v) for v in json.loads(raw)]
        return [float(raw)]
    raise TypeError(f"unsupported base_score type: {type(raw).__name__}")


def _make_ensemble(
    trees: list[Tree],
    tree_group: list[int],
    task: TaskType,
    base_scores: Optional[list[float]],
) -> TreeEnsemble:
    post = {
        TaskType.REGRESSION:  PostTransform.POST_TRANSFORM_UNSPECIFIED,
        TaskType.BINARY:      PostTransform.SIGMOID_BINARY,
        TaskType.MULTICLASS:  PostTransform.SOFTMAX,
    }[task]

    # XGBoost reports the binary intercept in output space (a probability), so
    # it is converted to margin space and the runtime can add it straight to the
    # raw tree sum before post_transform. Multiclass intercepts are already in
    # margin space — they are signed and sum to roughly zero — and regression
    # has no transform, so both pass through unchanged.
    margin = base_scores
    if base_scores is not None and task == TaskType.BINARY:
        # Clamp to a valid probability before the logit, to avoid log(0).
        margin = [math.log(p / (1.0 - p))
                  for p in (max(1e-7, min(1.0 - 1e-7, b)) for b in base_scores)]

    return TreeEnsemble(
        task_type=task,
        trees=trees,
        aggregation=TreeAggregation.SUM,
        post_transform=post,
        base_scores=(TensorValue.of_tensor(Tensor(float64_data=list(margin)))
                     if margin is not None else None),
        tree_group=tree_group,
    )


# ── Tree conversion ───────────────────────────────────────────────────────────

def _num_parallel_tree(booster) -> int:
    """Return num_parallel_tree from the booster config (1 for standard GBT, >1 for RF variants)."""
    try:
        cfg = json.loads(booster.save_config())
        return int(
            cfg.get("learner", {})
               .get("gradient_booster", {})
               .get("gbtree_model_param", {})
               .get("num_parallel_tree", 1)
        )
    except Exception:
        return 1


def _convert_trees(
    booster,
    feat_index: dict[str, int],
    n_classes: int,
    task: TaskType,
    tensor_entries: list,
) -> tuple[list[Tree], list[int]]:
    raw = booster.get_dump(dump_format="json")
    trees = [_parse_xgb_tree(json.loads(t), feat_index, tensor_entries) for t in raw]
    if task == TaskType.MULTICLASS and n_classes > 1:
        # For RF variants (num_parallel_tree > 1), trees are grouped as blocks of
        # num_parallel_tree per class within each boosting round. For standard GBT
        # (num_parallel_tree=1) this reduces to i % n_classes.
        npt = _num_parallel_tree(booster)
        tree_group = [i // npt % n_classes for i in range(len(trees))]
    else:
        tree_group = []
    return trees, tree_group


def _collect_nodes(node: dict, out: dict[int, dict]) -> None:
    """Recursively collect all nodes from an XGBoost JSON tree into a dict."""
    out[node["nodeid"]] = node
    for child in node.get("children", []):
        _collect_nodes(child, out)


def _parse_xgb_tree(tree_json: dict, feat_index: dict[str, int], tensor_entries: list) -> Tree:
    """Convert one XGBoost JSON tree to an OMLE Tree (BFS-ordered flat arrays).

    Numeric splits: split_condition is a float scalar → LESS_THAN.
    Categorical splits: split_condition is a list of integer codes → IN_SET.
      The code set is stored in category_set / category_set_offset / category_set_count.
    """
    # Collect all nodes
    nodes: dict[int, dict] = {}
    _collect_nodes(tree_json, nodes)

    # BFS traversal from root (nodeid 0) to get a stable sequential ordering
    bfs_order: list[int] = []
    queue = [0]
    while queue:
        nid = queue.pop(0)
        bfs_order.append(nid)
        nd = nodes[nid]
        if "leaf" not in nd:
            queue.append(nd["yes"])
            queue.append(nd["no"])

    id_map: dict[int, int] = {orig: seq for seq, orig in enumerate(bfs_order)}
    num_nodes = len(bfs_order)

    node_kind:            list[TreeNodeKind] = []
    split_feature:        list[int]          = []
    split_threshold:      list[float]        = []
    split_op:             list[TreeSplitOp]  = []
    category_set_offset:  list[int]          = []
    category_set_count:   list[int]          = []
    category_set:         list[int]          = []
    children_index:       list[int]          = []
    children_offset:      list[int]          = []
    children_count:       list[int]          = []
    default_child:        list[int]          = []
    leaf_value:           list[float]        = []

    child_ptr = 0
    cat_ptr = 0
    for orig_id in bfs_order:
        nd = nodes[orig_id]
        children_offset.append(child_ptr)
        category_set_offset.append(cat_ptr)

        if "leaf" in nd:
            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            category_set_count.append(0)
            children_count.append(0)
            default_child.append(0)
            leaf_value.append(float(nd["leaf"]))
        else:
            node_kind.append(TreeNodeKind.BRANCH)

            feat_str: str = nd["split"]
            if feat_str in feat_index:
                fidx = feat_index[feat_str]
            elif feat_str.startswith("f") and feat_str[1:].isdigit():
                fidx = int(feat_str[1:])
            else:
                fidx = 0

            split_feature.append(fidx)
            cond = nd["split_condition"]
            if isinstance(cond, list):
                # Categorical split: split_condition is a list of integer codes that
                # map to the YES (left) child. Uses IN_SET semantics.
                codes = [int(c) for c in cond]
                split_threshold.append(0.0)
                split_op.append(TreeSplitOp.IN_SET)
                category_set.extend(codes)
                category_set_count.append(len(codes))
                cat_ptr += len(codes)
            else:
                # Numeric split: go to "yes" branch when x < split_condition
                split_threshold.append(float(cond))
                split_op.append(TreeSplitOp.LESS_THAN)
                category_set_count.append(0)

            yes_seq = id_map[nd["yes"]]
            no_seq  = id_map[nd["no"]]
            missing_seq = id_map[nd.get("missing", nd["yes"])]

            children_index.extend([yes_seq, no_seq])
            children_count.append(2)
            child_ptr += 2
            default_child.append(missing_seq)
            leaf_value.append(0.0)

    return Tree(
        num_nodes=num_nodes,
        node_kind=node_kind,
        split_feature=split_feature,
        split_threshold=make_body_tensor_value(
            np.array(split_threshold, dtype=np.float32),
            "split_threshold", tensor_entries,
            dtype=omle.DataType.FLOAT32,
        ),
        split_op=split_op,
        category_set_offset=category_set_offset,
        category_set_count=category_set_count,
        category_set=category_set,
        children_index=children_index,
        children_offset=children_offset,
        children_count=children_count,
        default_child=default_child,
        leaf_value=make_body_tensor_value(
            np.array(leaf_value, dtype=np.float32),
            "leaf_value", tensor_entries,
            dtype=omle.DataType.FLOAT32,
        ),
    )


# ── Auxiliary-data helpers ────────────────────────────────────────────────────

def _prepare_X_arr(X) -> "np.ndarray":
    """Convert X to a float32 numpy array (numeric columns only) for booster prediction."""
    import numpy as np
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            num = X.select_dtypes(include=["number", "bool"]).values
            return num.astype(np.float32)
    except ImportError:
        pass
    arr = np.asarray(X, dtype=np.float32)
    return arr if arr.ndim == 2 else arr[np.newaxis, :]


def _get_xgb_outputs(
    model,
    booster,
    X,
    X_arr: "np.ndarray",
    task: TaskType,
    n_classes: int,
) -> dict:
    """Run native XGBoost predictions and return a dict of output_name → np.ndarray."""
    import numpy as np
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    if task == TaskType.REGRESSION:
        if hasattr(model, "predict") and not isinstance(model, xgb.Booster):
            pred = np.asarray(model.predict(X), dtype=np.float32)
        else:
            pred = booster.predict(xgb.DMatrix(X_arr)).astype(np.float32)
        return {"y_pred": pred}

    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(X), dtype=np.float32)
        if task == TaskType.BINARY:
            return {
                "y_pred": (raw[:, 1] >= 0.5).astype(np.float32),
                "y_prob": raw,
            }
        return {
            "y_pred": np.argmax(raw, axis=1).astype(np.float32),
            "y_prob": raw,
        }

    # Bare Booster
    proba = booster.predict(xgb.DMatrix(X_arr)).astype(np.float32)
    if task == TaskType.BINARY:
        return {
            "y_pred": (proba >= 0.5).astype(np.float32),
            "y_prob": np.column_stack([1 - proba, proba]),
        }
    y_prob = proba.reshape(-1, n_classes)
    return {
        "y_pred": np.argmax(y_prob, axis=1).astype(np.float32),
        "y_prob": y_prob,
    }
