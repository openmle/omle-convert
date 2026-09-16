"""Converter: LightGBM → OMLE.

Supports:
  - LGBMClassifier (binary and multiclass)
  - LGBMRegressor
  - LGBMRanker (learning-to-rank; produces raw scores)
  - lgb.Booster (native API)

Usage::

    import lightgbm as lgb
    from omle_convert.lightgbm import from_lightgbm
    import omle

    clf = lgb.LGBMClassifier(n_estimators=50).fit(X_train, y_train)
    model = from_lightgbm(clf)
    omle.save(model, "model.omle")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas is imported lazily inside the functions that need it
    import pandas as pd

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

def from_lightgbm_text(
    path,
    **kwargs,
) -> omle.OMLEModel:
    """Convert a LightGBM text model file to an OMLEModel.

    Parameters
    ----------
    path:
        Path to a text file saved with ``booster.save_model("model.txt")``.
    **kwargs:
        Forwarded to :func:`from_lightgbm` (``feature_names``, ``target_name``,
        ``class_labels``, ``model_name``).
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is required: pip install lightgbm") from e

    booster = lgb.Booster(model_file=str(path))
    return from_lightgbm(booster, **kwargs)


def from_lightgbm(
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
    """Convert a LightGBM model to an OMLEModel.

    Parameters
    ----------
    model:
        An ``LGBMClassifier``, ``LGBMRegressor``, ``LGBMRanker``, or native ``lgb.Booster``.
    X:
        Optional data (numpy array or DataFrame) used for schema inference
        (column names, dtypes, categorical domains) and to populate
        verification cases, warmup inputs, and sample inputs.  When a
        DataFrame is provided, per-column InputSpecs are generated.  When
        ``None`` (the default), those sections are omitted.
    feature_names:
        Column names for the input tensor.  Inferred from the model when
        possible (``feature_names_in_`` / ``booster.feature_name()``).
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
        import lightgbm  # noqa: F401
    except ImportError as e:
        raise ImportError("lightgbm is required: pip install lightgbm") from e

    # sklearn-compatible bare estimators (LGBMClassifier, LGBMRegressor) delegate
    # to from_sklearn so that to_omle() and from_lightgbm() produce identical
    # models.  Pipelines and models with LightGBM categorical features keep their
    # own handling path (LabelEncode nodes, booster category detection).
    try:
        from sklearn.base import BaseEstimator
        from sklearn.pipeline import Pipeline as _Pipeline
        if isinstance(model, BaseEstimator) and not isinstance(model, _Pipeline):
            _df_check = _as_dataframe(X)
            _has_cats = (
                (_df_check is not None and _has_string_categorical_columns(_df_check))
                or (_df_check is None and _booster_has_categorical_features(_get_booster(model)))
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
                # The model is LightGBM; sklearn is just the scikit-learn-compatible API wrapper.
                converted.metadata.source_frameworks[:] = [
                    omle.SourceFramework(name="lightgbm", version=_lgb_version())
                ]
                return converted
    except ImportError:
        pass

    ct, tree_model = _extract_pipeline_steps(model)
    _df = _as_dataframe(X)
    booster = _get_booster(tree_model)
    model_dump = booster.dump_model()

    task = _task_type(tree_model, model_dump)
    n_classes = _n_classes(tree_model, model_dump, task)

    if ct is not None:
        label_encode_nodes, ct_output_cols, ensemble_inputs, tensor_entries = make_label_encode_nodes_from_ct(ct, _df)
        feat_names = feature_names or [name for name, _ in ct_output_cols]
        if _df is None:
            import numpy as _np
            _df = _ct_stub_df(ct, numeric_dtype=_np.float64)
    else:
        feat_names = feature_names or (_df.columns.tolist() if _df is not None else None) or _feature_names(tree_model, model_dump)
        if _df is None:
            cat_map = _pandas_categorical_from_booster(booster, model_dump)
            if cat_map:
                _df = _lgb_cat_stub_df(feat_names, cat_map)
        label_encode_nodes, tensor_entries = make_label_encode_nodes(_df)
        ensemble_inputs = make_node_inputs(feat_names, _df)

    feat_index = make_feature_index(feat_names)

    if class_labels is None:
        class_labels = _class_labels(tree_model, n_classes)

    lgb_version = _lgb_version()
    trees, tree_group = _convert_trees(model_dump, feat_index, n_classes, task, tensor_entries)
    ensemble = _make_ensemble(trees, tree_group, task)

    ensemble_node = omle.Node(
        name=_estimator_node_name(tree_model),
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=ensemble_inputs,
        outputs=make_node_outputs(task, n_classes),
        tree_ensemble=ensemble,
    )
    nodes = label_encode_nodes + [ensemble_node]

    metadata = make_metadata("lightgbm", lgb_version, model_name, version=model_version, copyright=copyright)
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
        inputs=make_input_specs(feat_names, _df, feature_dtype=omle.DataType.FLOAT64,
                               input_dtype=numpy_input_dtype(X)),
        outputs=make_output_specs(task, n_classes),
        model_schema=make_model_schema(feat_names, task, target_name, class_labels, _df,
                                       feature_dtype=omle.DataType.FLOAT64),
        nodes=nodes,
        tensor_entries=tensor_entries,
    )

    if X is not None:
        output_map = _get_lgb_outputs(model, booster, X, task, n_classes) if n_verify > 0 else None
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
        return "lgb_booster"
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
    """True if the LightGBM booster has any categorical features."""
    try:
        model_dump = booster.dump_model()
        return bool(model_dump.get("pandas_categorical"))
    except Exception:
        return False


def _pandas_categorical_from_booster(booster, model_dump: dict) -> dict[str, list[str]]:
    """Return {feature_name: [category_strings]} for models trained with pandas CategoricalDtype.

    LightGBM stores the pandas category arrays in dump_model()['pandas_categorical'] and
    records the feature indices in the model text as '[categorical_feature: i,j,...]'.
    Returns an empty dict when the model was not trained with pandas categoricals.
    """
    pandas_cats = model_dump.get("pandas_categorical")
    if not pandas_cats:
        return {}

    import re
    text = booster.model_to_string()
    m = re.search(r'\[categorical_feature:\s*([\d,\s]+)\]', text)
    if m is None:
        return {}

    indices = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
    feat_names = booster.feature_name()

    result = {}
    for idx, cats in zip(indices, pandas_cats, strict=True):
        if idx < len(feat_names):
            result[feat_names[idx]] = [str(c) for c in cats]
    return result


def _lgb_cat_stub_df(feature_names: list[str], cat_map: dict[str, list[str]]) -> "pd.DataFrame":
    """Build a one-row stub DataFrame for schema inference when X is not provided.

    Categorical columns get pd.CategoricalDtype with the stored labels so that
    make_label_encode_nodes and make_input_specs emit the correct STRING InputSpecs
    and LabelEncoder nodes. Non-categorical columns get float64 (LightGBM default).
    """
    import numpy as np
    import pandas as pd

    data = {}
    for name in feature_names:
        if name in cat_map:
            data[name] = pd.Categorical([None], categories=cat_map[name])
        else:
            data[name] = np.array([np.nan], dtype=np.float64)
    return pd.DataFrame(data)


def _get_booster(model):
    # sklearn API: LGBMClassifier/LGBMRegressor have .booster_ after fit
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return booster
    return model  # already a lgb.Booster


def _task_type(model, model_dump: dict) -> TaskType:
    # sklearn API
    cls = type(model).__name__
    if "Regressor" in cls:
        return TaskType.REGRESSION
    if "Classifier" in cls:
        n_classes = model_dump.get("num_class", 1)
        return TaskType.MULTICLASS if n_classes > 2 else TaskType.BINARY

    # native booster: inspect objective
    objective = model_dump.get("objective", "")
    obj = objective.lower() if isinstance(objective, str) else ""
    if "regression" in obj or "mse" in obj or "mae" in obj or "huber" in obj:
        return TaskType.REGRESSION
    if "multiclass" in obj or "softmax" in obj or "softprob" in obj:
        return TaskType.MULTICLASS
    if "binary" in obj or "cross_entropy" in obj:
        return TaskType.BINARY
    return TaskType.REGRESSION


def _n_classes(model, model_dump: dict, task: TaskType) -> int:
    if task == TaskType.REGRESSION:
        return 1
    n = getattr(model, "n_classes_", None)
    if n is not None:
        return int(n)
    num_class = model_dump.get("num_class", 1)
    return int(num_class) if num_class > 1 else 2


def _feature_names(model, model_dump: dict) -> list[str]:
    # sklearn API
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None:
        return [str(f) for f in fn]
    # native booster: feature_names in dump
    fn = model_dump.get("feature_names")
    if fn:
        return list(fn)
    # fallback
    n = model_dump.get("max_feature_idx", -1) + 1
    return [f"f{i}" for i in range(n)] if n > 0 else ["f0"]


def _class_labels(model, n_classes: int) -> list:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return list(classes)
    return list(range(n_classes))


def _lgb_version() -> str:
    try:
        import lightgbm
        return lightgbm.__version__
    except Exception:
        return "unknown"


def _make_ensemble(
    trees: list[Tree],
    tree_group: list[int],
    task: TaskType,
) -> TreeEnsemble:
    post = {
        TaskType.REGRESSION:  PostTransform.POST_TRANSFORM_UNSPECIFIED,
        TaskType.BINARY:      PostTransform.SIGMOID_BINARY,
        TaskType.MULTICLASS:  PostTransform.SOFTMAX,
    }[task]

    return TreeEnsemble(
        task_type=task,
        trees=trees,
        aggregation=TreeAggregation.SUM,
        post_transform=post,
        tree_group=tree_group,
    )


# ── Tree conversion ───────────────────────────────────────────────────────────

def _convert_trees(
    model_dump: dict,
    feat_index: dict[str, int],
    n_classes: int,
    task: TaskType,
    tensor_entries: list,
) -> tuple[list[Tree], list[int]]:
    tree_info_list: list[dict] = model_dump.get("tree_info", [])
    trees = [_parse_lgb_tree(ti, tensor_entries) for ti in tree_info_list]

    num_tree_per_iter = model_dump.get("num_tree_per_iteration", 1)
    if num_tree_per_iter > 1:
        tree_group = [i % num_tree_per_iter for i in range(len(trees))]
    else:
        tree_group = []
    return trees, tree_group


def _lgb_decision_type_to_split_op(dt: str) -> TreeSplitOp:
    mapping = {
        "<=": TreeSplitOp.LESS_OR_EQUAL,
        "<":  TreeSplitOp.LESS_THAN,
        ">=": TreeSplitOp.GREATER_OR_EQUAL,
        ">":  TreeSplitOp.GREATER_THAN,
        "==": TreeSplitOp.EQUAL,
        "!=": TreeSplitOp.NOT_EQUAL,
    }
    return mapping.get(dt.strip(), TreeSplitOp.LESS_OR_EQUAL)


def _collect_lgb_nodes(node: dict, out: list) -> None:
    """BFS-collect all nodes from a recursive LightGBM tree_structure."""
    queue = [node]
    while queue:
        n = queue.pop(0)
        out.append(n)
        if "left_child" in n:
            queue.append(n["left_child"])
            queue.append(n["right_child"])


def _parse_lgb_tree(tree_info: dict, tensor_entries: list) -> Tree:
    """Convert one LightGBM tree_info dict to an OMLE Tree (BFS-ordered flat arrays).

    LightGBM exposes the tree as a recursive ``tree_structure`` dict.
    Branch nodes have ``split_feature``, ``threshold``, ``decision_type``,
    ``default_left``, ``left_child``, ``right_child``.
    Leaf nodes have ``leaf_value``.
    """
    root = tree_info.get("tree_structure")
    if root is None:
        # Degenerate: no tree_structure key
        return Tree(
            num_nodes=1,
            node_kind=[TreeNodeKind.LEAF],
            split_feature=[0],
            split_threshold=make_body_tensor_value(
                np.array([0.0]), "split_threshold", tensor_entries),
            split_op=[TreeSplitOp.SPLIT_OP_UNSPECIFIED],
            children_index=[], children_offset=[0],
            children_count=[0], default_child=[0],
            leaf_value=make_body_tensor_value(
                np.array([0.0]), "leaf_value", tensor_entries),
        )

    # BFS traversal: collect nodes in level order
    bfs_nodes: list[dict] = []
    _collect_lgb_nodes(root, bfs_nodes)

    # Map Python object id → sequential BFS index for child lookup
    node_seq: dict[int, int] = {id(n): i for i, n in enumerate(bfs_nodes)}
    num_nodes = len(bfs_nodes)

    node_kind:       list[TreeNodeKind] = []
    split_feature:   list[int]          = []
    split_threshold: list[float]        = []
    split_op:        list[TreeSplitOp]  = []
    children_index:  list[int]          = []
    children_offset: list[int]          = []
    children_count:  list[int]          = []
    default_child:   list[int]          = []
    leaf_value:      list[float]        = []

    child_ptr = 0
    for node in bfs_nodes:
        children_offset.append(child_ptr)

        if "leaf_value" in node:
            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            children_count.append(0)
            default_child.append(0)
            leaf_value.append(float(node["leaf_value"]))
        else:
            node_kind.append(TreeNodeKind.BRANCH)
            split_feature.append(int(node.get("split_feature", 0)))
            try:
                split_threshold.append(float(node.get("threshold", 0.0)))
            except (TypeError, ValueError):
                split_threshold.append(0.0)
            split_op.append(_lgb_decision_type_to_split_op(
                str(node.get("decision_type", "<="))
            ))

            left_node  = node["left_child"]
            right_node = node["right_child"]
            left_seq   = node_seq[id(left_node)]
            right_seq  = node_seq[id(right_node)]

            children_index.extend([left_seq, right_seq])
            children_count.append(2)
            child_ptr += 2

            is_default_left = bool(node.get("default_left", True))
            default_child.append(left_seq if is_default_left else right_seq)
            leaf_value.append(0.0)

    return Tree(
        num_nodes=num_nodes,
        node_kind=node_kind,
        split_feature=split_feature,
        split_threshold=make_body_tensor_value(
            np.array(split_threshold), "split_threshold", tensor_entries),
        split_op=split_op,
        children_index=children_index,
        children_offset=children_offset,
        children_count=children_count,
        default_child=default_child,
        leaf_value=make_body_tensor_value(
            np.array(leaf_value), "leaf_value", tensor_entries),
    )


# ── Auxiliary-data helpers ────────────────────────────────────────────────────

def _get_lgb_outputs(
    model,
    booster,
    X,
    task: TaskType,
    n_classes: int,
) -> dict:
    """Run native LightGBM predictions and return a dict of output_name → np.ndarray."""
    import numpy as np
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is required: pip install lightgbm") from e

    if task == TaskType.REGRESSION:
        if hasattr(model, "predict") and not isinstance(model, lgb.Booster):
            pred = np.asarray(model.predict(X), dtype=np.float32)
        else:
            pred = np.asarray(booster.predict(X), dtype=np.float32)
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

    # Native Booster
    proba = np.asarray(booster.predict(X), dtype=np.float32)
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
