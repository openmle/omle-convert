"""Read Spark ML saved models from disk without a Spark/JVM dependency.

Uses pyarrow to read Parquet files and plain JSON for metadata.
All Spark VectorUDT / MatrixUDT structs are decoded locally.

Supported classes:
  Transformers : StandardScalerModel, MinMaxScalerModel, MaxAbsScalerModel,
                 Normalizer, Binarizer, StringIndexerModel, OneHotEncoderModel,
                 Bucketizer, ImputerModel, PCAModel, ElementwiseProduct, VectorAssembler,
                 VectorSlicer, ChiSqSelectorModel, UnivariateFeatureSelectorModel,
                 VarianceThresholdSelectorModel, SQLTransformer,
                 Tokenizer, RegexTokenizer, StopWordsRemover, NGram,
                 CountVectorizerModel, HashingTF, IDFModel, Word2VecModel
  Estimators   : LinearRegressionModel, IsotonicRegressionModel, LogisticRegressionModel,
                 DecisionTree{Classification,Regression}Model,
                 RandomForest{Classification,Regression}Model,
                 GBT{Classification,Regression}Model,
                 NaiveBayesModel, KMeansModel, LinearSVCModel,
                 MultilayerPerceptronClassificationModel, GaussianMixtureModel,
                 GeneralizedLinearRegressionModel, OneVsRestModel,
                 CrossValidatorModel, TrainValidationSplitModel
"""
from __future__ import annotations

import glob as _glob
import json
import math
import os
import re as _re
from typing import List, Optional

import numpy as np

import omle
from omle.ir.bodies import (
    SVM,
    BernoulliNaiveBayes,
    Clustering,
    DenseLayer,
    GaussianMixtureClustering,
    GaussianNaiveBayes,
    Linear,
    LinearSVM,
    MultinomialNaiveBayes,
    NeuralNetwork,
    PrototypeClustering,
    Tree,
    TreeEnsemble,
)
from omle.ir.bodies import (
    NaiveBayes as NaiveBayesBody,
)
from omle.ir.enums import (
    CovarianceType,
    DistanceMeasure,
    NeuralNetworkActivation,
    PostTransform,
    TreeAggregation,
    TreeNodeKind,
    TreeSplitOp,
)
from omle.ir.enums import (
    DataType as IRDataType,
)
from omle.ir.tensor import Tensor as IRTensor
from omle.ir.types import TensorType as IRTensorType
from omle_convert._common import (
    TaskType,
    make_model_schema,
    make_node_outputs,
)

from ._builder import Builder, core_node, feature_node
from ._sql_transforms import apply_sql_transformer


def _f64_tensor(arr) -> IRTensor:
    a = np.asarray(arr, dtype=np.float64).ravel()
    return IRTensor(float64_data=list(a), type=IRTensorType(dtype=IRDataType.FLOAT64, shape=[len(a)]))


# ── Low-level I/O ──────────────────────────────────────────────────────────────

def _read_metadata(model_path: str) -> dict:
    meta_dir   = os.path.join(model_path, "metadata")
    candidates = sorted(_glob.glob(os.path.join(meta_dir, "part-*")))
    if not candidates:
        raise FileNotFoundError(f"No metadata part file found in {meta_dir}")
    with open(candidates[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_parquet_dict(model_path: str, subdir: str = "data") -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for no-Spark model reading. "
            "Install it with: pip install pyarrow"
        ) from None
    data_dir = os.path.join(model_path, subdir)
    return pq.read_table(data_dir).to_pydict()


def _short_class(full_class: str) -> str:
    return full_class.rsplit(".", 1)[-1]



def _cls_to_snake(cls: str) -> str:
    """Convert a Spark ML class name to snake_case node name.

    e.g. StandardScalerModel → standard_scaler_model
         RandomForestClassificationModel → random_forest_classification_model
    """
    s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', cls)
    return _re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s).lower()


def _column_of(data: dict, want: type) -> Optional[list]:
    """First column whose values are all of `want`.

    Used to read a frame Spark saved without column names, where the fields
    arrive as _1/_2/_3 and only their types distinguish them.
    """
    for values in data.values():
        if not values:
            continue
        # bool is a subclass of int, so exclude it explicitly rather than let a
        # boolean column masquerade as the tree id.
        if want is int and all(isinstance(v, int) and not isinstance(v, bool)
                               for v in values):
            return list(values)
        if want is float and all(isinstance(v, float) for v in values):
            return list(values)
    return None


def _read_gbt_tree_weights(path: str) -> List[float]:
    """Read per-tree weights from treesMetadata/ parquet (GBT regressor / classifier).

    These are the step sizes Spark scales each tree by — (1.0, lr, lr, ...) —
    so dropping them does not degrade the model, it changes it: every tree after
    the first counts about ten times too much.

    Spark does not always name the columns of this frame. Spark 4.x writes it
    from an unnamed tuple, so the fields come back as _1/_2/_3 and a lookup by
    name finds nothing. Match on name first, then fall back to picking the
    columns out by type.
    """
    trees_meta_dir = os.path.join(path, "treesMetadata")
    if not os.path.isdir(trees_meta_dir):
        return []
    data = _read_parquet_dict(path, subdir="treesMetadata")

    ids = data.get("treeID")
    weights = data.get("weights")
    if ids is None or weights is None:
        ids = _column_of(data, int)
        weights = _column_of(data, float)
    if ids is None or weights is None:
        return []

    pairs = sorted(zip(ids, weights, strict=True))
    return [float(w) for _, w in pairs]


def _gbt_weights_or_fail(path: str, meta: dict, n_trees: int, what: str) -> List[float]:
    """Per-tree weights for a GBT, or a clear error.

    Defaulting a missing weight to 1.0 is never right for a GBT: only the first
    tree has weight 1.0 and the rest carry the learning rate. Silently doing it
    produced a model that loaded, scored, and was wrong by a factor of ten on
    every tree but the first.
    """
    weights = _read_gbt_tree_weights(path) or list(meta.get("treeWeights", []))
    if len(weights) < n_trees:
        raise ValueError(
            f"{what}: found {len(weights)} tree weights for {n_trees} trees. "
            "Spark stores them in treesMetadata/; without them every tree after "
            "the first would be scaled wrongly. This usually means an "
            "unrecognised layout — please report the Spark version."
        )
    return [float(w) for w in weights]


# ── Spark type decoders ────────────────────────────────────────────────────────

def _to_array(v) -> np.ndarray:
    """Spark VectorUDT struct → numpy 1-D float64 array."""
    if not isinstance(v, dict):
        return np.asarray(v, dtype=np.float64)
    if v.get("type", 1) == 1:          # dense
        return np.asarray(v["values"], dtype=np.float64)
    size    = int(v["size"])
    indices = v.get("indices") or []
    values  = v.get("values")  or []
    arr     = np.zeros(size, dtype=np.float64)
    for i, val in zip(indices, values, strict=True):
        arr[int(i)] = float(val)
    return arr


def _to_matrix(m) -> np.ndarray:
    """Spark MatrixUDT struct → numpy 2-D float64 array (numRows × numCols)."""
    if not isinstance(m, dict):
        return np.asarray(m, dtype=np.float64)
    n_rows        = int(m["numRows"])
    n_cols        = int(m["numCols"])
    is_transposed = bool(m.get("isTransposed", False))
    col_ptrs      = m.get("colPtrs")
    if col_ptrs is not None:
        raise NotImplementedError("SparseMatrix reading is not supported")
    values = np.asarray(m["values"], dtype=np.float64)
    if not is_transposed:
        return values.reshape(n_rows, n_cols, order="F")
    else:
        return values.reshape(n_cols, n_rows, order="F").T


# ── Tree construction from parquet rows ────────────────────────────────────────

def _regression_leaf(row: dict) -> float:
    return float(row["prediction"])


def _class_prob_leaf(k: int):
    def fn(row: dict) -> float:
        stats = row.get("impurityStats") or []
        total = sum(float(s) for s in stats)
        return float(stats[k]) / total if total > 0 else 0.0
    return fn


def _gbt_leaf(weight: float):
    def fn(row: dict) -> float:
        return float(row["prediction"]) * weight
    return fn


def _build_tree(node_rows: List[dict], leaf_fn, builder=None) -> Tree:
    rows     = sorted(node_rows, key=lambda r: int(r["id"]))
    n        = len(rows)

    node_kind:       List[TreeNodeKind] = []
    split_feature:   List[int]          = []
    split_threshold: List[float]        = []
    split_op:        List[TreeSplitOp]  = []
    children_index:  List[int]          = []
    children_offset: List[int]          = []
    children_count:  List[int]          = []
    default_child:   List[int]          = []
    leaf_value:      List[float]        = []
    category_set:    List[int]          = []
    cat_set_offset:  List[int]          = []
    cat_set_count:   List[int]          = []
    has_cat  = False
    child_ptr = 0

    for row in rows:
        lc = int(row["leftChild"])
        rc = int(row["rightChild"])
        children_offset.append(child_ptr)
        cat_set_offset.append(len(category_set))

        if lc == -1:                                    # leaf
            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            children_count.append(0)
            default_child.append(0)
            leaf_value.append(leaf_fn(row))
            cat_set_count.append(0)
        else:                                           # branch
            node_kind.append(TreeNodeKind.BRANCH)
            children_count.append(2)
            default_child.append(0)
            leaf_value.append(float(row["prediction"]))

            split = row["split"]
            split_feature.append(int(split["featureIndex"]))

            if "leftCategoriesOrThreshold" in split:    # PySpark 4.x
                num_cat = int(split["numCategories"])
                if num_cat < 0:
                    split_threshold.append(float(split["leftCategoriesOrThreshold"][0]))
                    split_op.append(TreeSplitOp.LESS_OR_EQUAL)
                    cat_set_count.append(0)
                else:
                    has_cat = True
                    split_threshold.append(0.0)
                    split_op.append(TreeSplitOp.IN_SET)
                    cats = [int(c) for c in split["leftCategoriesOrThreshold"]]
                    cat_set_count.append(len(cats))
                    category_set.extend(cats)
            elif int(split.get("featureType", 0)) == 0: # PySpark 3.x continuous
                split_threshold.append(float(split["threshold"]))
                split_op.append(TreeSplitOp.LESS_OR_EQUAL)
                cat_set_count.append(0)
            else:                                       # PySpark 3.x categorical
                has_cat = True
                split_threshold.append(0.0)
                split_op.append(TreeSplitOp.IN_SET)
                cats = [int(c) for c in (split.get("categories") or [])]
                cat_set_count.append(len(cats))
                category_set.extend(cats)

            children_index.extend([lc, rc])
            child_ptr += 2

    if builder is not None:
        st_tv = builder.body_tensor_value("split_threshold", np.array(split_threshold))
        lv_tv = builder.body_tensor_value("leaf_value", np.array(leaf_value))
    else:
        from omle.ir.tensor import Tensor
        from omle.ir.types import TensorValue
        st_tv = TensorValue.of_tensor(Tensor(float64_data=split_threshold))
        lv_tv = TensorValue.of_tensor(Tensor(float64_data=leaf_value))

    return Tree(
        num_nodes       = n,
        node_kind       = node_kind,
        split_feature   = split_feature,
        split_threshold = st_tv,
        split_op        = split_op,
        children_index  = children_index,
        children_offset = children_offset,
        children_count  = children_count,
        default_child   = default_child,
        leaf_value      = lv_tv,
        category_set         = category_set   if has_cat else [],
        category_set_offset  = cat_set_offset if has_cat else [],
        category_set_count   = cat_set_count  if has_cat else [],
    )


def _dt_node_rows(data: dict) -> List[dict]:
    if "nodeData" in data:
        return data["nodeData"]
    keys = list(data.keys())
    return [dict(zip(keys, vals, strict=True)) for vals in zip(*[data[k] for k in keys], strict=True)]


def _group_by_tree(data: dict) -> dict:
    tree_dict: dict = {}
    for tid, nd in zip(data["treeID"], data["nodeData"], strict=True):
        tree_dict.setdefault(int(tid), []).append(nd)
    return tree_dict


def _tree_ensemble_node(
    input_name: str,
    task: TaskType,
    ensemble: TreeEnsemble,
    builder: Builder,
    node_name: str = "tree_ensemble",
) -> List[str]:
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(task)
    builder.add_node(omle.Node(
        name        = node_name,
        domain      = "omle.ml",
        op          = "TreeEnsemble",
        inputs      = [omle.NodeInput(name=input_name)],
        outputs     = outputs,
        tree_ensemble = ensemble,
    ))
    return [o.name for o in outputs]


# ── Transformer loaders ────────────────────────────────────────────────────────

def _load_standard_scaler(path, input_name, prefix, builder, meta):
    pm   = meta.get("paramMap", {})
    def _f64_out_type(n: int) -> omle.TensorType:
        return omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n])

    data = _read_parquet_dict(path)
    attrs = []
    n_feat = 0
    if pm.get("withMean", False) and data.get("mean") and data["mean"][0] is not None:
        mean_arr = _to_array(data["mean"][0])
        attrs.append(builder.tensor_attr("mean", f"{prefix}_mean", mean_arr))
        n_feat = len(mean_arr)
    if pm.get("withStd", True) and data.get("std") and data["std"][0] is not None:
        std_arr = _to_array(data["std"][0])
        attrs.append(builder.tensor_attr("scale", f"{prefix}_scale", std_arr))
        if not n_feat:
            n_feat = len(std_arr)
    if n_feat:
        builder.n_features = n_feat
    out = meta.get("paramMap", {}).get("outputCol") or builder.unique_name(f"{prefix}_out")
    out_type = _f64_out_type(n_feat) if n_feat else None
    return feature_node("StandardScaler", input_name, out, attrs, builder, output_type=out_type,
                        node_name=builder.unique_name("standard_scaler_model"))


def _load_min_max_scaler(path, input_name, prefix, builder, meta):
    pm       = meta.get("paramMap", {})
    data     = _read_parquet_dict(path)
    orig_min = _to_array(data["originalMin"][0])
    orig_max = _to_array(data["originalMax"][0])
    n_feat   = len(orig_min)
    if n_feat:
        builder.n_features = n_feat
    out      = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    out_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_feat]) if n_feat else None
    # Runtime interprets data_max as per-feature scale (max - min), not absolute maximum.
    return feature_node("MinMaxScaler", input_name, out, [
        builder.tensor_attr("data_min", f"{prefix}_min", orig_min),
        builder.tensor_attr("data_max", f"{prefix}_max", orig_max - orig_min),
        builder.attr_f("feature_range_min", float(pm.get("min", 0.0))),
        builder.attr_f("feature_range_max", float(pm.get("max", 1.0))),
    ], builder, output_type=out_type, node_name=builder.unique_name("min_max_scaler_model"))


def _load_max_abs_scaler(path, input_name, prefix, builder, meta):
    data    = _read_parquet_dict(path)
    scale   = _to_array(data["maxAbs"][0])
    n_feat  = len(scale)
    if n_feat:
        builder.n_features = n_feat
    out     = meta.get("paramMap", {}).get("outputCol") or builder.unique_name(f"{prefix}_out")
    out_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_feat]) if n_feat else None
    return feature_node("MaxAbsScaler", input_name, out, [
        builder.tensor_attr("scale", f"{prefix}_scale", scale),
    ], builder, output_type=out_type, node_name=builder.unique_name("max_abs_scaler_model"))


def _load_normalizer(path, input_name, prefix, builder, meta):
    pm  = meta.get("paramMap", {})
    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return feature_node("Normalizer", input_name, out, [
        builder.attr_f("p", float(pm.get("p", 2.0))),
    ], builder, node_name=builder.unique_name("normalizer"))


def _load_binarizer(path, input_name, prefix, builder, meta):
    pm          = meta.get("paramMap", {})
    input_cols  = pm.get("inputCols", [])
    output_cols = pm.get("outputCols", [])
    # Multi-col Spark uses "thresholds" (list); single-col uses "threshold" (scalar).
    raw = pm.get("thresholds") or [pm.get("threshold", 0.5)]
    thresholds  = np.array([float(v) for v in raw], dtype=np.float64)

    if input_cols and output_cols:
        col_inputs = [input_name] + list(input_cols[1:])
        out_names  = [builder.unique_name(oc) for oc in output_cols]
        builder.require_namespace("omle.feature")
        builder.add_node(omle.Node(
            name=builder.unique_name("binarizer"),
            domain="omle.feature", op="Binarizer",
            inputs=[omle.NodeInput(name=c) for c in col_inputs],
            outputs=[omle.NodeOutput(name=n, role=omle.OutputRole.TRANSFORMED_VALUE)
                     for n in out_names],
            attributes=[builder.tensor_attr("thresholds", f"{prefix}_thresholds", thresholds)],
        ))
        return out_names[-1]

    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return feature_node("Binarizer", input_name, out, [
        builder.tensor_attr("thresholds", f"{prefix}_thresholds", thresholds),
    ], builder, node_name=builder.unique_name("binarizer"))


def _load_string_indexer(path, input_name, prefix, builder, meta):
    data = _read_parquet_dict(path)
    if "labelsArray" in data and data["labelsArray"]:
        labels_array = list(data["labelsArray"][0])
    elif "labels" in data and data["labels"]:
        labels_array = [list(data["labels"][0])]
    else:
        raise ValueError(f"StringIndexerModel at {path}: cannot find labels data")

    explicit_pm = meta.get("paramMap", {})

    # Concatenate all per-column label lists and build offsets for the variadic spec.
    all_labels: List[str] = []
    offsets: List[int] = [0]
    for labels in labels_array:
        all_labels.extend(list(labels))
        offsets.append(len(all_labels))
    offsets_arr = np.array(offsets, dtype=np.int64)

    if len(labels_array) == 1:
        out = (explicit_pm.get("outputCol") or
               (explicit_pm.get("outputCols") or [None])[0] or
               builder.unique_name(f"{prefix}_out"))
        _le_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1])
        return feature_node("LabelEncoder", input_name, out, [
            builder.string_tensor_attr("labels", f"{prefix}_labels", all_labels),
            builder.int_tensor_attr("label_offsets", f"{prefix}_offsets", offsets_arr),
        ], builder, node_name=builder.unique_name("string_indexer_model"),
           output_type=_le_type)
    else:
        # Multi-col: single LabelEncoder with variadic xs/ys — one node, no Concat.
        input_cols  = explicit_pm.get("inputCols", [])
        output_cols = explicit_pm.get("outputCols", [])
        col_inputs  = list(input_cols) if input_cols else [input_name]
        out_names   = ([builder.unique_name(oc) for oc in output_cols]
                       if output_cols else
                       [builder.unique_name(f"{prefix}_col{i}_out") for i in range(len(labels_array))])
        builder.require_namespace("omle.feature")
        builder.add_node(omle.Node(
            name=builder.unique_name("string_indexer_model"),
            domain="omle.feature", op="LabelEncoder",
            inputs=[omle.NodeInput(name=n) for n in col_inputs],
            outputs=[omle.NodeOutput(
                         name=n, role=omle.OutputRole.TRANSFORMED_VALUE,
                         type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]),
                     ) for n in out_names],
            attributes=[
                builder.string_tensor_attr("labels", f"{prefix}_labels", all_labels),
                builder.int_tensor_attr("label_offsets", f"{prefix}_offsets", offsets_arr),
            ],
        ))
        return out_names[-1]


def _load_one_hot_encoder(path, input_name, prefix, builder, meta):
    pm   = meta.get("paramMap", {})
    data = _read_parquet_dict(path)
    sizes = list(data["categorySizes"][0])
    drop  = bool(pm.get("dropLast", True))

    # Build concatenated categories and offsets for all columns.
    all_cats: list[str] = []
    offsets = [0]
    for sz in sizes:
        n_out = sz - (1 if drop else 0)
        all_cats.extend(str(j) for j in range(n_out))
        offsets.append(len(all_cats))
    offsets_arr = np.array(offsets, dtype=np.int64)
    n_total = offsets[-1]

    if len(sizes) == 1:
        out = (pm.get("outputCol") or
               (pm.get("outputCols") or [None])[0] or
               builder.unique_name(f"{prefix}_out"))
        out_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_total])
        return feature_node("OneHotEncoder", input_name, out, [
            builder.string_tensor_attr("categories", f"{prefix}_categories", all_cats),
            builder.int_tensor_attr("category_offsets", f"{prefix}_offsets", offsets_arr),
        ], builder, output_type=out_type, node_name=builder.unique_name("one_hot_encoder_model"))
    else:
        # Single variadic OHE node: xs=[col0, col1, …], ys=[out0, out1, …].
        input_cols  = pm.get("inputCols", [])
        output_cols = pm.get("outputCols", [])
        col_inputs  = list(input_cols) if input_cols else [input_name]
        out_names   = []
        for i, _sz in enumerate(sizes):
            n_out = offsets[i + 1] - offsets[i]
            name  = (builder.unique_name(output_cols[i]) if i < len(output_cols)
                     else builder.unique_name(f"{prefix}_col{i}_ohe"))
            out_names.append((name, n_out))
        builder.require_namespace("omle.feature")
        builder.add_node(omle.Node(
            name=builder.unique_name("one_hot_encoder_model"),
            domain="omle.feature", op="OneHotEncoder",
            inputs=[omle.NodeInput(name=c) for c in col_inputs],
            outputs=[omle.NodeOutput(
                name=name,
                role=omle.OutputRole.TRANSFORMED_VALUE,
                type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_out]),
            ) for name, n_out in out_names],
            attributes=[
                builder.string_tensor_attr("categories", f"{prefix}_categories", all_cats),
                builder.int_tensor_attr("category_offsets", f"{prefix}_offsets", offsets_arr),
            ],
        ))
        return out_names[-1][0]


def _load_bucketizer(path, input_name, prefix, builder, meta):
    pm          = meta.get("paramMap", {})
    input_cols  = pm.get("inputCols", [])
    output_cols = pm.get("outputCols", [])
    splits_arr  = pm.get("splitsArray", [])

    if input_cols and output_cols and splits_arr:
        all_bounds: list[float] = []
        offsets_list = [0]
        for splits in splits_arr:
            bounds = [float(v) for v in splits if not math.isinf(float(v))]
            all_bounds.extend(bounds)
            offsets_list.append(len(all_bounds))
        col_inputs = [input_name] + list(input_cols[1:])
        out_names  = [builder.unique_name(oc) for oc in output_cols]
        builder.require_namespace("omle.feature")
        builder.add_node(omle.Node(
            name=builder.unique_name("bucketizer"),
            domain="omle.feature", op="Bucketizer",
            inputs=[omle.NodeInput(name=c) for c in col_inputs],
            outputs=[omle.NodeOutput(name=n, role=omle.OutputRole.TRANSFORMED_VALUE)
                     for n in out_names],
            attributes=[
                builder.tensor_attr("boundaries", f"{prefix}_boundaries",
                                    np.array(all_bounds, dtype=np.float64)),
                builder.int_tensor_attr("boundary_offsets", f"{prefix}_boundary_offsets",
                                        np.array(offsets_list, dtype=np.int64)),
            ],
        ))
        return out_names[-1]

    splits  = [float(v) for v in pm.get("splits", []) if not math.isinf(float(v))]
    offsets = np.array([0, len(splits)], dtype=np.int64)
    out     = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return feature_node("Bucketizer", input_name, out, [
        builder.tensor_attr("boundaries",           f"{prefix}_boundaries",
                            np.array(splits, dtype=np.float64)),
        builder.int_tensor_attr("boundary_offsets", f"{prefix}_boundary_offsets", offsets),
    ], builder, node_name=builder.unique_name("bucketizer"))


def _load_imputer(path, input_name, prefix, builder, meta):
    pm          = meta.get("paramMap", {})
    data        = _read_parquet_dict(path)
    input_cols  = pm.get("inputCols", [])
    output_cols = pm.get("outputCols", [])

    if input_cols and output_cols:
        # Multi-column mode: single Imputer node with variadic xs/ys.
        # input_name is the resolved tensor name for inputCols[0]; remaining columns
        # use their Spark column name directly (standalone, where inputCols == graph inputs).
        # Pipeline callers use the _load_pipeline special case to pre-resolve all names.
        col_inputs = [input_name] + list(input_cols[1:])
        fills      = np.array([float(data.get(c, [0.0])[0]) for c in input_cols], dtype=np.float64)
        out_names  = [builder.unique_name(oc) for oc in output_cols]
        builder.require_namespace("omle.feature")
        builder.add_node(omle.Node(
            name=builder.unique_name("imputer_model"), domain="omle.feature", op="Imputer",
            inputs=[omle.NodeInput(name=n) for n in col_inputs],
            outputs=[omle.NodeOutput(name=n, role=omle.OutputRole.TRANSFORMED_VALUE)
                     for n in out_names],
            attributes=[builder.tensor_attr("fill_tensor", f"{prefix}_fill", fills)],
        ))
        return out_names[-1]

    # Single-column (inputCol) or vector fallback.
    surrogate_cols = [k for k in data if not k.startswith("__")]
    surrogates = np.array([float(data[k][0]) for k in surrogate_cols], dtype=np.float64)
    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return feature_node("Imputer", input_name, out, [
        builder.tensor_attr("fill_tensor", f"{prefix}_fill", surrogates),
    ], builder, node_name=builder.unique_name("imputer_model"))


def _load_pca(path, input_name, prefix, builder, meta):
    pm   = meta.get("paramMap", {})
    data = _read_parquet_dict(path)
    k    = int(pm.get("k", 1))
    pc   = _to_matrix(data["pc"][0])
    comp = pc.T
    builder.n_features = pc.shape[0]
    attrs = [
        builder.attr_i("n_components", k),
        builder.tensor_attr("components", f"{prefix}_components", comp),
    ]
    if "mean" in data and data["mean"] and data["mean"][0] is not None:
        mean_arr = _to_array(data["mean"][0])
        if len(mean_arr) > 0:
            attrs.append(builder.tensor_attr("mean", f"{prefix}_mean", mean_arr))
    out      = meta.get("paramMap", {}).get("outputCol") or builder.unique_name(f"{prefix}_out")
    fn       = [f"pca{i}" for i in range(k)]
    out_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, k])
    return feature_node("PCA", input_name, out, attrs, builder, field_names=fn, output_type=out_type,
                        node_name=builder.unique_name("pca_model"))


def _load_vector_assembler(input_names, prefix, builder):
    out = builder.unique_name(f"{prefix}_out")
    # field_names: the individual column names assembled into the vector
    fn = list(input_names) if isinstance(input_names, (list, tuple)) else []
    return core_node("Concat", input_names, out, [], builder, field_names=fn)


# ── Estimator loaders ──────────────────────────────────────────────────────────

def _load_linear_regression(path, input_name, builder, meta):
    data    = _read_parquet_dict(path)
    coef    = _to_array(data["coefficients"][0])
    if builder.n_features < 0:
        builder.n_features = len(coef)
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(TaskType.REGRESSION)
    builder.add_node(omle.Node(
        name="linear_regression_model", domain="omle.ml", op="Linear",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        linear=Linear(
            task_type=TaskType.REGRESSION,
            coefficients=builder.body_tensor_value("linear_weights", coef),
            intercept=builder.body_tensor_value("linear_intercept",
                                                np.array([float(data["intercept"][0])])),
        ),
    ))
    return [o.name for o in outputs]


def _load_logistic_regression(path, input_name, builder, meta):
    data        = _read_parquet_dict(path)
    # numClasses is stored in the parquet data (not in the metadata JSON) for Spark LR.
    num_classes = int(data["numClasses"][0]) if "numClasses" in data else int(meta.get("numClasses", 2))
    task        = TaskType.BINARY if num_classes == 2 else TaskType.MULTICLASS
    coef_col = "coefficientMatrix" if "coefficientMatrix" in data else "coefficients"
    bias_col = "interceptVector"   if "interceptVector"   in data else "intercept"
    coef_mat = _to_matrix(data[coef_col][0])
    bias_vec = _to_array(data[bias_col][0])
    if builder.n_features < 0:
        builder.n_features = coef_mat.shape[1]
    if num_classes == 2:
        coef_data = coef_mat.ravel()
        post      = PostTransform.SIGMOID
    else:
        coef_data = coef_mat
        post      = PostTransform.SOFTMAX
    builder.require_namespace("omle.ml")
    builder.n_classes = num_classes
    outputs = make_node_outputs(task)
    builder.add_node(omle.Node(
        name="logistic_regression_model", domain="omle.ml", op="Linear",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        linear=Linear(
            task_type=task,
            coefficients=builder.body_tensor_value("linear_weights", coef_data),
            intercept=builder.body_tensor_value("linear_intercept", bias_vec),
            post_transform=post,
        ),
    ))
    return [o.name for o in outputs]


def _load_dt_classifier(path, input_name, builder, meta):
    num_classes = int(meta.get("numClasses", 2))
    task        = TaskType.BINARY if num_classes == 2 else TaskType.MULTICLASS
    builder.n_classes = num_classes
    if meta.get("numFeatures") and builder.n_features < 0:
        builder.n_features = int(meta["numFeatures"])
    data        = _read_parquet_dict(path)
    rows        = _dt_node_rows(data)
    if num_classes == 2:
        tree     = _build_tree(rows, _class_prob_leaf(1), builder)
        ensemble = TreeEnsemble(task_type=task, trees=[tree], aggregation=TreeAggregation.AVERAGE)
    else:
        trees, tree_group = [], []
        for k in range(num_classes):
            trees.append(_build_tree(rows, _class_prob_leaf(k), builder))
            tree_group.append(k)
        # Per-class leaf fractions already sum to 1; SOFT_VOTE without SOFTMAX is correct.
        ensemble = TreeEnsemble(
            task_type=task, trees=trees, aggregation=TreeAggregation.SOFT_VOTE, tree_group=tree_group,
        )
    return _tree_ensemble_node(input_name, task, ensemble, builder, "decision_tree_classification_model")


def _load_dt_regressor(path, input_name, builder, meta):
    if meta.get("numFeatures") and builder.n_features < 0:
        builder.n_features = int(meta["numFeatures"])
    data     = _read_parquet_dict(path)
    tree     = _build_tree(_dt_node_rows(data), _regression_leaf, builder)
    ensemble = TreeEnsemble(task_type=TaskType.REGRESSION, trees=[tree], aggregation=TreeAggregation.AVERAGE)
    return _tree_ensemble_node(input_name, TaskType.REGRESSION, ensemble, builder, "decision_tree_regression_model")


def _load_rf_classifier(path, input_name, builder, meta):
    num_classes = int(meta.get("numClasses", 2))
    task        = TaskType.BINARY if num_classes == 2 else TaskType.MULTICLASS
    builder.n_classes = num_classes
    if meta.get("numFeatures") and builder.n_features < 0:
        builder.n_features = int(meta["numFeatures"])
    data        = _read_parquet_dict(path)
    tree_dict   = _group_by_tree(data)
    if num_classes == 2:
        trees    = [_build_tree(tree_dict[i], _class_prob_leaf(1), builder) for i in sorted(tree_dict)]
        ensemble = TreeEnsemble(task_type=task, trees=trees, aggregation=TreeAggregation.AVERAGE)
    else:
        trees, tree_group = [], []
        for i in sorted(tree_dict):
            for k in range(num_classes):
                trees.append(_build_tree(tree_dict[i], _class_prob_leaf(k), builder))
                tree_group.append(k)
        # Per-class leaf fractions already sum to 1; SOFT_VOTE without SOFTMAX is correct.
        ensemble = TreeEnsemble(
            task_type=task, trees=trees, aggregation=TreeAggregation.SOFT_VOTE, tree_group=tree_group,
        )
    return _tree_ensemble_node(input_name, task, ensemble, builder, "random_forest_classification_model")


def _load_rf_regressor(path, input_name, builder, meta):
    if meta.get("numFeatures") and builder.n_features < 0:
        builder.n_features = int(meta["numFeatures"])
    data      = _read_parquet_dict(path)
    tree_dict = _group_by_tree(data)
    trees     = [_build_tree(tree_dict[i], _regression_leaf, builder) for i in sorted(tree_dict)]
    ensemble  = TreeEnsemble(task_type=TaskType.REGRESSION, trees=trees, aggregation=TreeAggregation.AVERAGE)
    return _tree_ensemble_node(input_name, TaskType.REGRESSION, ensemble, builder, "random_forest_regression_model")


def _load_gbt_classifier(path, input_name, builder, meta):
    builder.n_classes = 2
    if meta.get("numFeatures") and builder.n_features < 0:
        builder.n_features = int(meta["numFeatures"])
    data      = _read_parquet_dict(path)
    tree_dict = _group_by_tree(data)
    weights   = _gbt_weights_or_fail(path, meta, len(tree_dict), "GBTClassifier")
    # Spark binary GBT converts raw score F → P via sigmoid(2*F); multiply weights by 2
    # so the runtime's sigmoid(sum) matches Spark's sigmoid(2*sum).
    trees = [
        _build_tree(tree_dict[i], _gbt_leaf(2.0 * weights[i]), builder)
        for i in sorted(tree_dict)
    ]
    ensemble = TreeEnsemble(
        task_type=TaskType.BINARY, trees=trees, aggregation=TreeAggregation.SUM,
        post_transform=PostTransform.SIGMOID,
    )
    return _tree_ensemble_node(input_name, TaskType.BINARY, ensemble, builder, "gbt_classification_model")


def _load_gbt_regressor(path, input_name, builder, meta):
    if meta.get("numFeatures") and builder.n_features < 0:
        builder.n_features = int(meta["numFeatures"])
    data      = _read_parquet_dict(path)
    tree_dict = _group_by_tree(data)
    weights   = _gbt_weights_or_fail(path, meta, len(tree_dict), "GBTRegressor")
    trees = [
        _build_tree(tree_dict[i], _gbt_leaf(weights[i]), builder)
        for i in sorted(tree_dict)
    ]
    ensemble = TreeEnsemble(task_type=TaskType.REGRESSION, trees=trees, aggregation=TreeAggregation.SUM)
    return _tree_ensemble_node(input_name, TaskType.REGRESSION, ensemble, builder, "gbt_regression_model")


def _load_naive_bayes(path, input_name, builder, meta):
    pm         = meta.get("paramMap", {})
    model_type = pm.get("modelType", "multinomial")
    data       = _read_parquet_dict(path)
    pi_arr    = _to_array(data["pi"][0])
    theta_mat = _to_matrix(data["theta"][0])
    num_classes, num_features = theta_mat.shape
    builder.n_classes = num_classes
    if builder.n_features < 0:
        builder.n_features = num_features
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(TaskType.MULTICLASS)
    if model_type == "gaussian":
        sigma_mat = _to_matrix(data["sigma"][0])
        nb_body   = NaiveBayesBody(
            task_type=TaskType.MULTICLASS,
            class_log_priors=builder.body_tensor_value("nb_class_log_priors", pi_arr),
            gaussian=GaussianNaiveBayes(
                means=builder.body_tensor_value("nb_means", theta_mat),
                variances=builder.body_tensor_value("nb_variances", sigma_mat),
            ),
        )
    elif model_type == "bernoulli":
        nb_body = NaiveBayesBody(
            task_type=TaskType.MULTICLASS,
            class_log_priors=builder.body_tensor_value("nb_class_log_priors", pi_arr),
            bernoulli=BernoulliNaiveBayes(
                feature_log_prob=builder.body_tensor_value("nb_feature_log_prob", theta_mat),
            ),
        )
    else:
        nb_body = NaiveBayesBody(
            task_type=TaskType.MULTICLASS,
            class_log_priors=builder.body_tensor_value("nb_class_log_priors", pi_arr),
            multinomial=MultinomialNaiveBayes(
                feature_log_prob=builder.body_tensor_value("nb_feature_log_prob", theta_mat),
            ),
        )
    builder.add_node(omle.Node(
        name="naive_bayes_model", domain="omle.ml", op="NaiveBayes",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        naive_bayes=nb_body,
    ))
    return [o.name for o in outputs]


def _load_kmeans(path, input_name, builder, meta):
    data       = _read_parquet_dict(path)
    idx_col    = data["clusterIdx"]
    center_col = data["clusterCenter"]
    centers    = [_to_array(c) for _, c in sorted(zip(idx_col, center_col, strict=True))]
    mat        = np.vstack(centers)
    if builder.n_features < 0:
        builder.n_features = mat.shape[1]
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(TaskType.CLUSTERING)
    builder.add_node(omle.Node(
        name="k_means_model", domain="omle.ml", op="Clustering",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        clustering=Clustering(
            task_type=TaskType.CLUSTERING,
            prototype=PrototypeClustering(
                centers=builder.body_tensor_value("kmeans_centers", mat),
                distance_measure=DistanceMeasure.EUCLIDEAN,
            )
        ),
    ))
    return [o.name for o in outputs]


def _load_linear_svc(path, input_name, builder, meta):
    builder.n_classes = 2
    data    = _read_parquet_dict(path)
    if builder.n_features < 0:
        builder.n_features = len(_to_array(data["coefficients"][0]))
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(TaskType.BINARY)
    builder.add_node(omle.Node(
        name="linear_svc_model", domain="omle.ml", op="SVM",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        svm=SVM(
            task_type=TaskType.BINARY,
            linear=LinearSVM(
                coefficients=builder.body_tensor_value("svm_coef", _to_array(data["coefficients"][0])),
                intercept=builder.body_tensor_value("svm_intercept",
                                                    np.array([float(data["intercept"][0])])),
            ),
        ),
    ))
    return [o.name for o in outputs]


def _load_mlp_classifier(path, input_name, builder, meta):
    pm          = meta.get("paramMap", {})
    layer_sizes = list(pm["layers"])
    data        = _read_parquet_dict(path)
    weights_flat = _to_array(data["weights"][0])
    n_layers  = len(layer_sizes)
    task      = TaskType.BINARY if layer_sizes[-1] == 2 else TaskType.MULTICLASS
    builder.n_classes = layer_sizes[-1]
    if builder.n_features < 0:
        builder.n_features = layer_sizes[0]
    dense_layers: List[DenseLayer] = []
    offset = 0
    for layer in range(n_layers - 1):
        in_size, out_size = layer_sizes[layer], layer_sizes[layer + 1]
        w_slice = weights_flat[offset:offset + out_size * in_size]
        b_slice = weights_flat[offset + out_size * in_size:offset + out_size * in_size + out_size]
        offset += out_size * in_size + out_size
        w    = w_slice.reshape(out_size, in_size, order="F")
        is_output  = layer == n_layers - 2
        activation = NeuralNetworkActivation.SOFTMAX if is_output else NeuralNetworkActivation.LOGISTIC
        dense_layers.append(DenseLayer(
            weights=builder.body_tensor_value(f"mlp_w{layer}", w),
            bias=builder.body_tensor_value(f"mlp_b{layer}", b_slice),
            activation=activation,
            name=f"layer_{layer}",
        ))
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(task)
    builder.add_node(omle.Node(
        name="multilayer_perceptron_classification_model", domain="omle.ml", op="NeuralNetwork",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        neural_network=NeuralNetwork(task_type=TaskType.MULTICLASS, layers=dense_layers),
    ))
    return [o.name for o in outputs]


def _gmm_precision_cholesky(sigma: np.ndarray) -> np.ndarray:
    """Convert full covariance matrix to upper-triangular precision Cholesky (sklearn format)."""
    d = sigma.shape[0]
    cov_chol = np.linalg.cholesky(sigma + 1e-6 * np.eye(d))
    return np.linalg.solve(cov_chol, np.eye(d)).T  # upper triangular, matches runtime expectation


def _load_gaussian_mixture(path, input_name, builder, meta):
    data = _read_parquet_dict(path)
    # Spark 4.0+: single-row list columns "weights"/"mus"/"sigmas"
    # Spark 3.x:  per-component rows "weight"/"mean"/"cov"
    if "weights" in data:
        weights = np.asarray(data["weights"][0], dtype=np.float64)
        means   = np.array([_to_array(m) for m in data["mus"][0]], dtype=np.float64)
        covs    = np.array([_to_matrix(c) for c in data["sigmas"][0]], dtype=np.float64)
    else:
        weights = np.asarray(data["weight"], dtype=np.float64)
        means   = np.array([_to_array(m) for m in data["mean"]], dtype=np.float64)
        covs    = np.array([_to_matrix(c) for c in data["cov"]], dtype=np.float64)
    k, d    = means.shape
    if builder.n_features < 0:
        builder.n_features = d
    # Convert covariance matrices to precision Cholesky (format expected by the runtime).
    prec_chols = np.array([_gmm_precision_cholesky(covs[i]) for i in range(k)])
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(TaskType.CLUSTERING)
    builder.add_node(omle.Node(
        name="gaussian_mixture_model", domain="omle.ml", op="Clustering",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        clustering=Clustering(
            task_type=TaskType.CLUSTERING,
            gaussian_mixture=GaussianMixtureClustering(
                weights=builder.body_tensor_value("gmm_weights", weights),
                means=builder.body_tensor_value("gmm_means", means),
                covariances=builder.body_tensor_value("gmm_covs", prec_chols.reshape(k * d, d)),
                covariance_type=CovarianceType.FULL,
            )
        ),
    ))
    return [o.name for o in outputs]


_GLM_LINK_TO_POST: dict = {
    "identity": PostTransform.POST_TRANSFORM_UNSPECIFIED,
    "log":      PostTransform.EXP,
    "logit":    PostTransform.SIGMOID,
    "cloglog":  PostTransform.CLOGLOG,
    "loglog":   PostTransform.LOGLOG,
    "cauchit":  PostTransform.CAUCHIT,
}


def _load_generalized_linear_regression(path, input_name, builder, meta):
    pm      = meta.get("paramMap", {})
    data    = _read_parquet_dict(path)
    coef    = _to_array(data["coefficients"][0])
    if builder.n_features < 0:
        builder.n_features = len(coef)
    link    = str(pm.get("link", "identity")).lower()
    post    = _GLM_LINK_TO_POST.get(link, PostTransform.POST_TRANSFORM_UNSPECIFIED)
    builder.require_namespace("omle.ml")
    outputs = make_node_outputs(TaskType.REGRESSION)
    builder.add_node(omle.Node(
        name="generalized_linear_regression_model", domain="omle.ml", op="Linear",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=outputs,
        linear=Linear(
            task_type=TaskType.REGRESSION,
            coefficients=builder.body_tensor_value("glm_coef", coef),
            intercept=builder.body_tensor_value("glm_intercept",
                                                np.array([float(data["intercept"][0])])),
            post_transform=post,
        ),
    ))
    return [o.name for o in outputs]


def _merge_external_model(
    external_model: omle.OMLEModel,
    input_name: str,
    builder: Builder,
) -> List[str]:
    """Merge nodes from an externally-converted model (XGBoost, LightGBM) into the builder.

    The external model's first node is assumed to take a single input named "X",
    which is rewired to ``input_name`` so it connects to the preceding pipeline stage.
    All tensor entries and namespace imports are merged in as-is.
    """
    for ns_import in external_model.operator_imports:
        builder.require_namespace(ns_import.namespace)
    for entry in external_model.tensor_entries:
        builder.tensor_entries.append(entry)
    for i, node in enumerate(external_model.nodes):
        if i == 0:
            for inp in node.inputs:
                if inp.name == "X":
                    inp.name = omle.NameRef(value=input_name)
        builder.add_node(node)
    # Propagate n_classes so _make_outputs can set correct probability shape.
    for output in external_model.outputs:
        if output.role == omle.OutputRole.PROBABILITY:
            if output.type and output.type.shape and len(output.type.shape) >= 2:
                builder.n_classes = output.type.shape[-1]
            break
    return [o.name for o in external_model.nodes[-1].outputs] if external_model.nodes else []


def _load_xgboost_spark(path, input_name, builder, meta):
    """Load a saved XGBoost Spark model (Python xgboost.spark or Scala XGBoost4J)."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    booster = xgb.Booster()
    model_dir = os.path.join(path, "model")
    data_dir  = os.path.join(path, "data")

    if os.path.isdir(model_dir) and _glob.glob(os.path.join(model_dir, "part-*")):
        # Python xgboost.spark format: chunked JSON text in model/part-*
        part_files = sorted(_glob.glob(os.path.join(model_dir, "part-*")))
        lines: List[str] = []
        for pf in part_files:
            with open(pf, "r", encoding="utf-8") as fh:
                lines.extend(line.rstrip("\n") for line in fh)
        booster.load_model(bytearray("".join(lines).encode("utf-8")))
    elif os.path.isdir(data_dir) and _glob.glob(os.path.join(data_dir, "part-*")):
        # Scala XGBoost4J format: native binary in data/part-*
        import shutil
        import tempfile
        part_files = sorted(_glob.glob(os.path.join(data_dir, "part-*")))
        content = b"".join(open(pf, "rb").read() for pf in part_files)
        tmp = tempfile.mkdtemp(prefix="omle_xgb_")
        try:
            tmp_path = os.path.join(tmp, "model")
            with open(tmp_path, "wb") as fh:
                fh.write(content)
            booster.load_model(tmp_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        raise FileNotFoundError(
            f"Cannot find XGBoost model data in {model_dir} or {data_dir}"
        )

    from omle_convert.xgboost import from_xgboost
    external = from_xgboost(booster, n_verify=0, n_warmup=0, n_sample=0)
    return _merge_external_model(external, input_name, builder)


def _load_lightgbm_spark(path, input_name, builder, meta):
    """Load a saved SynapseML LightGBM Spark model (modelStr stored in metadata paramMap)."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is required: pip install lightgbm") from e

    model_str = meta.get("paramMap", {}).get("modelStr")
    if not model_str:
        raise ValueError(
            f"LightGBM SynapseML model at {path}: 'modelStr' not found in metadata paramMap. "
            "Ensure the model was saved by SynapseML LightGBMClassificationModel or "
            "LightGBMRegressionModel."
        )

    booster = lgb.Booster(model_str=model_str)
    from omle_convert.lightgbm import from_lightgbm
    external = from_lightgbm(booster, n_verify=0, n_warmup=0, n_sample=0)
    return _merge_external_model(external, input_name, builder)


def _load_isotonic_regression(path, input_name, builder, meta):
    data        = _read_parquet_dict(path)
    boundaries  = np.asarray(data["boundaries"][0], dtype=np.float64)
    predictions = np.asarray(data["predictions"][0], dtype=np.float64)
    n           = len(boundaries)
    offsets     = np.array([0, n], dtype=np.int64)
    builder.require_namespace("omle.feature")
    node_name = builder.unique_name("isotonic_regression_model")
    builder.add_node(omle.Node(
        name=node_name, domain="omle.feature", op="NormContinuous",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name="y_pred", role=omle.OutputRole.PREDICTION)],
        attributes=[
            builder.tensor_attr("orig_points", "ir_boundaries", boundaries),
            builder.tensor_attr("norm_points", "ir_predictions", predictions),
            builder.int_tensor_attr("point_offsets", "ir_offsets", offsets),
            omle.Attribute(name="outlier_treatment", s="as_extreme_values"),
        ],
    ))
    return ["y_pred"]


def _load_feature_selector(path, input_name, prefix, builder, meta):
    data     = _read_parquet_dict(path)
    raw      = data["selectedFeatures"]
    indices  = sorted(int(i) for i in (raw[0] if isinstance(raw[0], (list, tuple)) else raw))
    out      = meta.get("paramMap", {}).get("outputCol") or builder.unique_name(f"{prefix}_out")
    node_nm  = builder.unique_name(_cls_to_snake(_short_class(meta.get("class", ""))))
    out_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, len(indices)]) if indices else None
    builder.require_namespace("omle.core")
    builder.add_node(omle.Node(
        name=node_nm, domain="omle.core", op="TakeSlots",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name=out, role=omle.OutputRole.TRANSFORMED_VALUE, type=out_type)],
        attributes=[omle.Attribute(name="indices", ints=indices)],
    ))
    return out


def _load_vector_slicer(path, input_name, prefix, builder, meta):
    pm      = meta.get("paramMap", {})
    indices = sorted(int(i) for i in pm.get("indices", []))
    out     = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    node_nm = builder.unique_name("vector_slicer")
    out_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, len(indices)]) if indices else None
    builder.require_namespace("omle.core")
    builder.add_node(omle.Node(
        name=node_nm, domain="omle.core", op="TakeSlots",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name=out, role=omle.OutputRole.TRANSFORMED_VALUE, type=out_type)],
        attributes=[omle.Attribute(name="indices", ints=indices)],
    ))
    return out


# ── Text transformer loaders ───────────────────────────────────────────────────

def _text_node(op, input_name, output_name, attrs, builder, node_name=""):
    builder.require_namespace("omle.text")
    builder.add_node(omle.Node(
        name=node_name or output_name, domain="omle.text", op=op,
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name=output_name, role=omle.OutputRole.TRANSFORMED_VALUE)],
        attributes=attrs,
    ))
    return output_name


def _load_tokenizer(path, input_name, prefix, builder, meta):
    pm  = meta.get("paramMap", {})
    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("Tokenizer", input_name, out, [], builder,
                      node_name=builder.unique_name("tokenizer"))


def _load_regex_tokenizer(path, input_name, prefix, builder, meta):
    pm  = meta.get("paramMap", {})
    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("RegexTokenizer", input_name, out, [
        omle.Attribute(name="pattern",          s=str(pm.get("pattern", "\\s+"))),
        omle.Attribute(name="gaps",             b=bool(pm.get("gaps", True))),
        omle.Attribute(name="min_token_length", i=int(pm.get("minTokenLength", 1))),
    ], builder, node_name=builder.unique_name("regex_tokenizer"))


def _load_stop_words_remover(path, input_name, prefix, builder, meta):
    pm  = meta.get("paramMap", {})
    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("StopWordsRemover", input_name, out, [
        builder.string_tensor_attr("stop_words", f"{prefix}_stopwords", list(pm.get("stopWords", []))),
        omle.Attribute(name="case_sensitive", b=bool(pm.get("caseSensitive", False))),
    ], builder, node_name=builder.unique_name("stop_words_remover"))


def _load_ngram(path, input_name, prefix, builder, meta):
    pm  = meta.get("paramMap", {})
    n   = int(pm.get("n", 2))
    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("NGram", input_name, out, [
        omle.Attribute(name="n_min", i=n),
        omle.Attribute(name="n_max", i=n),
    ], builder, node_name=builder.unique_name("n_gram"))


def _load_count_vectorizer(path, input_name, prefix, builder, meta):
    pm    = meta.get("paramMap", {})
    data  = _read_parquet_dict(path)
    vocab = list(data.get("vocabulary", [pm.get("vocabulary", [])])[0])
    out   = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("CountVectorizer", input_name, out, [
        builder.string_tensor_attr("vocabulary", f"{prefix}_vocab", vocab),
        omle.Attribute(name="binary", b=bool(pm.get("binary", False))),
    ], builder, node_name=builder.unique_name("count_vectorizer_model"))


def _load_hashing_tf(path, input_name, prefix, builder, meta):
    pm  = meta.get("paramMap", {})
    out = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("HashingVectorizer", input_name, out, [
        omle.Attribute(name="num_features",   i=int(pm.get("numFeatures", 1 << 18))),
        omle.Attribute(name="binary",         b=bool(pm.get("binary", False))),
        omle.Attribute(name="alternate_sign", b=bool(pm.get("alternateSign", False))),
    ], builder, node_name=builder.unique_name("hashing_tf"))


def _load_idf(path, input_name, prefix, builder, meta):
    data = _read_parquet_dict(path)
    pm   = meta.get("paramMap", {})
    out  = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("TfIdfTransformer", input_name, out, [
        builder.tensor_attr("idf", f"{prefix}_idf", _to_array(data["idf"][0])),
    ], builder, node_name=builder.unique_name("idf_model"))


def _load_word2vec(path, input_name, prefix, builder, meta):
    data       = _read_parquet_dict(path)
    pm         = meta.get("paramMap", {})
    vocab      = list(data["word"])
    embeddings = np.array([_to_array(v) for v in data["vector"]], dtype=np.float64)
    out        = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    return _text_node("Word2Vec", input_name, out, [
        builder.string_tensor_attr("vocabulary", f"{prefix}_vocab", vocab),
        builder.tensor_attr("embeddings", f"{prefix}_embeddings", embeddings),
        omle.Attribute(name="pooling", s="average"),
    ], builder, node_name=builder.unique_name("word2_vec_model"))


def _load_best_model(path, input_name, builder, meta):
    return _load_model(os.path.join(path, "bestModel"), input_name, builder)


def _load_one_vs_rest(path, input_name, builder, meta):
    """Convert a Spark OneVsRestModel.

    Spark's OneVsRest has no probabilityCol — it only produces a prediction.
    Each binary sub-model is loaded; its positive-class score is extracted via
    TakeSlots.  The K scores are concatenated and ArgMax picks the winning class.
    """
    num_classes = int(meta.get("numClasses", 2))
    builder.n_classes = num_classes

    def _ovr_model_path(k: int) -> str:
        # Spark 3.x: path/models/{k}/
        # Spark 4.x: path/model_{k}/  (flat, no "models" subdirectory)
        spark3 = os.path.join(path, "models", str(k))
        if os.path.isdir(spark3):
            return spark3
        spark4 = os.path.join(path, f"model_{k}")
        if os.path.isdir(spark4):
            return spark4
        raise FileNotFoundError(
            f"OneVsRestModel sub-model {k} not found under {path} "
            "(tried models/{k}/ and model_{k}/)"
        )

    score_names: List[str] = []
    for k in range(num_classes):
        model_path = _ovr_model_path(k)
        score_name = _load_ovr_binary_model(model_path, input_name, k, builder)
        score_names.append(score_name)

    # Sub-model loaders overwrite builder.n_classes with 2 (binary). Restore.
    builder.n_classes = num_classes

    # Concatenate the K per-class positive-class scores → [n, K] matrix (intermediate).
    scores_name = builder.unique_name("ovr_scores")
    builder.require_namespace("omle.core")
    builder.add_node(omle.Node(
        name=scores_name, domain="omle.core", op="Concat",
        inputs=[omle.NodeInput(name=n) for n in score_names],
        outputs=[omle.NodeOutput(name=scores_name, role=omle.OutputRole.TRANSFORMED_VALUE)],
        attributes=[],
    ))

    # ArgMax over the K scores → the winning class index (multiclass prediction).
    pred_name = builder.unique_name("y_pred")
    builder.require_namespace("omle.core")
    builder.add_node(omle.Node(
        name="one_vs_rest_model", domain="omle.core", op="ArgMax",
        inputs=[omle.NodeInput(name=scores_name)],
        outputs=[omle.NodeOutput(
            name=pred_name,
            role=omle.OutputRole.PREDICTION,
            type=omle.TensorType(dtype=omle.DataType.INT64, shape=[-1]),
        )],
        attributes=[],
    ))
    return [pred_name]


def _load_ovr_binary_model(model_path: str, input_name: str, k: int, builder) -> str:
    """Load one OVR binary sub-model and return the positive-class probability output name.

    All outputs of the newly added nodes are prefixed with ``ovr{k}_`` so that
    K binary models can coexist in the same graph without name collisions.
    PREDICTION and SCORE outputs of sub-model nodes are dropped: they are binary
    0/1 artifacts that must not appear as outputs of the OVR model.
    """
    prefix = f"ovr{k}_"
    n_before = len(builder.nodes)
    _load_model(model_path, input_name, builder)

    # Rename outputs of every newly added node; patch any intra-model input references.
    # After renaming, strip PREDICTION and SCORE outputs — they are binary (0/1)
    # sub-classifier artifacts and must not surface as final model outputs.
    _intermediate_roles = {omle.OutputRole.PREDICTION, omle.OutputRole.SCORE}
    name_remap: dict = {}
    for node in builder.nodes[n_before:]:
        node.name = prefix + node.name
        for inp in node.inputs:
            if inp.name in name_remap:
                inp.name = name_remap[inp.name]
        for out in node.outputs:
            new_name = prefix + out.name
            builder._used_names.add(new_name)
            name_remap[out.name] = new_name
            out.name = new_name
        node.outputs[:] = [o for o in node.outputs if o.role not in _intermediate_roles]

    # Extract the positive-class probability column for concatenation.
    last_node = builder.nodes[-1]
    prob_out = next(
        (o.name for o in last_node.outputs if o.role == omle.OutputRole.PROBABILITY),
        last_node.outputs[-1].name,
    )
    pos_name = builder.unique_name(f"ovr{k}_pos_prob")
    builder.require_namespace("omle.core")
    builder.add_node(omle.Node(
        name=pos_name, domain="omle.core", op="TakeSlots",
        inputs=[omle.NodeInput(name=prob_out)],
        outputs=[omle.NodeOutput(name=pos_name, role=omle.OutputRole.TRANSFORMED_VALUE)],
        attributes=[omle.Attribute(name="indices", ints=[1])],
    ))
    return pos_name


# ── Dispatch tables ────────────────────────────────────────────────────────────

def _load_elementwise_product(path, input_name, prefix, builder, meta):
    pm      = meta.get("paramMap", {})
    scaling = _to_array(pm.get("scalingVec", []))
    out     = pm.get("outputCol") or builder.unique_name(f"{prefix}_out")
    n_feat  = len(scaling)
    out_type = omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_feat]) if n_feat else None
    return core_node("WeightedSum", [input_name], out, [
        builder.tensor_attr("weights", f"{prefix}_weights", scaling),
    ], builder, output_type=out_type, node_name=builder.unique_name("elementwise_product"))


_TRANSFORMER_LOADERS: dict = {
    "StandardScalerModel":              _load_standard_scaler,
    "MinMaxScalerModel":                _load_min_max_scaler,
    "MaxAbsScalerModel":                _load_max_abs_scaler,
    "Normalizer":                       _load_normalizer,
    "Binarizer":                        _load_binarizer,
    "StringIndexerModel":               _load_string_indexer,
    "OneHotEncoderModel":               _load_one_hot_encoder,
    "Bucketizer":                       _load_bucketizer,
    "ImputerModel":                     _load_imputer,
    "PCAModel":                         _load_pca,
    "ElementwiseProduct":               _load_elementwise_product,
    "Tokenizer":                        _load_tokenizer,
    "RegexTokenizer":                   _load_regex_tokenizer,
    "StopWordsRemover":                 _load_stop_words_remover,
    "NGram":                            _load_ngram,
    "CountVectorizerModel":             _load_count_vectorizer,
    "HashingTF":                        _load_hashing_tf,
    "IDFModel":                         _load_idf,
    "Word2VecModel":                    _load_word2vec,
    "VectorSlicer":                     _load_vector_slicer,
    "ChiSqSelectorModel":               _load_feature_selector,
    "UnivariateFeatureSelectorModel":   _load_feature_selector,
    "VarianceThresholdSelectorModel":   _load_feature_selector,
}

_ESTIMATOR_LOADERS: dict = {
    "LinearRegressionModel":                    _load_linear_regression,
    "LogisticRegressionModel":                  _load_logistic_regression,
    "DecisionTreeClassificationModel":          _load_dt_classifier,
    "DecisionTreeRegressionModel":              _load_dt_regressor,
    "RandomForestClassificationModel":          _load_rf_classifier,
    "RandomForestRegressionModel":              _load_rf_regressor,
    "GBTClassificationModel":                   _load_gbt_classifier,
    "GBTRegressionModel":                       _load_gbt_regressor,
    "NaiveBayesModel":                          _load_naive_bayes,
    "KMeansModel":                              _load_kmeans,
    "LinearSVCModel":                           _load_linear_svc,
    "MultilayerPerceptronClassificationModel":  _load_mlp_classifier,
    "GaussianMixtureModel":                     _load_gaussian_mixture,
    "GeneralizedLinearRegressionModel":         _load_generalized_linear_regression,
    "IsotonicRegressionModel":                   _load_isotonic_regression,
    # XGBoost Spark — Python xgboost.spark API and Scala XGBoost4J
    "SparkXGBClassifierModel":                  _load_xgboost_spark,
    "SparkXGBRegressorModel":                   _load_xgboost_spark,
    "XGBoostClassificationModel":               _load_xgboost_spark,
    "XGBoostRegressionModel":                   _load_xgboost_spark,
    # LightGBM Spark — SynapseML LightGBMClassificationModel / LightGBMRegressionModel
    "LightGBMClassificationModel":              _load_lightgbm_spark,
    "LightGBMRegressionModel":                  _load_lightgbm_spark,
    "CrossValidatorModel":                      _load_best_model,
    "TrainValidationSplitModel":                _load_best_model,
    "OneVsRestModel":                           _load_one_vs_rest,
}

_ALL_SUPPORTED = sorted({*_TRANSFORMER_LOADERS, *_ESTIMATOR_LOADERS, "VectorAssembler", "SQLTransformer"})


# ── Single model / pipeline loaders ───────────────────────────────────────────

def _load_model(path: str, input_name: str, builder: Builder) -> List[str]:
    meta = _read_metadata(path)
    cls  = _short_class(meta.get("class", ""))
    if cls in _ESTIMATOR_LOADERS:
        return _ESTIMATOR_LOADERS[cls](path, input_name, builder, meta)
    if cls in _TRANSFORMER_LOADERS:
        prefix = builder.unique_name(cls.lower()[:8])
        out    = _TRANSFORMER_LOADERS[cls](path, input_name, prefix, builder, meta)
        return [out]
    raise NotImplementedError(
        f"No-Spark loader not implemented for: '{cls}'. "
        f"Supported: {_ALL_SUPPORTED}"
    )


def _load_pipeline(
    path: str,
    feature_cols: List[str],
    builder: Builder,
) -> List[str]:
    stages_dir = os.path.join(path, "stages")
    stage_dirs = sorted(
        _glob.glob(os.path.join(stages_dir, "*")),
        key=lambda p: int(os.path.basename(p).split("_")[0]),
    )
    if not stage_dirs:
        raise FileNotFoundError(
            f"No stage directories found in {stages_dir}. "
            "Is this a PipelineModel save directory?"
        )

    # All feature_cols are raw pipeline inputs; each maps to itself as a tensor name.
    col_to_tensor: dict = {col: col for col in feature_cols}
    current = feature_cols[0] if feature_cols else "features"

    for idx, stage_path in enumerate(stage_dirs):
        stage_meta = _read_metadata(stage_path)
        cls        = _short_class(stage_meta.get("class", ""))
        pm         = stage_meta.get("paramMap", {})
        prefix     = builder.unique_name(f"s{idx}_{cls[:8].lower()}")

        if cls == "VectorAssembler":
            input_cols    = pm.get("inputCols", [])
            output_col    = pm.get("outputCol", "features")
            input_tensors = [col_to_tensor.get(c, c) for c in input_cols]
            # Compute the assembled feature width: vector inputs (e.g. OHE output)
            # contribute their actual column count; scalar inputs contribute 1.
            _prior = {o.name: o.type for nd in builder.nodes for o in nd.outputs}
            n_cols = 0
            for _tn in input_tensors:
                _t = _prior.get(_tn)
                if _t and _t.shape and len(_t.shape) >= 2:
                    _w = _t.shape[-1]
                    n_cols = (n_cols + _w) if n_cols >= 0 and _w > 0 else -1
                else:
                    n_cols = (n_cols + 1) if n_cols >= 0 else -1
            # If all inputs resolve to the same tensor (e.g. after a multi-col Imputer),
            # skip the Concat and pass through directly.
            unique = list(dict.fromkeys(input_tensors))
            if len(unique) == 1:
                out = unique[0]
                if builder.nodes:
                    for o in builder.nodes[-1].outputs:
                        if o.name == out and not o.field_names:
                            o.field_names = list(input_cols)
            else:
                out = _load_vector_assembler(input_tensors, prefix, builder)
                # Apply consistent naming: snake_case node name, outputCol tensor name,
                # field_names = inputCols, type = FLOAT64 vector of total assembled width.
                if builder.nodes:
                    last = builder.nodes[-1]
                    last.name = builder.unique_name(_cls_to_snake(cls))
                    for o in last.outputs:
                        if o.name == out:
                            o.field_names = list(input_cols)
                            o.type = omle.TensorType(
                                dtype=omle.DataType.FLOAT64,
                                shape=[-1, n_cols] if n_cols > 0 else [-1, -1],
                            )
                            if output_col != out:
                                o.name = output_col
                                builder._used_names.add(output_col)
                                out = output_col
            col_to_tensor[output_col] = out
            current = out
        elif cls == "SQLTransformer":
            stmt = pm.get("statement", "")
            if not stmt:
                raise ValueError(f"SQLTransformer at stage {idx} has no 'statement' in paramMap")
            col_to_tensor = apply_sql_transformer(stmt, col_to_tensor, prefix, builder)
            # After SQL transform, 'current' is the last known column (unchanged for SELECT *)
        elif cls == "StringIndexerModel" and len(pm.get("inputCols") or []) > 1:
            # Multi-column StringIndexer: resolve each inputCol through col_to_tensor,
            # then emit a single variadic LabelEncoder node (one node, no Concat).
            in_cols  = pm["inputCols"]
            out_cols = pm.get("outputCols", [])
            data     = _read_parquet_dict(stage_path)
            if "labelsArray" in data and data["labelsArray"]:
                labels_arr = list(data["labelsArray"][0])
            else:
                labels_arr = [[] for _ in in_cols]
            all_labels: List[str] = []
            offsets: List[int] = [0]
            for labels in labels_arr:
                all_labels.extend(list(labels))
                offsets.append(len(all_labels))
            resolved_ins = [col_to_tensor.get(c, c) for c in in_cols]
            out_names    = ([builder.unique_name(oc) for oc in out_cols]
                            if out_cols else
                            [builder.unique_name(f"{prefix}_col{i}_out") for i in range(len(in_cols))])
            builder.require_namespace("omle.feature")
            builder.add_node(omle.Node(
                name=builder.unique_name("string_indexer_model"),
                domain="omle.feature", op="LabelEncoder",
                inputs=[omle.NodeInput(name=n) for n in resolved_ins],
                outputs=[omle.NodeOutput(
                             name=n, role=omle.OutputRole.TRANSFORMED_VALUE,
                             type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]),
                         ) for n in out_names],
                attributes=[
                    builder.string_tensor_attr("labels", f"{prefix}_labels", all_labels),
                    builder.int_tensor_attr("label_offsets", f"{prefix}_offsets",
                                           np.array(offsets, dtype=np.int64)),
                ],
            ))
            for out_col, out_name in zip(out_cols, out_names, strict=True):
                col_to_tensor[out_col] = out_name
            current = out_names[-1]
        elif cls == "OneHotEncoderModel" and len(pm.get("inputCols") or []) > 1:
            # Single variadic OHE node: xs=resolved inputs, ys=named outputs per column.
            in_cols  = pm["inputCols"]
            out_cols = pm.get("outputCols", [])
            data     = _read_parquet_dict(stage_path)
            sizes    = list(data["categorySizes"][0])
            drop     = bool(pm.get("dropLast", True))
            all_cats: list[str] = []
            offsets = [0]
            for sz in sizes:
                n = sz - (1 if drop else 0)
                all_cats.extend(str(j) for j in range(n))
                offsets.append(len(all_cats))
            offsets_arr = np.array(offsets, dtype=np.int64)
            resolved_ins = [col_to_tensor.get(c, c) for c in in_cols]
            out_names = []
            for i, _sz in enumerate(sizes):
                n_out = offsets[i + 1] - offsets[i]
                name  = (builder.unique_name(out_cols[i]) if i < len(out_cols)
                         else builder.unique_name(f"{prefix}_col{i}_ohe"))
                out_names.append((name, n_out))
            builder.require_namespace("omle.feature")
            builder.add_node(omle.Node(
                name=builder.unique_name("one_hot_encoder_model"),
                domain="omle.feature", op="OneHotEncoder",
                inputs=[omle.NodeInput(name=c) for c in resolved_ins],
                outputs=[omle.NodeOutput(
                    name=name,
                    role=omle.OutputRole.TRANSFORMED_VALUE,
                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_out]),
                ) for name, n_out in out_names],
                attributes=[
                    builder.string_tensor_attr("categories", f"{prefix}_categories", all_cats),
                    builder.int_tensor_attr("category_offsets", f"{prefix}_offsets", offsets_arr),
                ],
            ))
            for (name, _), out_col in zip(out_names, out_cols, strict=True):
                col_to_tensor[out_col] = name
            current = out_names[-1][0]
        elif cls == "ImputerModel" and pm.get("inputCols") and pm.get("outputCols"):
            # Single Imputer node with variadic xs/ys; resolve each inputCol through
            # col_to_tensor so nodes reference the correct graph tensors.
            in_cols      = pm["inputCols"]
            out_cols     = pm["outputCols"]
            data         = _read_parquet_dict(stage_path)
            resolved_ins = [col_to_tensor.get(c, c) for c in in_cols]
            fills        = np.array([float(data.get(c, [0.0])[0]) for c in in_cols], dtype=np.float64)
            out_names    = [builder.unique_name(oc) for oc in out_cols]
            builder.require_namespace("omle.feature")
            builder.add_node(omle.Node(
                name=builder.unique_name("imputer_model"), domain="omle.feature", op="Imputer",
                inputs=[omle.NodeInput(name=n) for n in resolved_ins],
                outputs=[omle.NodeOutput(name=n, role=omle.OutputRole.TRANSFORMED_VALUE)
                         for n in out_names],
                attributes=[builder.tensor_attr("fill_tensor", f"{prefix}_fill", fills)],
            ))
            for out_col, out_name in zip(out_cols, out_names, strict=True):
                col_to_tensor[out_col] = out_name
            current = out_names[-1]
        elif cls in _TRANSFORMER_LOADERS:
            in_col     = pm.get("inputCol") or (pm.get("inputCols") or [current])[0]
            in_tensor  = col_to_tensor.get(in_col, current)
            out        = _TRANSFORMER_LOADERS[cls](stage_path, in_tensor, prefix, builder, stage_meta)
            output_col = pm.get("outputCol") or (pm.get("outputCols") or [in_col])[0]
            col_to_tensor[output_col] = out
            current = out
        elif cls in _ESTIMATOR_LOADERS:
            feat_col  = pm.get("featuresCol", "features")
            in_tensor = col_to_tensor.get(feat_col, current)
            return _ESTIMATOR_LOADERS[cls](stage_path, in_tensor, builder, stage_meta)
        else:
            raise NotImplementedError(
                f"No-Spark loader not implemented for stage '{cls}' "
                f"(stage directory: {stage_path}). Supported: {_ALL_SUPPORTED}"
            )

    return [current]


# ── Helpers ────────────────────────────────────────────────────────────────────

_STRING_INPUT_CLASSES = frozenset({
    "StringIndexerModel",
    "Tokenizer",
    "RegexTokenizer",
})

# Transformers whose input column is a sequence (array) of strings, not a scalar string.
# Their InputSpec uses shape [-1, -1] (n_rows × variable_tokens).
_SEQUENCE_INPUT_CLASSES = frozenset({
    "StopWordsRemover",
    "NGram",
    "CountVectorizerModel",
    "HashingTF",
    "Word2VecModel",
})

_SCALAR_INPUT_CLASSES = frozenset({
    "OneHotEncoderModel",
    "Bucketizer",
})


def _make_inputs(
    feature_cols: List[str],
    n_features: int,
    input_name: str = "features",
    input_dtype: omle.DataType = omle.DataType.FLOAT64,
    scalar_input: bool = False,
    sequence_input: bool = False,
) -> List[omle.InputSpec]:
    """Build InputSpec list. Always includes dtype and shape.

    n_features > 0   → 2-D float matrix with known column count
    n_features == 0  → 1-D scalar column (runtime-detected; also triggered by scalar_input)
    n_features < 0   → 2-D float matrix with unknown column count (shape [-1, -1])
    input_dtype STRING, sequence_input=False → 1-D string column  (scalar string per row)
    input_dtype STRING, sequence_input=True  → 2-D string array   (token list per row)
    scalar_input     → 1-D float column (class-detected; overrides n_features)
    """
    if feature_cols:
        col_shape = [-1, -1] if sequence_input else [-1]
        return [
            omle.InputSpec(
                name=col,
                type=omle.TensorType(dtype=input_dtype, shape=col_shape),
            )
            for col in feature_cols
        ]
    if input_dtype == omle.DataType.STRING:
        shape = [-1, -1] if sequence_input else [-1]
        return [omle.InputSpec(
            name=input_name,
            type=omle.TensorType(dtype=omle.DataType.STRING, shape=shape),
        )]
    if scalar_input or n_features == 0:
        return [omle.InputSpec(
            name=input_name,
            type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]),
        )]
    if n_features > 0:
        return [omle.InputSpec(
            name=input_name,
            type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, n_features]),
        )]
    # n_features < 0: 2-D matrix with unknown column count
    return [omle.InputSpec(
        name=input_name,
        type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1, -1]),
    )]


_TEXT_STRING_OUTPUT_OPS = frozenset({
    "Tokenizer", "RegexTokenizer", "StopWordsRemover", "NGram",
})


def _make_outputs(builder: Builder) -> List[omle.OutputSpec]:
    if not builder.nodes:
        return []

    # Tensor names consumed as inputs by any node — used to identify leaf outputs.
    consumed = {inp.name for nd in builder.nodes for inp in nd.inputs}

    # Pre-build a map of tensor name → annotated type for Concat field-name lookup.
    tensor_type_map = {
        nd_out.name: nd_out.type
        for nd in builder.nodes
        for nd_out in nd.outputs
    }

    specs = []
    n_classes = builder.n_classes
    for nd in builder.nodes:
        for o in nd.outputs:
            if o.name in consumed:
                continue  # intermediate tensor, not a model output
            if o.role == omle.OutputRole.PROBABILITY:
                shape = [-1, n_classes] if n_classes > 2 else [-1, 2] if n_classes == 2 else [-1, -1]
                specs.append(omle.OutputSpec(
                    name=o.name, role=o.role,
                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=shape),
                ))
            elif o.role == omle.OutputRole.PREDICTION:
                pred_dtype = omle.DataType.FLOAT64 if n_classes == 0 else omle.DataType.INT64
                specs.append(omle.OutputSpec(
                    name=o.name, role=o.role,
                    type=omle.TensorType(dtype=pred_dtype, shape=[-1]),
                ))
            elif o.role == omle.OutputRole.TRANSFORMED_VALUE:
                is_text_seq = (nd.domain == "omle.text" and nd.op in _TEXT_STRING_OUTPUT_OPS)
                if o.field_names and nd.op == "Concat":
                    # Multi-col transformer with a Concat (e.g. StringIndexer outputCols):
                    # emit one OutputSpec per field, resolved to each input's annotated type.
                    for fn, inp in zip(o.field_names, nd.inputs, strict=True):
                        t = tensor_type_map.get(inp.name)
                        specs.append(omle.OutputSpec(name=fn, role=o.role, type=t))
                elif o.type is not None and not is_text_seq:
                    specs.append(omle.OutputSpec(name=o.name, role=o.role, type=o.type))
                else:
                    out_dtype = omle.DataType.STRING if is_text_seq else omle.DataType.FLOAT64
                    out_shape = [-1, -1] if is_text_seq else [-1]
                    specs.append(omle.OutputSpec(
                        name=o.name, role=o.role,
                        type=omle.TensorType(dtype=out_dtype, shape=out_shape),
                    ))
            elif o.role == omle.OutputRole.ENTITY_ID:
                specs.append(omle.OutputSpec(
                    name=o.name, role=o.role,
                    type=omle.TensorType(dtype=omle.DataType.INT32, shape=[-1]),
                ))
            else:  # SCORE and any other roles
                specs.append(omle.OutputSpec(
                    name=o.name, role=o.role,
                    type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[-1]),
                ))
    return specs


def _rename_ml_outputs(builder: Builder, pm: dict) -> None:
    """Rename the last node's ML output tensors to the Spark standard column names.

    Spark models expose well-known column names (prediction, probability,
    rawPrediction) that should be reflected in the OMLE model interface
    instead of generic names like y_pred / y_prob.  The actual names are read
    from the model's paramMap so user-configured column names are preserved.
    """
    if not builder.nodes:
        return
    for o in builder.nodes[-1].outputs:
        if o.role == omle.OutputRole.PREDICTION:
            o.name = pm.get("predictionCol", "prediction")
        elif o.role == omle.OutputRole.PROBABILITY:
            o.name = pm.get("probabilityCol", "probability")
        elif o.role == omle.OutputRole.SCORE:
            o.name = pm.get("rawPredictionCol", "rawPrediction")
        elif o.role == omle.OutputRole.ENTITY_ID:
            o.name = pm.get("predictionCol", "prediction")


def _task_from_builder(builder: Builder) -> TaskType:
    """Infer task type from builder state."""
    if builder.nodes:
        last_outs = builder.nodes[-1].outputs
        if any(o.role == omle.OutputRole.ENTITY_ID for o in last_outs):
            return TaskType.CLUSTERING
    n = builder.n_classes
    if n == 0:
        return TaskType.REGRESSION
    if n == 2:
        return TaskType.BINARY
    return TaskType.MULTICLASS


def _is_transformer_only(builder: Builder) -> bool:
    """Return True when all outputs of the final graph node are TRANSFORMED_VALUE (no ML estimator)."""
    if not builder.nodes:
        return True
    return all(o.role == omle.OutputRole.TRANSFORMED_VALUE for o in builder.nodes[-1].outputs)


# ── Public API ─────────────────────────────────────────────────────────────────

def from_saved_pipeline(
    spark_save_path: str,
    feature_names: Optional[List[str]] = None,
    n_features: int = -1,
    target_name: str = "label",
    class_labels: Optional[List] = None,
    model_name: str = "",
    model_version: str = "",
    copyright: str = "",
) -> omle.OMLEModel:
    """Load and convert a saved PipelineModel without a Spark/JVM dependency.

    Parameters
    ----------
    spark_save_path:
        Path to the directory produced by ``pipeline.save(path)``.
    feature_names:
        Names of the raw input scalar columns fed into the pipeline.
    n_features:
        Total number of features for the input shape.  Automatically inferred from
        the saved model weights (coefficient matrix, cluster centres, etc.) when
        omitted; only needed when auto-inference is not possible (e.g. pure
        transformer pipelines with no estimator stage).
    target_name:
        Name for the prediction output column in ``ModelSchema`` (default ``"label"``).
    class_labels:
        Class label values for classifiers (e.g. ``[0, 1]`` or ``["cat", "dog"]``).
        Stored in ``ModelSchema.targets[0].class_labels``.
    model_name:
        Human-readable name stored in ``ModelMetadata.name``.
    model_version:
        Version string for the model artifact (e.g. ``"1.0.0"``), stored in
        ``ModelMetadata.version``.
    copyright:
        Copyright notice stored in ``ModelMetadata.copyright``.
    """
    feature_cols  = list(feature_names) if feature_names else []
    meta          = _read_metadata(spark_save_path)
    spark_version = meta.get("sparkVersion", "")

    # Scan stage directories once: infer pipeline entry columns + input dtype from
    # the first stage, and find the last estimator stage for output renaming.
    stages_dir  = os.path.join(spark_save_path, "stages")
    stage_dirs  = sorted(_glob.glob(os.path.join(stages_dir, "*")),
                         key=lambda p: int(os.path.basename(p).split("_")[0]))

    # Scan ALL stages to collect every raw pipeline input column, its dtype, and
    # whether it is a "scalar" or "vector" column.
    #   scalar  — individual column assembled by VectorAssembler / Imputer (inputCols)
    #   vector  — dense feature vector consumed by numeric transformer (inputCol singular)
    # A raw input is any column needed by a stage but not produced by a preceding stage.
    _SCALAR = "scalar"
    _VECTOR = "vector"
    raw_inputs: "dict[str, tuple]" = {}  # col → (dtype, kind)
    produced:   "set[str]" = set()

    for sd in stage_dirs:
        sm = _read_metadata(sd)
        sc = _short_class(sm.get("class", ""))
        sp = {**sm.get("defaultParamMap", {}), **sm.get("paramMap", {})}

        # Single inputCol — kind depends on the stage class
        if sp.get("inputCol"):
            col = sp["inputCol"]
            if col not in produced and col not in raw_inputs:
                if sc in _STRING_INPUT_CLASSES:
                    raw_inputs[col] = (omle.DataType.STRING, _SCALAR)
                elif sc in _SEQUENCE_INPUT_CLASSES:
                    raw_inputs[col] = (omle.DataType.STRING, _SCALAR)
                else:
                    # Numeric transformer consuming a single inputCol → dense vector
                    raw_inputs[col] = (omle.DataType.FLOAT64, _VECTOR)

        # Multiple inputCols — dtype depends on the stage class
        for col in sp.get("inputCols", []):
            if col not in produced and col not in raw_inputs:
                if sc in _STRING_INPUT_CLASSES:
                    raw_inputs[col] = (omle.DataType.STRING, _SCALAR)
                else:
                    raw_inputs[col] = (omle.DataType.FLOAT64, _SCALAR)

        # featuresCol for estimators → dense vector
        feat = sp.get("featuresCol")
        if feat and feat not in produced and feat not in raw_inputs:
            raw_inputs[feat] = (omle.DataType.FLOAT64, _VECTOR)

        # Mark columns produced by this stage
        if sp.get("outputCol"):
            produced.add(sp["outputCol"])
        for col in sp.get("outputCols", []):
            produced.add(col)
        for key in ("predictionCol", "probabilityCol", "rawPredictionCol"):
            if sp.get(key):
                produced.add(sp[key])

    # Classify the raw inputs:
    #   "per_col" mode  — multiple SCALAR columns (assembled downstream by VectorAssembler)
    #   "single"  mode  — one VECTOR column (dense feature vector, e.g. "features")
    scalar_inputs = [(col, dt) for col, (dt, k) in raw_inputs.items() if k == _SCALAR]
    vector_inputs = [(col, dt) for col, (dt, k) in raw_inputs.items() if k == _VECTOR]

    # If the caller provided explicit feature_names, honour them.
    if not feature_cols:
        if scalar_inputs:
            # Multiple named scalar columns → per-column mode
            feature_cols = [c for c, _ in scalar_inputs]
        elif len(vector_inputs) == 1:
            # Single vector column → keep old single-tensor approach (with real col name)
            feature_cols = []   # use single-tensor path
        elif vector_inputs:
            # Multiple vector inputs (unusual) → fall back to first
            feature_cols = []

    # The primary vector-input column name (used when feature_cols is empty)
    single_vector_col = vector_inputs[0][0] if vector_inputs and not scalar_inputs else "features"

    # Build InputSpecs:
    #   scalar cols  → per-column, shape [-1]
    #   single vector → one 2-D InputSpec named after the actual Spark column
    def _build_inputs() -> List[omle.InputSpec]:
        if feature_cols:
            # Per-column scalar inputs
            specs = []
            for col in feature_cols:
                dtype = raw_inputs.get(col, (omle.DataType.FLOAT64, _SCALAR))[0]
                specs.append(omle.InputSpec(
                    name=col,
                    type=omle.TensorType(dtype=dtype, shape=[-1]),
                ))
            return specs
        # Single tensor (vector column or unknown)
        dtype = vector_inputs[0][1] if vector_inputs else omle.DataType.FLOAT64
        shape = [-1, n_features] if n_features > 0 else [-1, -1]
        return [omle.InputSpec(
            name=single_vector_col,
            type=omle.TensorType(dtype=dtype, shape=shape),
        )]

    # Extract per-column discrete domains from StringIndexer stages so the schema
    # captures category values for string features (e.g. color → ["cat","dog","bird"]).
    col_domains: "dict[str, omle.ValueDomain]" = {}
    for sd in stage_dirs:
        sm = _read_metadata(sd)
        sc = _short_class(sm.get("class", ""))
        if sc != "StringIndexerModel":
            continue
        sp_exp = sm.get("paramMap", {})
        in_col = sp_exp.get("inputCol")
        if not in_col:
            continue
        try:
            data = _read_parquet_dict(sd)
            # Follow the same unpacking as _load_string_indexer:
            # labelsArray row 0 is a list of per-column label arrays.
            if "labelsArray" in data and data["labelsArray"]:
                labels_arr = list(data["labelsArray"][0])  # [["cat","dog",...], ...]
                labels = list(labels_arr[0]) if labels_arr else []
            elif "labels" in data and data["labels"]:
                labels = list(data["labels"][0])
            else:
                labels = []
            if labels:
                from omle_convert._common import _make_discrete_domain
                col_domains[in_col] = _make_discrete_domain(labels)
        except Exception:
            pass

    # Find the last estimator stage's paramMap for output column renaming.
    last_est_pm: dict = {}
    for sd in reversed(stage_dirs):
        sm  = _read_metadata(sd)
        sc  = _short_class(sm.get("class", ""))
        if sc in _ESTIMATOR_LOADERS:
            last_est_pm = {**sm.get("defaultParamMap", {}), **sm.get("paramMap", {})}
            break

    # Pass ALL raw inputs to _load_pipeline so it can wire scalar AND vector columns.
    all_input_cols = feature_cols if feature_cols else ([single_vector_col] if single_vector_col != "X" else [])
    builder = Builder()
    _load_pipeline(spark_save_path, all_input_cols, builder)
    _rename_ml_outputs(builder, last_est_pm)

    # For transformer-only pipelines (e.g. SQLTransformer) where no inputCol/
    # inputCols/featuresCol appear in metadata, infer feature_cols from Derive node
    # inputs so that _attach_auxiliary can find the right DataFrame columns.
    # NodeInput.name is a NameRef; extract .value to get the plain column name.
    if not feature_cols:
        derive_inputs = list(dict.fromkeys(
            (inp.name.value if hasattr(inp.name, "value") else str(inp.name))
            for nd in builder.nodes if nd.op == "Derive"
            for inp in nd.inputs
        ))
        if derive_inputs:
            feature_cols = derive_inputs
            for col in feature_cols:
                if col not in raw_inputs:
                    # Use STRING dtype if this column exclusively feeds string-output Derive nodes.
                    feeding = [
                        nd for nd in builder.nodes if nd.op == "Derive"
                        and any(
                            (inp.name.value if hasattr(inp.name, "value") else str(inp.name)) == col
                            for inp in nd.inputs
                        )
                    ]
                    is_str = bool(feeding) and all(
                        any(
                            o.type is not None and o.type.dtype == omle.DataType.STRING
                            for o in nd.outputs
                        )
                        for nd in feeding
                    )
                    raw_inputs[col] = (
                        omle.DataType.STRING if is_str else omle.DataType.FLOAT64,
                        _SCALAR,
                    )

    task             = _task_from_builder(builder)
    effective_target = last_est_pm.get("labelCol") or target_name
    effective_labels = class_labels if class_labels is not None else (
        list(range(builder.n_classes)) if builder.n_classes > 0 else None
    )
    # per_feature_inputs=True only for multi-scalar-column pipelines (assembled by VectorAssembler)
    if _is_transformer_only(builder):
        schema = None
    else:
        schema = make_model_schema(feature_cols, task, effective_target, effective_labels,
                                   per_feature_inputs=bool(feature_cols),
                                   feature_dtype=omle.DataType.FLOAT64)

    # Correct schema features: apply each column's actual dtype (STRING vs FLOAT64)
    # and attach discrete domain (categories) extracted from StringIndexer stages.
    for f in (schema.features if schema else []):
        col = f.name
        if not col:
            continue
        col_dtype, _ = raw_inputs.get(col, (omle.DataType.FLOAT64, _SCALAR))
        if f.type is not None:
            f.type = omle.TensorType(dtype=col_dtype, shape=f.type.shape)
        if col in col_domains:
            f.domain  = col_domains[col]
            f.measure_level = omle.MeasureLevel.NOMINAL   # string categorical

    # Use the feature count inferred from model weights if the caller didn't specify it.
    if n_features <= 0 and builder.n_features > 0:
        n_features = builder.n_features  # noqa: PLW0621 — intentional rebind of enclosing var

    built_inputs = _build_inputs()
    return builder.build_model(
        inputs=built_inputs,
        outputs=_make_outputs(builder),
        model_name=model_name,
        version=model_version,
        copyright=copyright,
        model_schema=schema,
        framework_version=spark_version,
    )


def from_saved_model(
    spark_save_path: str,
    feature_names: Optional[List[str]] = None,
    n_features: int = -1,
    target_name: str = "label",
    class_labels: Optional[List] = None,
    model_name: str = "",
    model_version: str = "",
    copyright: str = "",
) -> omle.OMLEModel:
    """Load and convert a single saved Spark ML model without a Spark/JVM dependency.

    Parameters
    ----------
    spark_save_path:
        Path to the directory produced by ``model.save(path)``.
    feature_names:
        Names of the raw input scalar columns.
    n_features:
        Total number of features for the input shape.  Automatically inferred from
        the saved model weights (coefficient matrix, cluster centres, etc.) when
        omitted; only needed when auto-inference is not possible (e.g. pure
        transformer pipelines with no estimator stage).
    target_name:
        Name for the prediction output column in ``ModelSchema`` (default ``"label"``).
    class_labels:
        Class label values for classifiers (e.g. ``[0, 1]`` or ``["cat", "dog"]``).
        Stored in ``ModelSchema.targets[0].class_labels``.
    model_name:
        Human-readable name stored in ``ModelMetadata.name``.
    model_version:
        Version string for the model artifact (e.g. ``"1.0.0"``), stored in
        ``ModelMetadata.version``.
    copyright:
        Copyright notice stored in ``ModelMetadata.copyright``.
    """
    feature_cols = list(feature_names) if feature_names else []
    meta         = _read_metadata(spark_save_path)
    spark_version = meta.get("sparkVersion", "")
    cls           = _short_class(meta.get("class", ""))
    # Merge defaultParamMap (class defaults) under explicit paramMap values so
    # that columns using their default name (e.g. featuresCol="features") are
    # still reflected correctly even when not written into paramMap.
    pm = {**meta.get("defaultParamMap", {}), **meta.get("paramMap", {})}

    explicit_pm      = meta.get("paramMap", {})
    output_cols_list = explicit_pm.get("outputCols") or []
    multi_col        = bool(output_cols_list) and len(output_cols_list) > 1

    # Multi-column transformers (e.g. ImputerModel with inputCols): auto-infer
    # feature_cols from inputCols so the model gets one InputSpec per column.
    if not feature_cols and multi_col and explicit_pm.get("inputCols"):
        feature_cols = list(explicit_pm["inputCols"])

    # Use the actual Spark column names from the model metadata so the OMLE
    # model reflects the real data flow (e.g. "features" → "scaled_features").
    # Per-feature mode (feature_cols provided) always uses a single "X" graph input.
    # For multi-col models, graph_input is set to feature_cols[0] so _load_imputer
    # can use it as the resolved name for the first column.
    if feature_cols and multi_col:
        graph_input = feature_cols[0]
    elif feature_cols:
        graph_input = "X"
    else:
        graph_input = (pm.get("inputCol") or
                       (pm.get("inputCols") or [None])[0] or
                       pm.get("featuresCol") or "features")

    input_dtype     = (omle.DataType.STRING
                       if cls in _STRING_INPUT_CLASSES or cls in _SEQUENCE_INPUT_CLASSES
                       else omle.DataType.FLOAT64)
    scalar_input    = cls in _SCALAR_INPUT_CLASSES or (cls == "ImputerModel" and not multi_col)
    sequence_input  = cls in _SEQUENCE_INPUT_CLASSES

    builder = Builder()
    _load_model(spark_save_path, graph_input, builder)

    # Rename ML estimator outputs to Spark's standard column names.
    _rename_ml_outputs(builder, pm)

    task          = _task_from_builder(builder)
    effective_target = pm.get("labelCol") or target_name
    effective_labels = class_labels if class_labels is not None else (
        list(range(builder.n_classes)) if builder.n_classes > 0 else None
    )
    if _is_transformer_only(builder):
        schema = None
    else:
        schema = make_model_schema(feature_cols, task, effective_target, effective_labels,
                                   per_feature_inputs=bool(feature_cols),
                                   feature_dtype=omle.DataType.FLOAT64)

    outputs = _make_outputs(builder)

    resolved_n_features = n_features if n_features > 0 else builder.n_features
    return builder.build_model(
        inputs=_make_inputs(feature_cols, resolved_n_features, input_name=graph_input,
                            input_dtype=input_dtype, scalar_input=scalar_input,
                            sequence_input=sequence_input),
        outputs=outputs,
        model_name=model_name,
        version=model_version,
        copyright=copyright,
        model_schema=schema,
        framework_version=spark_version,
    )
