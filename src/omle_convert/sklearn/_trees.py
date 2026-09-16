"""Low-level tree parsing: sklearn Tree_ and HistGradientBoosting nodes → OMLE Tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from omle.ir.bodies import Tree
from omle.ir.enums import TreeNodeKind, TreeSplitOp

if TYPE_CHECKING:
    from ._builder import Builder

# sklearn internal sentinel value for leaf nodes
_TREE_LEAF = -1


def cls_prob(v: np.ndarray, k: int) -> float:
    """Normalised probability of class *k* from a raw sklearn leaf value array."""
    total = float(v.sum())
    return float(v[k]) / total if total > 0.0 else 0.0


def parse_dt_tree(tree_, leaf_fn, builder: "Builder | None" = None) -> Tree:
    """Convert a sklearn ``Tree_`` internal object to an OMLE ``Tree``.

    Nodes are reordered from sklearn's DFS layout to BFS order.
    sklearn split rule: ``x[feature] <= threshold`` → left child.

    Parameters
    ----------
    tree_:
        The ``estimator.tree_`` attribute of a fitted sklearn decision tree.
    leaf_fn:
        Callable applied to ``tree_.value[node_id, 0, :]`` at each leaf to
        produce the scalar leaf value stored in the OMLE Tree.
    """
    # BFS traversal from root (node 0)
    bfs_order: list[int] = []
    queue = [0]
    while queue:
        nid = queue.pop(0)
        bfs_order.append(nid)
        if tree_.children_left[nid] != _TREE_LEAF:
            queue.append(int(tree_.children_left[nid]))
            queue.append(int(tree_.children_right[nid]))

    id_map = {orig: seq for seq, orig in enumerate(bfs_order)}

    node_kind:       list[TreeNodeKind] = []
    split_feature:   list[int]          = []
    split_threshold: list[float]        = []
    split_op:        list[TreeSplitOp]  = []
    children_index:  list[int]          = []
    children_offset: list[int]          = []
    children_count:  list[int]          = []
    default_child:   list[int]          = []
    leaf_value:      list[float]        = []

    # sklearn 1.4+ stores per-node missing direction; older versions default to left
    missing_left = getattr(tree_, "missing_go_to_left", None)

    child_ptr = 0
    for orig_id in bfs_order:
        children_offset.append(child_ptr)

        if tree_.children_left[orig_id] == _TREE_LEAF:
            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            children_count.append(0)
            default_child.append(0)
            leaf_value.append(leaf_fn(tree_.value[orig_id, 0, :]))
        else:
            node_kind.append(TreeNodeKind.BRANCH)
            split_feature.append(int(tree_.feature[orig_id]))
            split_threshold.append(float(tree_.threshold[orig_id]))
            split_op.append(TreeSplitOp.LESS_OR_EQUAL)

            left_seq  = id_map[int(tree_.children_left[orig_id])]
            right_seq = id_map[int(tree_.children_right[orig_id])]
            children_index.extend([left_seq, right_seq])
            children_count.append(2)
            child_ptr += 2

            goes_left = bool(missing_left[orig_id]) if missing_left is not None else True
            default_child.append(left_seq if goes_left else right_seq)
            leaf_value.append(0.0)

    if builder is not None:
        st_tv = builder.body_tensor_value("split_threshold", np.array(split_threshold))
        lv_tv = builder.body_tensor_value("leaf_value", np.array(leaf_value))
    else:
        from omle.ir.tensor import Tensor
        from omle.ir.types import TensorValue
        st_tv = TensorValue.of_tensor(Tensor(float64_data=split_threshold))
        lv_tv = TensorValue.of_tensor(Tensor(float64_data=leaf_value))

    return Tree(
        num_nodes=len(bfs_order),
        node_kind=node_kind,
        split_feature=split_feature,
        split_threshold=st_tv,
        split_op=split_op,
        children_index=children_index,
        children_offset=children_offset,
        children_count=children_count,
        default_child=default_child,
        leaf_value=lv_tv,
    )


def parse_dt_tree_multiclass(tree_, n_classes: int, builder: "Builder | None" = None) -> Tree:
    """Convert a sklearn Tree_ to an OMLE Tree with vector leaves for multiclass.

    Each leaf stores a flat probability vector of length n_classes in leaf_vector,
    addressed by leaf_vector_index.  Branch nodes store 0 in leaf_vector_index.
    """
    bfs_order: list[int] = []
    queue = [0]
    while queue:
        nid = queue.pop(0)
        bfs_order.append(nid)
        if tree_.children_left[nid] != _TREE_LEAF:
            queue.append(int(tree_.children_left[nid]))
            queue.append(int(tree_.children_right[nid]))

    id_map = {orig: seq for seq, orig in enumerate(bfs_order)}

    node_kind:         list[TreeNodeKind] = []
    split_feature:     list[int]          = []
    split_threshold:   list[float]        = []
    split_op:          list[TreeSplitOp]  = []
    children_index:    list[int]          = []
    children_offset:   list[int]          = []
    children_count:    list[int]          = []
    default_child:     list[int]          = []
    leaf_vector:       list[float]        = []
    leaf_vector_index: list[int]          = []

    missing_left = getattr(tree_, "missing_go_to_left", None)
    child_ptr = 0
    vec_ptr   = 0

    for orig_id in bfs_order:
        children_offset.append(child_ptr)

        if tree_.children_left[orig_id] == _TREE_LEAF:
            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            children_count.append(0)
            default_child.append(0)
            leaf_vector_index.append(vec_ptr)
            vals  = tree_.value[orig_id, 0, :]
            total = float(vals.sum())
            for k in range(n_classes):
                leaf_vector.append(float(vals[k]) / total if total > 0.0 else 0.0)
            vec_ptr += n_classes
        else:
            node_kind.append(TreeNodeKind.BRANCH)
            split_feature.append(int(tree_.feature[orig_id]))
            split_threshold.append(float(tree_.threshold[orig_id]))
            split_op.append(TreeSplitOp.LESS_OR_EQUAL)

            left_seq  = id_map[int(tree_.children_left[orig_id])]
            right_seq = id_map[int(tree_.children_right[orig_id])]
            children_index.extend([left_seq, right_seq])
            children_count.append(2)
            child_ptr += 2

            goes_left = bool(missing_left[orig_id]) if missing_left is not None else True
            default_child.append(left_seq if goes_left else right_seq)
            leaf_vector_index.append(0)

    if builder is not None:
        st_tv = builder.body_tensor_value("split_threshold", np.array(split_threshold))
        lv_tv = builder.body_tensor_value("leaf_vector", np.array(leaf_vector))
    else:
        from omle.ir.tensor import Tensor
        from omle.ir.types import TensorValue
        st_tv = TensorValue.of_tensor(Tensor(float64_data=split_threshold))
        lv_tv = TensorValue.of_tensor(Tensor(float64_data=leaf_vector))

    return Tree(
        num_nodes=len(bfs_order),
        node_kind=node_kind,
        split_feature=split_feature,
        split_threshold=st_tv,
        split_op=split_op,
        children_index=children_index,
        children_offset=children_offset,
        children_count=children_count,
        default_child=default_child,
        leaf_width=n_classes,
        leaf_vector=lv_tv,
        leaf_vector_index=leaf_vector_index,
    )


def make_bias_tree(value: float, builder: "Builder | None" = None) -> Tree:
    """Create a single-leaf Tree that always returns *value*."""
    if builder is not None:
        st_tv = builder.body_tensor_value("split_threshold", np.array([0.0]))
        lv_tv = builder.body_tensor_value("leaf_value", np.array([value]))
    else:
        from omle.ir.tensor import Tensor
        from omle.ir.types import TensorValue
        st_tv = TensorValue.of_tensor(Tensor(float64_data=[0.0]))
        lv_tv = TensorValue.of_tensor(Tensor(float64_data=[value]))

    return Tree(
        num_nodes=1,
        node_kind=[TreeNodeKind.LEAF],
        split_feature=[0],
        split_threshold=st_tv,
        split_op=[TreeSplitOp.SPLIT_OP_UNSPECIFIED],
        children_index=[],
        children_offset=[0],
        children_count=[0],
        default_child=[0],
        leaf_value=lv_tv,
    )


def parse_histgb_nodes(nodes, builder: "Builder | None" = None) -> Tree:
    """Convert a HistGradientBoosting ``TreePredictor.nodes`` array to an OMLE Tree.

    Nodes are stored as a flat structured numpy array; leaves have ``is_leaf == 1``,
    and branch children are referenced by ``left`` / ``right`` indices.
    Split rule: ``x[feature_idx] <= num_threshold`` → left child.
    HGB leaf values already incorporate ``learning_rate``.
    """
    bfs_order: list[int] = []
    queue = [0]
    while queue:
        nid = queue.pop(0)
        bfs_order.append(nid)
        if not nodes[nid]["is_leaf"]:
            queue.append(int(nodes[nid]["left"]))
            queue.append(int(nodes[nid]["right"]))

    id_map = {orig: seq for seq, orig in enumerate(bfs_order)}

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
    for orig_id in bfs_order:
        nd = nodes[orig_id]
        children_offset.append(child_ptr)

        if nd["is_leaf"]:
            node_kind.append(TreeNodeKind.LEAF)
            split_feature.append(0)
            split_threshold.append(0.0)
            split_op.append(TreeSplitOp.SPLIT_OP_UNSPECIFIED)
            children_count.append(0)
            default_child.append(0)
            leaf_value.append(float(nd["value"]))
        else:
            node_kind.append(TreeNodeKind.BRANCH)
            split_feature.append(int(nd["feature_idx"]))
            split_threshold.append(float(nd["num_threshold"]))
            split_op.append(TreeSplitOp.LESS_OR_EQUAL)

            left_seq  = id_map[int(nd["left"])]
            right_seq = id_map[int(nd["right"])]
            children_index.extend([left_seq, right_seq])
            children_count.append(2)
            child_ptr += 2

            default_child.append(left_seq if nd["missing_go_to_left"] else right_seq)
            leaf_value.append(0.0)

    if builder is not None:
        st_tv = builder.body_tensor_value("split_threshold", np.array(split_threshold))
        lv_tv = builder.body_tensor_value("leaf_value", np.array(leaf_value))
    else:
        from omle.ir.tensor import Tensor
        from omle.ir.types import TensorValue
        st_tv = TensorValue.of_tensor(Tensor(float64_data=split_threshold))
        lv_tv = TensorValue.of_tensor(Tensor(float64_data=leaf_value))

    return Tree(
        num_nodes=len(bfs_order),
        node_kind=node_kind,
        split_feature=split_feature,
        split_threshold=st_tv,
        split_op=split_op,
        children_index=children_index,
        children_offset=children_offset,
        children_count=children_count,
        default_child=default_child,
        leaf_value=lv_tv,
    )
