"""Builder: accumulates nodes, constants, and namespace imports."""

from __future__ import annotations

import numpy as np

import omle

from .._common import _inline_threshold


class Builder:
    """Accumulates nodes, constants, and namespace imports for an OMLE model."""

    def __init__(self) -> None:
        self.nodes: list[omle.Node] = []
        self.tensor_entries: list[omle.TensorEntry] = []
        self._used_names: set[str] = set()
        self._used_node_names: set[str] = set()
        self._namespaces: set[str] = set()
        self.str_columns: set[str] = set()

    def unique_name(self, base: str) -> str:
        """Return a tensor/data-flow name derived from *base* that has not been used yet."""
        name, i = base, 0
        while name in self._used_names:
            i += 1
            name = f"{base}_{i}"
        self._used_names.add(name)
        return name

    def unique_node_name(self, base: str) -> str:
        """Return a node label derived from *base* that has not been used yet."""
        name, i = base, 0
        while name in self._used_node_names:
            i += 1
            name = f"{base}_{i}"
        self._used_node_names.add(name)
        return name

    def add_tensor(self, base_name: str, data: np.ndarray) -> str:
        """Store a float64 constant tensor and return its unique name."""
        name = self.unique_name(base_name)
        flat = data.ravel()
        shape = list(data.shape) if data.ndim > 1 else [len(flat)]
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            dense=omle.Tensor(
                name=name,
                type=omle.TensorType(
                    dtype=omle.DataType.FLOAT64,
                    shape=shape,
                ),
                float64_data=flat.tolist(),
            )
        ))
        return name

    def add_int_tensor(self, base_name: str, data: np.ndarray) -> str:
        """Store an int64 constant tensor and return its unique name."""
        name = self.unique_name(base_name)
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            dense=omle.Tensor(
                name=name,
                type=omle.TensorType(
                    dtype=omle.DataType.INT64,
                    shape=list(data.shape) if data.ndim > 1 else [len(data)],
                ),
                int64_data=data.ravel().astype(np.int64).tolist(),
            )
        ))
        return name

    def add_string_tensor(self, base_name: str, strings: list[str]) -> str:
        """Store a STRING constant tensor and return its unique name."""
        name = self.unique_name(base_name)
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            dense=omle.Tensor(
                name=name,
                type=omle.TensorType(
                    dtype=omle.DataType.STRING,
                    shape=[len(strings)],
                ),
                string_data=list(strings),
            )
        ))
        return name

    def add_sparse_tensor(self, base_name: str, X) -> str:
        """Store a float64 sparse CSR constant and return its unique name.

        Falls back to ``add_tensor`` when scipy is not available.
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            return self.add_tensor(base_name, np.asarray(X, dtype=np.float64))
        csr = sp.csr_matrix(X, dtype=np.float64)
        name = self.unique_name(base_name)
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            sparse=omle.SparseTensor(
                name=name,
                csr=omle.CSRMatrix(
                    indices=csr.indices.tolist(),
                    indptr=csr.indptr.tolist(),
                    float64_data=csr.data.tolist(),
                ),
            )
        ))
        return name

    def body_tensor_value(
        self,
        base_name: str,
        data: np.ndarray,
        dtype: omle.DataType = omle.DataType.FLOAT64,
    ) -> omle.TensorValue:
        """TensorValue for model-body parameter, following OMLE_INLINE_TENSOR_LIMIT.

        Small tensors (≤ threshold) are stored inline; large ones go to
        tensor_entries and are referenced via TensorRef.
        """
        flat  = data.ravel()
        n     = len(flat)
        shape = list(data.shape) if data.ndim > 1 else [n]
        ttype = omle.TensorType(dtype=dtype, shape=shape)
        if n <= _inline_threshold():
            if dtype == omle.DataType.FLOAT32:
                t = omle.Tensor(type=ttype, float32_data=flat.astype(np.float32).tolist())
            else:
                t = omle.Tensor(type=ttype, float64_data=flat.astype(np.float64).tolist())
            return omle.TensorValue(tensor=t)
        name = self.add_tensor(base_name, data) if dtype != omle.DataType.FLOAT32 \
               else self.add_tensor(base_name, data.astype(np.float64))
        return omle.TensorValue(tensor_ref=omle.TensorRef(id=name))

    def tensor_attr(self, attr_name: str, base_name: str, data: np.ndarray) -> omle.Attribute:
        """FLOAT64 Attribute: inline if ≤ threshold elements, tensor_ref otherwise."""
        flat = data.ravel()
        shape = list(data.shape) if data.ndim > 1 else [len(flat)]
        if flat.size <= _inline_threshold():
            return omle.Attribute(name=attr_name, tensor=omle.Tensor(
                name=attr_name,
                type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=shape),
                float64_data=flat.tolist(),
            ))
        name = self.add_tensor(base_name, data)
        return omle.Attribute(name=attr_name, tensor_ref=omle.TensorRef(id=name))

    def int_tensor_attr(self, attr_name: str, base_name: str, data: np.ndarray) -> omle.Attribute:
        """INT64 Attribute: inline if ≤ threshold elements, tensor_ref otherwise."""
        arr = data.ravel().astype(np.int64)
        shape = list(data.shape) if data.ndim > 1 else [len(arr)]
        if arr.size <= _inline_threshold():
            return omle.Attribute(name=attr_name, tensor=omle.Tensor(
                name=attr_name,
                type=omle.TensorType(dtype=omle.DataType.INT64, shape=shape),
                int64_data=arr.tolist(),
            ))
        name = self.add_int_tensor(base_name, data)
        return omle.Attribute(name=attr_name, tensor_ref=omle.TensorRef(id=name))

    def string_tensor_attr(self, attr_name: str, base_name: str, strings: list[str]) -> omle.Attribute:
        """STRING Attribute: inline if ≤ threshold elements, tensor_ref otherwise."""
        if len(strings) <= _inline_threshold():
            return omle.Attribute(name=attr_name, tensor=omle.Tensor(
                name=attr_name,
                type=omle.TensorType(dtype=omle.DataType.STRING, shape=[len(strings)]),
                string_data=list(strings),
            ))
        name = self.add_string_tensor(base_name, strings)
        return omle.Attribute(name=attr_name, tensor_ref=omle.TensorRef(id=name))

    def add_node(self, node: omle.Node) -> None:
        self._namespaces.add(node.domain)
        self.nodes.append(node)

    def namespace_imports(self) -> list[omle.NamespaceImport]:
        return [omle.NamespaceImport(namespace=ns, version="0.1")
                for ns in sorted(self._namespaces) if ns]


def _node_inputs(input_name) -> list[omle.NodeInput]:
    """Return NodeInput list for *input_name* (str or list[str])."""
    if isinstance(input_name, list):
        return [omle.NodeInput(name=n) for n in input_name]
    return [omle.NodeInput(name=input_name)]


def out_type(n_cols: int,
             dtype: omle.DataType = omle.DataType.FLOAT64) -> omle.TensorType:
    """Return a TensorType for a 2-D output with dynamic batch dimension."""
    return omle.TensorType(dtype=dtype, shape=[-1, n_cols])
