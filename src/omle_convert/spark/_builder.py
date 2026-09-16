"""Builder: accumulates nodes, tensors, and namespace imports for OMLE models."""
from __future__ import annotations

from typing import List

import numpy as np

import omle

from .._common import _inline_threshold


class Builder:
    """Accumulates graph nodes and tensor entries, then assembles an OMLEModel."""

    def __init__(self) -> None:
        self.nodes: List[omle.Node] = []
        self.tensor_entries: List[omle.TensorEntry] = []
        self._used_names: set = set()
        self._namespaces: set = set()
        self.n_classes: int = 0   # set by multiclass loaders; 0 = unknown/binary
        self.n_features: int = -1  # set by estimator loaders; -1 = unknown

    def unique_name(self, base: str) -> str:
        name, i = base, 0
        while name in self._used_names:
            i += 1
            name = f"{base}_{i}"
        self._used_names.add(name)
        return name

    def require_namespace(self, ns: str) -> None:
        self._namespaces.add(ns)

    def add_tensor(self, base_name: str, data: np.ndarray) -> str:
        name = self.unique_name(base_name)
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            dense=omle.Tensor(
                name=name,
                type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=list(data.shape)),
                float64_data=data.ravel().tolist(),
            ),
        ))
        return name

    def add_tensor_2d(self, base_name: str, rows: int, cols: int, data: np.ndarray) -> str:
        name = self.unique_name(base_name)
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            dense=omle.Tensor(
                name=name,
                type=omle.TensorType(dtype=omle.DataType.FLOAT64, shape=[rows, cols]),
                float64_data=data.ravel().tolist(),
            ),
        ))
        return name

    def add_string_tensor(self, base_name: str, strings: List[str]) -> str:
        name = self.unique_name(base_name)
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            dense=omle.Tensor(
                name=name,
                type=omle.TensorType(dtype=omle.DataType.STRING),
                string_data=strings,
            ),
        ))
        return name

    def add_int_tensor(self, base_name: str, data: np.ndarray) -> str:
        name = self.unique_name(base_name)
        self.tensor_entries.append(omle.TensorEntry(
            id=name,
            dense=omle.Tensor(
                name=name,
                type=omle.TensorType(dtype=omle.DataType.INT64, shape=[len(data)]),
                int64_data=data.ravel().astype(np.int64).tolist(),
            ),
        ))
        return name

    def add_node(self, node: omle.Node) -> None:
        self.nodes.append(node)

    @staticmethod
    def attr_f(name: str, v: float) -> omle.Attribute:
        return omle.Attribute(name=name, f64=float(v))

    @staticmethod
    def attr_i(name: str, v: int) -> omle.Attribute:
        return omle.Attribute(name=name, i=int(v))

    @staticmethod
    def attr_s(name: str, v: str) -> omle.Attribute:
        return omle.Attribute(name=name, s=str(v))

    @staticmethod
    def attr_b(name: str, v: bool) -> omle.Attribute:
        return omle.Attribute(name=name, b=bool(v))

    @staticmethod
    def attr_fs(name: str, vs: List[float]) -> omle.Attribute:
        return omle.Attribute(name=name, float64s=[float(v) for v in vs])

    @staticmethod
    def attr_is(name: str, vs: List[int]) -> omle.Attribute:
        return omle.Attribute(name=name, ints=[int(v) for v in vs])

    @staticmethod
    def attr_ss(name: str, vs: List[str]) -> omle.Attribute:
        return omle.Attribute(name=name, strings=list(vs))

    @staticmethod
    def attr_tensor(name: str, tensor_id: str) -> omle.Attribute:
        return omle.Attribute(name=name, tensor_ref=omle.TensorRef(id=tensor_id))

    def body_tensor_value(
        self,
        base_name: str,
        data: np.ndarray,
        dtype: omle.DataType = omle.DataType.FLOAT64,
    ) -> omle.TensorValue:
        """TensorValue for model-body parameter, following OMLE_INLINE_TENSOR_LIMIT."""
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
        name = self.add_tensor(base_name, data.astype(np.float64))
        return omle.TensorValue(tensor_ref=omle.TensorRef(id=name))

    def tensor_attr(self, attr_name: str, base_name: str, data: np.ndarray) -> omle.Attribute:
        """FLOAT64 Attribute: inline tensor if ≤ threshold elements, tensor_ref otherwise."""
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
        """INT64 Attribute: inline tensor if ≤ threshold elements, tensor_ref otherwise."""
        arr = data.ravel().astype(np.int64)
        if arr.size <= _inline_threshold():
            return omle.Attribute(name=attr_name, tensor=omle.Tensor(
                name=attr_name,
                type=omle.TensorType(dtype=omle.DataType.INT64, shape=[len(arr)]),
                int64_data=arr.tolist(),
            ))
        name = self.add_int_tensor(base_name, data)
        return omle.Attribute(name=attr_name, tensor_ref=omle.TensorRef(id=name))

    def string_tensor_attr(self, attr_name: str, base_name: str, strings: List[str]) -> omle.Attribute:
        """STRING Attribute: inline tensor if ≤ threshold elements, tensor_ref otherwise."""
        if len(strings) <= _inline_threshold():
            return omle.Attribute(name=attr_name, tensor=omle.Tensor(
                name=attr_name,
                type=omle.TensorType(dtype=omle.DataType.STRING, shape=[len(strings)]),
                string_data=list(strings),
            ))
        name = self.add_string_tensor(base_name, strings)
        return omle.Attribute(name=attr_name, tensor_ref=omle.TensorRef(id=name))

    def build_model(
        self,
        inputs: List[omle.InputSpec] | None = None,
        outputs: List[omle.OutputSpec] | None = None,
        model_name: str = "",
        version: str = "",
        timestamp: str = "",
        copyright: str = "",
        model_schema: "omle.ModelSchema | None" = None,
        framework_version: str = "",
    ) -> omle.OMLEModel:
        from .._common import make_metadata
        operator_imports = [
            omle.NamespaceImport(namespace=ns, version="0.1")
            for ns in sorted(self._namespaces)
        ]
        return omle.OMLEModel(
            metadata=make_metadata("spark", framework_version, model_name, version=version, timestamp=timestamp, copyright=copyright),
            operator_imports=operator_imports,
            inputs=inputs or [],
            outputs=outputs or [],
            model_schema=model_schema,
            nodes=list(self.nodes),
            tensor_entries=list(self.tensor_entries),
        )


def feature_node(
    op: str,
    input_name: str,
    output_name: str,
    attrs: List[omle.Attribute],
    builder: Builder,
    field_names: List[str] = None,
    output_type: "omle.TensorType | None" = None,
    node_name: str = "",
) -> str:
    builder.require_namespace("omle.feature")
    builder.add_node(omle.Node(
        name=node_name or output_name,
        domain="omle.feature",
        op=op,
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(
            name=output_name,
            role=omle.OutputRole.TRANSFORMED_VALUE,
            field_names=field_names or [],
            type=output_type,
        )],
        attributes=attrs,
    ))
    return output_name


def core_node(
    op: str,
    input_names: List[str],
    output_name: str,
    attrs: List[omle.Attribute],
    builder: Builder,
    field_names: List[str] = None,
    output_type: "omle.TensorType | None" = None,
    node_name: str = "",
) -> str:
    builder.require_namespace("omle.core")
    builder.add_node(omle.Node(
        name=node_name or output_name,
        domain="omle.core",
        op=op,
        inputs=[omle.NodeInput(name=n) for n in input_names],
        outputs=[omle.NodeOutput(
            name=output_name,
            role=omle.OutputRole.TRANSFORMED_VALUE,
            field_names=field_names or [],
            type=output_type,
        )],
        attributes=attrs,
    ))
    return output_name
