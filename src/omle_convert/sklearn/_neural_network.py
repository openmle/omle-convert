"""sklearn.neural_network → OMLE: MLPClassifier, MLPRegressor."""
from __future__ import annotations

import numpy as np

import omle
from omle.ir.bodies import DenseLayer, NeuralNetwork
from omle.ir.enums import NeuralNetworkActivation, TaskType

from .._common import make_node_outputs
from ._builder import Builder, _node_inputs

_NN_CLASSES: frozenset[str] = frozenset({"MLPClassifier", "MLPRegressor"})

_ACTIVATION_MAP: dict[str, NeuralNetworkActivation] = {
    "relu":     NeuralNetworkActivation.RELU,
    "tanh":     NeuralNetworkActivation.TANH,
    "logistic": NeuralNetworkActivation.LOGISTIC,
    "identity": NeuralNetworkActivation.IDENTITY,
    "softmax":  NeuralNetworkActivation.SOFTMAX,
}


def _convert_mlp(estimator, input_name, task: TaskType, n_classes: int, builder: Builder) -> None:
    """Convert an sklearn MLPClassifier/MLPRegressor to a NeuralNetwork node."""
    ir_task = task
    hidden_act = _ACTIVATION_MAP.get(
        getattr(estimator, "activation", "relu"),
        NeuralNetworkActivation.RELU,
    )
    out_act = _ACTIVATION_MAP.get(
        getattr(estimator, "out_activation_", "identity"),
        NeuralNetworkActivation.IDENTITY,
    )

    layers: list[DenseLayer] = []
    n_layers = len(estimator.coefs_)
    for i, (W, b) in enumerate(zip(estimator.coefs_, estimator.intercepts_, strict=True)):
        act = out_act if i == n_layers - 1 else hidden_act
        # sklearn coefs_[i] has shape [in_features, out_features]; runtime expects [out, in]
        layers.append(DenseLayer(
            weights=builder.body_tensor_value(f"mlp_W{i}", np.asarray(W.T, dtype=np.float64)),
            bias=builder.body_tensor_value(f"mlp_b{i}", np.asarray(b, dtype=np.float64)),
            activation=act,
            name=f"layer_{i}",
        ))

    builder.add_node(omle.Node(
        name="neural_network",
        domain="omle.ml",
        op="NeuralNetwork",
        inputs=_node_inputs(input_name),
        outputs=make_node_outputs(task, n_classes),
        neural_network=NeuralNetwork(task_type=ir_task, layers=layers),
    ))
