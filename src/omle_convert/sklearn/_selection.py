"""sklearn feature selection → OMLE: TakeSlots using get_support(indices=True)."""
from __future__ import annotations

import omle

from ._builder import Builder, out_type

_SELECTION_CLASSES: frozenset[str] = frozenset({
    "SelectKBest",
    "SelectPercentile",
    "SelectFdr",
    "SelectFpr",
    "SelectFwe",
    "VarianceThreshold",
    "SelectFromModel",
    "RFE",
    "RFECV",
    "SequentialFeatureSelector",
})


def _feature_selector(t, input_name: str, prefix: str, builder: Builder) -> str:
    indices = t.get_support(indices=True).tolist()
    out = builder.unique_name(f"{prefix}_take_slots")
    builder.add_node(omle.Node(
        name=out,
        domain="omle.core",
        op="TakeSlots",
        inputs=[omle.NodeInput(name=input_name)],
        outputs=[omle.NodeOutput(name=out, type=out_type(len(indices)))],
        attributes=[omle.Attribute(name="indices", ints=indices)],
    ))
    return out


_SELECTION_DISPATCH = {cls: _feature_selector for cls in _SELECTION_CLASSES}
