"""Adapters that let XGBoost/LightGBM sklearn-compatible estimators run
through the sklearn dispatcher (``convert_estimator_to_node``).

XGBClassifier, XGBRegressor, LGBMClassifier, and LGBMRegressor are all
sklearn BaseEstimator subclasses and can appear anywhere a sklearn estimator
can: as the final step of a Pipeline, inside StackingClassifier, etc.
These adapters extract the underlying booster and delegate to the same tree
conversion logic used by the standalone xgboost/lightgbm converters.
"""
from __future__ import annotations

import omle

from .._common import TaskType, make_feature_index, make_node_outputs
from ._builder import Builder, _node_inputs

_XGB_CLASSES: frozenset[str] = frozenset({
    "XGBClassifier", "XGBRegressor",
    "XGBRFClassifier", "XGBRFRegressor",
    "XGBRanker",
})
_LGB_CLASSES: frozenset[str] = frozenset({"LGBMClassifier", "LGBMRegressor", "LGBMRanker"})


def _convert_xgb(
    estimator,
    input_name,
    task: TaskType,
    n_classes: int,
    builder: Builder,
) -> None:
    """Convert an XGBClassifier or XGBRegressor node into the builder."""
    from omle_convert.xgboost import (
        _base_score,
        _convert_trees,
        _get_booster,
        _make_ensemble,
    )
    booster = _get_booster(estimator)
    fn = booster.feature_names or [f"f{i}" for i in range(booster.num_features())]
    feat_index = make_feature_index(fn)
    trees, tree_group = _convert_trees(booster, feat_index, n_classes, task,
                                       builder.tensor_entries)
    ensemble = _make_ensemble(trees, tree_group, task, _base_score(estimator, booster))
    builder.add_node(omle.Node(
        name="tree_ensemble",
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=_node_inputs(input_name),
        outputs=make_node_outputs(task),
        tree_ensemble=ensemble,
    ))


def _convert_lgb(
    estimator,
    input_name,
    task: TaskType,
    n_classes: int,
    builder: Builder,
) -> None:
    """Convert an LGBMClassifier or LGBMRegressor node into the builder."""
    from omle_convert.lightgbm import (
        _convert_trees,
        _get_booster,
        _make_ensemble,
    )
    booster = _get_booster(estimator)
    model_dump = booster.dump_model()
    fn = (model_dump.get("feature_names")
          or [f"f{i}" for i in range(model_dump.get("max_feature_idx", -1) + 1)]
          or ["f0"])
    feat_index = make_feature_index(fn)
    trees, tree_group = _convert_trees(model_dump, feat_index, n_classes, task,
                                       builder.tensor_entries)
    ensemble = _make_ensemble(trees, tree_group, task)
    builder.add_node(omle.Node(
        name="tree_ensemble",
        domain="omle.ml",
        op="TreeEnsemble",
        inputs=_node_inputs(input_name),
        outputs=make_node_outputs(task),
        tree_ensemble=ensemble,
    ))
