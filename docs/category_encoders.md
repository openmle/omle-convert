# category_encoders — Full Support Reference

[category_encoders](https://contrib.scikit-learn.org/category_encoders/) is a
`scikit-learn-contrib` package of target-statistic and contrast encoders. Nine of
its transformers convert, inside a `Pipeline` or a `ColumnTransformer`.

It is **not** a dependency of omle-convert. The package is imported only when a
model actually contains one of its transformers, so `pip install omle-convert`
stays free of it.

## Supported encoders

| Class | OMLE op | Extraction |
|---|---|---|
| `TargetEncoder` | `TargetEncoder` | Categories, encoded values and per-column default as tensor refs |
| `MEstimateEncoder` | `OrdinalEncoder` (with values) | Ordinal-code-keyed mapping |
| `WOEEncoder` | `OrdinalEncoder` (with values) | Ordinal-code-keyed mapping |
| `JamesSteinEncoder` | `OrdinalEncoder` (with values) | Ordinal-code-keyed mapping |
| `QuantileEncoder` | `OrdinalEncoder` (with values) | Ordinal-code-keyed mapping |
| `SummaryEncoder` | `OrdinalEncoder` (with values) | Ordinal-code-keyed mapping |
| `CountEncoder` | `OrdinalEncoder` (with values) | String-keyed mapping |
| `CatBoostEncoder` | `OrdinalEncoder` (with values) | Ordered-statistics values from the fitted sum/count frame |
| `LeaveOneOutEncoder` | `OrdinalEncoder` (with values) | Per-category means from the fitted sum/count frame |

## Why they collapse to one operator

Eight of the nine lower to a single `OrdinalEncoder` node carrying an explicit
value table. The encoders differ entirely in how they *derive* their numbers
during `fit` — m-estimate smoothing, weight of evidence, James-Stein shrinkage,
quantiles, ordered target statistics, leave-one-out means. At inference time each
one is the same thing: a fitted category → float lookup. Preserving the training
formula in the artifact would add operators that behave identically.

`TargetEncoder` is the exception only because OMLE already has a `TargetEncoder`
operator for sklearn's own transformer, and the runtime accepts string input
directly on that op — so no prior ordinal step is needed.

## Emitted node attributes

Both lowerings carry the same four tensors, flattened across columns:

| Attribute | dtype | Meaning |
|---|---|---|
| `categories` | string | Every category of every column, concatenated |
| `category_offsets` | int64 | Start index per column into `categories`; length `n_cols + 1` |
| `encoded_values` | float64 | The fitted value for each entry of `categories` |
| `default_values` | float64 | Per-column fallback for a category not seen during `fit` |

`TargetEncoder` additionally carries `target_kind`.

Unseen categories resolve to `default_values[col]`, which is taken from the
encoder's code `-1` mapping where it has one and otherwise from the global target
mean (`enc._mean`). Categories that are `NaN`, or that carry a negative ordinal
code, are dropped during extraction rather than encoded.

## Extraction groups

The fitted state differs by encoder, so extraction runs down four paths:

| Group | Encoders | Fitted state read |
|---|---|---|
| Ordinal-code keyed | `MEstimateEncoder`, `WOEEncoder`, `JamesSteinEncoder`, `QuantileEncoder`, `SummaryEncoder` | `ordinal_encoder.category_mapping` (str → code) joined with `mapping[col]` (code → float) |
| String keyed | `CountEncoder` | `mapping[col]` indexed by the category string directly |
| Sum/count frame, CatBoost formula | `CatBoostEncoder` | `mapping[col]` as a sum/count DataFrame, ordered-statistics formula applied |
| Sum/count frame, leave-one-out formula | `LeaveOneOutEncoder` | Same frame, leave-one-out formula applied |

## Name collision with sklearn

`category_encoders.TargetEncoder` and `sklearn.preprocessing.TargetEncoder` share a
class name but are different transformers with different fitted attributes.
Dispatch is by `type(transformer).__module__`, never by class name, so both
convert correctly — including in the same pipeline.

The same rule has a consequence worth knowing: `category_encoders.OneHotEncoder`
and `category_encoders.OrdinalEncoder` are routed to the category_encoders path by
their module, *not* to the sklearn converters that share their names. They are not
among the nine supported classes, so they raise `NotImplementedError`. Use the
`sklearn.preprocessing` versions instead.

## Not supported

Every other upstream encoder raises `NotImplementedError` naming the supported set:

| Class | Note |
|---|---|
| `OneHotEncoder`, `OrdinalEncoder` | Use the `sklearn.preprocessing` equivalents, which are supported |
| `BinaryEncoder`, `BaseNEncoder`, `GrayEncoder`, `RankHotEncoder` | Multi-column code expansions with no single lookup table |
| `BackwardDifferenceEncoder`, `HelmertEncoder`, `PolynomialEncoder`, `SumEncoder` | Contrast codings; expand one column into a contrast matrix |
| `HashingEncoder` | Hashing is applied at transform time, not a fitted lookup |
| `GLMMEncoder` | Requires the fitted mixed-effects model, not reducible to a value table |
| `CountTargetEncoder`, `MultiHotEncoder` | Added in 2.11; not yet converted |

## Version notes

The supported set is stable across releases: the fitted attributes the converter
reads — `ordinal_encoder.category_mapping`, `mapping`, `_mean` — are unchanged
from 2.6 through 2.11. 2.11 added `CountTargetEncoder` and `MultiHotEncoder`,
which are not converted.

The series boundaries are set by the interpreter, not by the package:

| Python | Newest installable | Note |
|---|---|---|
| 3.10 | 2.8.1 | 2.9+ requires 3.11 |
| 3.11+ | 2.11.x | |

CI pins one release per row — 2.8.1, 2.9.0, 2.10.0, 2.11.x — and leaves the
Python 3.14 row unpinned so a new release is exercised as soon as it ships.
