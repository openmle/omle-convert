# sklearn — Full Support Reference

## Supported estimators

| Class | Task | OMLE Op | Notes |
|---|---|---|---|
| `DecisionTreeClassifier` | Binary, Multiclass | `Tree` | Single tree; multiclass emits one tree per class |
| `DecisionTreeRegressor` | Regression | `Tree` | Single tree |
| `RandomForestClassifier` | Binary, Multiclass | `TreeEnsemble` | One tree per estimator; multiclass expands to n_classes sub-trees |
| `RandomForestRegressor` | Regression | `TreeEnsemble` | |
| `ExtraTreesClassifier` | Binary, Multiclass | `TreeEnsemble` | Same expansion as RandomForest |
| `ExtraTreesRegressor` | Regression | `TreeEnsemble` | |
| `GradientBoostingClassifier` | Binary, Multiclass | `TreeEnsemble` | Leaf values scaled by `learning_rate`; base score inferred from init; multiclass per-class base scores embedded as single-leaf bias trees |
| `GradientBoostingRegressor` | Regression | `TreeEnsemble` | |
| `HistGradientBoostingClassifier` | Binary, Multiclass | `TreeEnsemble` | Uses `_predictors` / `_baseline_prediction` internals; multiclass per-class baselines embedded as single-leaf bias trees |
| `HistGradientBoostingRegressor` | Regression | `TreeEnsemble` | |
| `SVC` | Binary, Multiclass | `SVM` (KernelSVM) | Always OVO for multiclass; sparse support vectors supported; `probability=True` stores Platt scaling params (`probA_`/`probB_`) in `KernelSVM.prob_a`/`prob_b`; binary Platt probability matches sklearn within ~0.003 (libsvm iterative algorithm); multiclass OVO Platt combination not yet implemented |
| `SVR` | Regression | `SVM` (KernelSVM) | |
| `NuSVC` | Binary, Multiclass | `SVM` (KernelSVM) | `probability=True` stores Platt scaling params; binary Platt implemented |
| `NuSVR` | Regression | `SVM` (KernelSVM) | |
| `LinearSVC` | Binary, Multiclass | `SVM` (LinearSVM) | |
| `LinearSVR` | Regression | `SVM` (LinearSVM) | |
| `MLPClassifier` | Binary, Multiclass | `NeuralNetwork` | Each layer → `DenseLayer`; weights transposed to `[out, in]`; activation inferred per layer; `y_pred`=class label, `y_prob`=full probability matrix |
| `MLPRegressor` | Regression | `NeuralNetwork` | Weights transposed to `[out, in]` |
| `KNeighborsClassifier` | Binary, Multiclass | `KNN` | `neighbor_mode=k_neighbors`; sparse `_fit_X` stored as `SparseTensor` when scipy available |
| `KNeighborsRegressor` | Regression | `KNN` | `neighbor_mode=k_neighbors` |
| `RadiusNeighborsClassifier` | Binary, Multiclass | `KNN` | `neighbor_mode=radius`; `outlier_label` stored as tensor ref when set |
| `RadiusNeighborsRegressor` | Regression | `KNN` | `neighbor_mode=radius` |
| `LinearRegression` | Regression | `Linear` | |
| `Ridge` | Regression | `Linear` | |
| `Lasso` | Regression | `Linear` | |
| `ElasticNet` | Regression | `Linear` | |
| `Lars` / `LassoLars` | Regression | `Linear` | |
| `BayesianRidge` | Regression | `Linear` | Exports posterior covariance (`weight_covariance`) and noise precision (`noise_precision`) |
| `ARDRegression` | Regression | `Linear` | Same Bayesian fields as `BayesianRidge` |
| `MultiTaskLasso` / `MultiTaskElasticNet` | Regression (multi-output) | `Linear` | 2-D `coef_` stored in sklearn's native `(n_targets, n_features)` layout |
| `HuberRegressor` / `TheilSenRegressor` | Regression | `Linear` | |
| `QuantileRegressor` | Regression | `Linear` | |
| `OrthogonalMatchingPursuit` | Regression | `Linear` | |
| `SGDRegressor` / `PassiveAggressiveRegressor` | Regression | `Linear` | |
| `LogisticRegression` / `LogisticRegressionCV` | Binary, Multiclass | `Linear` | Binary → SIGMOID post-transform; multiclass → SOFTMAX |
| `RidgeClassifier` / `RidgeClassifierCV` | Binary, Multiclass | `Linear` | |
| `SGDClassifier` / `PassiveAggressiveClassifier` / `Perceptron` | Binary, Multiclass | `Linear` | |
| `KMeans` | Clustering | `Clustering` (prototype) | `cluster_centers_`; Euclidean distance |
| `MiniBatchKMeans` | Clustering | `Clustering` (prototype) | |
| `GaussianMixture` | Clustering | `Clustering` (gaussian_mixture) | Supports `full`, `tied`, `diag`, `spherical` covariance types |
| `GaussianNB` | Binary, Multiclass | `NaiveBayes` (gaussian) | Stores `theta_` / `var_`; `var_smoothing` as `variance_epsilon` |
| `MultinomialNB` | Binary, Multiclass | `NaiveBayes` (multinomial) | |
| `ComplementNB` | Binary, Multiclass | `NaiveBayes` (multinomial) | Same log-prob structure as MultinomialNB |
| `BernoulliNB` | Binary, Multiclass | `NaiveBayes` (bernoulli) | `binarize` threshold preserved |
| `CategoricalNB` | Binary, Multiclass | `NaiveBayes` (categorical) | Per-feature category arrays concatenated with offset/count index |
| `VotingClassifier` (`voting='soft'`) | Binary, Multiclass | child nodes + `SoftVote` | Each estimator exported as its own node; `SoftVote` averages `y_prob`; optional `weights` attr |
| `VotingClassifier` (`voting='hard'`) | Binary, Multiclass | child nodes + `MajorityVote` / `WeightedMajorityVote` | Hard majority vote on `y_pred` outputs |
| `VotingRegressor` | Regression | child nodes + `Average` / `WeightedAverage` | Averages `y_pred` outputs |
| `StackingClassifier` | Binary, Multiclass | child nodes + `Concat` + final estimator | Base `y_prob` tensors (or `y_pred` for `predict` method) concatenated; `passthrough=True` prepends raw `X` |
| `StackingRegressor` | Regression | child nodes + `Concat` + final estimator | Base `y_pred` scalars concatenated into meta-feature matrix |
| `BaggingClassifier` | Binary, Multiclass | child classifier nodes + `SoftVote` | Each estimator's `y_prob` averaged; optional `TakeSlots` per estimator when `max_features < 1.0` |
| `BaggingRegressor` | Regression | child regressor nodes + `Average` | Each estimator's `y_pred` averaged; optional `TakeSlots` per estimator when `max_features < 1.0` |
| `AdaBoostClassifier` | Binary, Multiclass | child classifier nodes + `SAMMEVote` / `WeightedMajorityVote` | When base estimators support `predict_proba`: `SAMMEVote` aggregates `y_pred` hard votes using the SAMME decision function (weighted sum + softmax) to match sklearn 1.6+ exactly; otherwise: `WeightedMajorityVote` on `y_pred`. `estimator_weights_` stored as `weights` attr; `n_classes` stored as int attr |
| `AdaBoostRegressor` | Regression | child regressor nodes + `WeightedMedian` | Each weak learner's `y_pred` aggregated via `WeightedMedian`; `estimator_weights_` stored as `weights` attr |
| `XGBClassifier` | Binary, Multiclass | `TreeEnsemble` | Delegates to the XGBoost converter; works anywhere a sklearn estimator can (Pipeline, Stacking, etc.); emits two `SourceFramework` entries (sklearn + xgboost) |
| `XGBRegressor` | Regression | `TreeEnsemble` | Same dispatch as `XGBClassifier` |
| `LGBMClassifier` | Binary, Multiclass | `TreeEnsemble` | Delegates to the LightGBM converter; works in any sklearn Pipeline or meta-estimator; emits two `SourceFramework` entries (sklearn + lightgbm) |
| `LGBMRegressor` | Regression | `TreeEnsemble` | Same dispatch as `LGBMClassifier` |
| `OneVsRestClassifier` | Binary, Multiclass | child binary nodes + `TakeSlots` + `SoftVote` | Each binary child's single positive-class probability column is extracted via `TakeSlots`; `SoftVote` stacks the columns and normalises to sum-1 per row, so no `Concat` is needed. Multiclass probabilities are close but not bit-exact to sklearn due to L1 vs softmax normalisation differences |
| `MultiOutputClassifier` | Binary, Multiclass | child classifier nodes + `Concat` | One classifier node per output; `y_pred` outputs concatenated |
| `MultiOutputRegressor` | Regression | child regressor nodes + `Concat` | One regressor node per output; `y_pred` outputs concatenated |
| `ClassifierChain` | Binary, Multiclass | sequential child nodes + `Concat` | Each link receives `Concat(X, prior y_pred)`; final outputs re-ordered to original column order |
| `RegressorChain` | Regression | sequential child nodes + `Concat` | Same chaining pattern as `ClassifierChain` |
| `IsolationForest` | Anomaly detection | `AnomalyDetection` (IsolationForest) | Each `ExtraTreeRegressor` converted to a `Tree`; leaf values precomputed as `depth + c(n_samples)` so the runtime averages path lengths directly; `max_samples_` and `offset_` stored |
| `LocalOutlierFactor` (`novelty=True`) | Anomaly detection | `AnomalyDetection` (LocalOutlierFactor) | Training data stored as `reference_samples`; `n_neighbors`, `metric`, `metric_params`, and `offset_` stored; requires `novelty=True` |
| `EllipticEnvelope` | Anomaly detection | `AnomalyDetection` (EllipticEnvelope) | `location_`, `covariance_`, `precision_`, and `offset_` stored |

## Feature dtypes

`Feature.type.dtype` is set based on the estimator's internal precision:

| Estimator category | `Feature.type.dtype` |
|---|---|
| `XGBClassifier` / `XGBRegressor` | `FLOAT32` (XGBoost converts all inputs to float32 internally) |
| All other sklearn estimators (trees, forests, GB, HGB, linear, SVM, MLP, KNN, NB, clustering) | `FLOAT64` (sklearn stores thresholds and leaf values as float64) |

## Per-column inputs (DataFrame)

When `X` is a pandas DataFrame and the model has no preprocessing pipeline, `from_sklearn` generates **per-column `InputSpec`s** (one per DataFrame column) instead of a single `X` matrix spec. The estimator node receives one `NodeInput` per column so the runtime's `gather_slots` concatenates them automatically. Pipelines with preprocessing steps always fall back to a single `X` tensor input.

## Graph shape

Most operators — preprocessing, estimators, and meta-estimators like `StackingClassifier` — accept multiple inputs directly, so no intermediate `Concat` node is emitted. `StackingClassifier` base-estimator outputs feed straight into the final estimator, including the `passthrough=True` case where raw features are appended.

`XGBClassifier` and `LGBMClassifier` inside a `Pipeline` are handled transparently by delegation to the xgboost and lightgbm converters.

When a `ColumnTransformer` selects columns by string name (i.e. the pipeline was fitted on a pandas DataFrame), the converter emits `TakeSlots` with a `names` attribute (e.g. `names=["feat_a", "feat_b"]`) instead of integer `indices`. The runtime resolves names to indices at model load time using the model schema, so the graph stays self-describing while execution remains index-based. Integer-indexed `ColumnTransformer` groups emit `TakeSlots` with `indices` as before.

### `SoftVote` column-stacking

When every input to `SoftVote` is an `(n, 1)` column tensor, the operator stacks
them column-wise into an `(n, k)` matrix instead of averaging. `OneVsRestClassifier`
relies on this: each binary child's positive-class column flows straight into
`SoftVote`, so no `Concat` node is emitted. `normalize_rows=true` then applies the
row-wise L1 normalization that matches sklearn's OVR `predict_proba`.

Binary OVR (a single child estimator) is the exception — the child already emits a
normalized `(n, 2)` distribution, so its `y_prob` is passed through whole and
`normalize_rows` is omitted.

## Output type annotations

Every node output — preprocessing, estimator, and aggregation nodes such as `MajorityVote`, `Average` and `WeightedMedian` — carries a `TensorType` with `dtype` and `shape`:

| Output | dtype | shape |
|---|---|---|
| Preprocessing (`StandardScaler`, `PCA`, `OHE`, …) | `FLOAT64` (`BOOL` for `MissingIndicator`) | `[-1, n_features_out]` |
| Estimator `y_pred` (classifier) | `INT64` | `[-1]` |
| Estimator `y_pred` (regressor) | `FLOAT64` | `[-1]` |
| Estimator `y_prob` | `FLOAT64` | `[-1, n_classes]` |
| Clustering `cluster_id` | `INT32` | `[-1]` |
| Anomaly `anomaly_score` | `FLOAT64` | `[-1]` |

These match the model-level `OutputSpec` types. The test suite verifies the contract across every supported estimator class, preprocessing pipeline, ensemble method and meta-estimator via the `_assert_std_io` helper.

## Preprocessing (inside `Pipeline` or `ColumnTransformer`)

| Class | OMLE Op | Notes |
|---|---|---|
| `StandardScaler` | `StandardScaler` | Stores `mean_`, `scale_` tensors |
| `MinMaxScaler` | `MinMaxScaler` | Stores `data_min_`, `data_max_` (= `data_range_`); `feature_range` as float attrs |
| `RobustScaler` | `RobustScaler` | Falls back to zero center / unit scale when `with_centering/scaling=False` |
| `MaxAbsScaler` | `MaxAbsScaler` | |
| `Normalizer` | `Normalizer` | `norm` attribute stored as string (`l1`, `l2`, `max`) |
| `SimpleImputer` | `Imputer` | `statistics_` stored as `fill_tensor` |
| `Binarizer` | `Binarizer` | `threshold` stored as float attr |
| `PCA` | `Linear` | `components_` as `(n_components, n_features)` coefficient tensor; intercept `= −mean_ @ components_.T` |
| `OneHotEncoder` | `OneHotEncoder` + `Concat` | One node per feature column when `n_features > 1` |
| `OrdinalEncoder` | `OrdinalEncoder` + `Concat` | One node per feature column when `n_features > 1`; categories stored as tensor entries (string + INT64 offsets) |
| `ColumnTransformer` | `TakeSlots` + sub-transformer + `Concat` | Recurses into each named sub-transformer; `"drop"` groups skipped |
| `Pipeline` (as transformer) | chained sub-transformer nodes | All steps converted as transformers; used inside `ColumnTransformer` or `FeatureUnion` |
| `FeatureUnion` | parallel sub-transformer nodes + `Concat` | Each branch transformer receives the same input; outputs concatenated; `transformer_weights` not applied |
| `FunctionTransformer` (passthrough) | — | Returns input unchanged when `func=None` |
| `FunctionTransformer` (numpy func) | `Derive` + optional `Concat` | See supported functions below |
| `LabelEncoder` | `LabelEncoder` | `classes_` stored as tensor entries (string labels + INT64 offsets) |
| `LabelBinarizer` | `LabelBinarizer` | `classes_` as string tensor; `neg_label`, `pos_label` as int attrs |
| `KBinsDiscretizer` | `Bucketizer` + `OneHotEncoder` + optional `Concat` + optional `DenseToSparse` | `ordinal`: `Bucketizer` per feature; `onehot-dense`: `Bucketizer` → `OneHotEncoder` per feature + `Concat`; `onehot`: same + `DenseToSparse` |
| `TargetEncoder` | `TargetEncoder` + optional `Concat` | sklearn ≥ 1.3; categories, encoded values, and default value as tensor refs; one node per feature column when `n_features > 1` |
| `PowerTransformer` | `PowerTransformer` | `method` (`box_cox`/`yeo_johnson`), `lambdas_` as tensor ref; when `standardize=True` also stores `mean_` and `scale_` from the inner scaler |
| `QuantileTransformer` | `QuantileTransformer` | `quantiles_` [Q, D] and `references_` [Q] as tensor refs; `output_distribution` as string attr |
| `SplineTransformer` | `SplineTransformer` | Full augmented knot vectors for all features stored as a [n_aug, n_features] tensor ref; `degree`, `include_bias`, `extrapolation` as attrs |
| `PolynomialFeatures` | `PolynomialFeatures` | `min_degree`, `max_degree`, `interaction_only`, `include_bias` as attrs; `powers_` matrix stored as INT64 tensor ref for exact sklearn feature ordering |
| `MultiLabelBinarizer` | `MultiLabelBinarizer` | `classes_` as string tensor; input must be a collection-valued (label-set) column |
| `TruncatedSVD` | `TruncatedSVD` | `components_` [K, D] as tensor ref |
| `FastICA` | `FastICA` | `components_` as tensor ref; when `whiten=True` also stores `mean_` and `whitening_` |
| `FactorAnalysis` | `FactorAnalysis` | `mean_`, `components_`, and `noise_variance_` as tensor refs |
| `KernelPCA` | `KernelPCA` | `X_fit_` as `fit_samples`; `eigenvectors_ / sqrt(eigenvalues_)` as `dual_components`; `gamma_`, `degree`, `coef0` as attrs; kernel centerer stats stored when present |
| `SparsePCA` | `SparsePCA` | `components_` as tensor ref; `mean_` included when present |
| `NMF` | `NMF` | `components_` [K, D] as tensor ref |
| `LatentDirichletAllocation` | `LatentDirichletAllocation` | `components_` as tensor ref; `doc_topic_prior_` and `topic_word_prior_` as float attrs |
| `MissingIndicator` | `MissingIndicator` | `features_mode` attr; `feature_indices` INT tensor ref when `features='missing-only'`; `error_on_new` bool attr |
| `KNNImputer` | `KNNImputer` | `_fit_X` as `train_features` tensor ref; `n_neighbors`, `weights`, `metric`, `keep_empty_features` as attrs |

### `category_encoders` package

Nine transformers from the third-party
[category_encoders](https://contrib.scikit-learn.org/category_encoders/) package
convert inside a `Pipeline` or `ColumnTransformer`: `TargetEncoder`,
`MEstimateEncoder`, `WOEEncoder`, `JamesSteinEncoder`, `QuantileEncoder`,
`SummaryEncoder`, `CountEncoder`, `CatBoostEncoder` and `LeaveOneOutEncoder`. The
package is not a dependency — it is imported only when a model uses one.

See [category_encoders.md](category_encoders.md) for the lowering of each class,
the emitted node attributes, the sklearn `TargetEncoder` name collision, and what
is not supported.

### Supported numpy functions for `FunctionTransformer`

| `func` | `omle.functions` expression |
|---|---|
| `np.log` | `log(x)` |
| `np.log1p` | `ln1p(x)` |
| `np.log2` | `log2(x)` |
| `np.log10` | `log10(x)` |
| `np.exp` | `exp(x)` |
| `np.sqrt` | `sqrt(x)` |
| `np.abs` / `np.absolute` | `abs(x)` |
| `np.negative` | `neg(x)` |
| `np.floor` | `floor(x)` |
| `np.ceil` | `ceil(x)` |
| `np.sign` | `sign(x)` |
| `np.square` | `pow(x, 2)` |
| `np.cbrt` | `pow(x, 1/3)` |

Multi-feature transforms use `TakeSlots` per column → `Derive` → `Concat`.

## Feature selection (inside `Pipeline` or `ColumnTransformer`)

All sklearn feature selectors are converted to a single `TakeSlots` node (`omle.core`) using `get_support(indices=True)` to determine which feature columns to keep.

| Class | Notes |
|---|---|
| `SelectKBest` | Score function (e.g. `f_classif`, `chi2`) applied at fit time; top-k indices stored |
| `SelectPercentile` | Same as `SelectKBest`; percentile threshold applied at fit time |
| `SelectFdr` | False discovery rate selection; indices fixed after fit |
| `SelectFpr` | False positive rate selection |
| `SelectFwe` | Family-wise error rate selection |
| `VarianceThreshold` | Columns below the variance threshold are dropped |
| `SelectFromModel` | Uses a fitted estimator's `feature_importances_` or `coef_`; supports `prefit=True` |
| `RFE` | Recursive feature elimination; selected indices from `support_` |
| `RFECV` | RFE with cross-validation; selected indices from `support_` |
| `SequentialFeatureSelector` | Forward or backward stepwise selection |

## Text transformers (inside `Pipeline` or `ColumnTransformer`)

All text transformers emit nodes in the `omle.text` domain.

| Class | Emitted node chain | Notes |
|---|---|---|
| `CountVectorizer` | `[RegexTokenizer]` → `[StopWordsRemover]` → `[NGram]` → `CountVectorizer` | `RegexTokenizer` skipped for `analyzer='char'`/`'char_wb'`; `StopWordsRemover` skipped when `stop_words` not set; `NGram` skipped when `ngram_range=(1,1)` |
| `TfidfTransformer` | `TfIdfTransformer` | IDF weights stored as a float64 tensor ref |
| `TfidfVectorizer` | `[RegexTokenizer]` → `[StopWordsRemover]` → `[NGram]` → `CountVectorizer` → `TfIdfTransformer` | Combines `CountVectorizer` chain with `TfIdfTransformer` in one converter |
| `HashingVectorizer` | `[RegexTokenizer]` → `[NGram]` → `HashingVectorizer` | No vocabulary tensor; `num_features`, `binary`, `alternate_sign` stored as attrs |

### Text node attributes

| Node | Key attributes |
|---|---|
| `RegexTokenizer` | `pattern` — the `token_pattern` from the vectorizer (default `(?u)\b\w\w+\b`); `gaps=true` |
| `StopWordsRemover` | `stop_words` — tensor ref to a STRING tensor of sorted stop word strings |
| `NGram` | `n_max` (int), `n_min` (int, omitted when 1) |
| `CountVectorizer` | `vocabulary` — tensor ref to a STRING tensor ordered by vocabulary index; `binary` (bool) |
| `TfIdfTransformer` | `idf` — tensor ref to a float64 tensor of per-term IDF weights |
| `HashingVectorizer` | `num_features` (int), `binary` (bool), `alternate_sign` (bool) |

## Not supported

### Hard unsupported — raise `NotImplementedError`

| Class / feature | Reason |
|---|---|
| `DBSCAN`, `AgglomerativeClustering`, `SpectralClustering`, other density/hierarchical clustering | No fixed cluster centers; cannot be represented as a `Clustering` prototype node |
| `OneVsOneClassifier`, `OutputCodeClassifier` | Require indexed pairwise voting operations not available in the current op set |
| `ColumnTransformer` with string column names | Column selection uses integer indices; pass `feature_names` to `from_sklearn` to work around |
| `FunctionTransformer` with non-numpy `func` | Only the numpy functions listed above are mapped; arbitrary callables cannot be serialised |

### Silent fallbacks / partial support

| Class / feature | Behaviour |
|---|---|
| `KNeighborsClassifier` / `KNeighborsRegressor` with unsupported distance metric | Falls back to `euclidean` silently |
| `KNeighborsClassifier` / `KNeighborsRegressor` with callable `weights` | Falls back to `'uniform'` silently |
| `GradientBoostingClassifier` / `GradientBoostingRegressor` with custom `init_` estimator | `base_score` extraction may fail for estimators other than `DummyClassifier` / `DummyRegressor` |
| `FeatureUnion` with `transformer_weights` | Weights are ignored; all branches are concatenated unweighted |
| `VotingClassifier` with dropped estimators | Dropped estimator weights are not renormalized |

### Estimators with no converter

| Category | Classes |
|---|---|
| Gaussian processes | `GaussianProcessClassifier`, `GaussianProcessRegressor` |
| Discriminant analysis | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |
| Calibration | `CalibratedClassifierCV` |
| Semi-supervised | `LabelPropagation`, `LabelSpreading` |
| Neighbors | `NearestCentroid` |

### Preprocessing transformers with no converter

| Category | Classes |
|---|---|
| Imputation | `IterativeImputer` |
