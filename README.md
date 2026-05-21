# fraud_model
End-to-end fraud model development pipeline

#### future experiments
- tabular transformer comparison (TabNet, FT-Transformer)
  to evaluate attention-based architectures against
  gradient boosting on fraud detection


###### rule-based model routing

rather than selecting a single best model, segment transactions
using hard business rules derived from eda findings and route
each segment to the model that performs best on it.

example: transactions with identity data (has_identity=1) may
be better scored by xgboost while transactions without identity
data are better scored by catboost. a simple routing rule:

if has_identity == 1:
    score = xgboost_model.predict(transaction)
else:
    score = catboost_model.predict(transaction)

natural segmentation candidates from this dataset:
- has_identity vs no identity (fraud rate 7.96% vs 2.10%)
- mobile vs desktop
- credit vs debit
- discover vs other networks
- high velocity cards vs low velocity

this approach is interpretable, stable, and defensible in mrm
review — unlike kmeans or other unsupervised segmentation methods
which produce segments that are hard to explain and may shift
with data drift. segment-level auc comparison would determine
whether routing produces meaningful lift over a simple weighted
ensemble.
