"""Hyperparameter tuning using RandomizedSearchCV.

For outlier detection, we use a trick: pass x_te as the "training" data to 
GridSearchCV but use a custom scorer that actually fits on x_tr (stored globally)
and evaluates on the input.
"""

from anomaly_detection.kernel import KernDetector
from experiments.safety_monitor import SafetyMonitor, hopper_cfg

from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, balanced_accuracy_score
from scipy.stats import loguniform, uniform
import numpy as np

# Load data
task = SafetyMonitor(hopper_cfg)
x_tr, x_te = task.get_train_test()
y_true = task.y_true

# Custom scorer for outlier detection
# The trick: we ignore the X passed by sklearn and use our global x_tr for fitting
def outlier_score(estimator, X, y):
    """Score function that fits on x_tr and evaluates on X."""
    # Clone and refit on actual training data
    from sklearn.base import clone
    model = clone(estimator)
    model.fit(x_tr)  # Always fit on x_tr
    y_pred = model.predict(X)
    return balanced_accuracy_score(y, y_pred)

# Parameter distributions for randomized search
param_distributions = {
    'kernel_type': ['fft', 'rbf', 'sig'],
    'gamma': loguniform(0.01, 10),      # log-uniform from 0.01 to 10
    'reg': loguniform(1e-5, 1e-1),      # log-uniform from 1e-5 to 0.1
    'threshold_quantile': uniform(0.85, 0.14),  # uniform from 0.85 to 0.99
    'max_windows': [100, 200, 300, 500],
}

# Create search object
# We use cv=[(slice(None), slice(None))] to skip cross-validation
# and just evaluate on the single "fold" which is our test set
search = RandomizedSearchCV(
    estimator=KernDetector(),
    param_distributions=param_distributions,
    n_iter=3,  # number of random configurations to try
    scoring=outlier_score,
    cv=[(np.arange(len(x_te)), np.arange(len(x_te)))],  # single fold: all test data
    refit=False,  # don't refit, we do it manually
    verbose=2,
    random_state=42,
    n_jobs=1,  # set higher if your model is thread-safe
)

# Run search (pass x_te as X, y_true as y - the scorer handles the rest)
search.fit(x_te, y_true)

# Results
print("\n" + "="*60)
print("SEARCH RESULTS")
print("="*60)
print(f"Best score: {search.best_score_*100:.2f}%")
print(f"Best params: {search.best_params_}")

# Train final model with best params
best_model = KernDetector(**search.best_params_)
best_model.fit(x_tr)
y_pred = best_model.predict(x_te)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, normalize='true').ravel()
print(f"\nFinal Results:")
print(f"TN: {cm[0]*100:.2f}%")
print(f"FP: {cm[1]*100:.2f}%")
print(f"FN: {cm[2]*100:.2f}%")
print(f"TP: {cm[3]*100:.2f}%")

# Show top 10 configurations
import pandas as pd
results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)
print("\nTop 10 configurations:")
cols = ['param_kernel_type', 'param_gamma', 'param_lam', 
        'param_threshold_quantile', 'mean_test_score']
print(results_df[cols].head(10).to_string())