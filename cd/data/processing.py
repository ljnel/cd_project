"""Pure transformations on Dataset / dict-of-Dataset."""

from sklearn.preprocessing import StandardScaler


def normalize_channels(splits, fit_on='train'):
    """Per-channel z-score across splits, fit on `splits[fit_on]`.

    Operates on dict[str, Dataset]; returns a new dict with each Dataset's
    `X` field rescaled.
    """
    X_fit = splits[fit_on].X
    D = X_fit.shape[-1]
    scaler = StandardScaler().fit(X_fit.reshape(-1, D))

    def apply(X):
        return scaler.transform(X.reshape(-1, D)).reshape(X.shape)

    return {k: s.map(X=apply) for k, s in splits.items()}
