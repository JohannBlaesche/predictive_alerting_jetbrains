import numpy as np


def make_synthetic_series(n_steps=5000, seed=42):
    rng = np.random.default_rng(seed)

    t = np.arange(n_steps)

    # base signal
    x = 0.6 * np.sin(t / 30.0) + 0.3 * np.sin(t / 9.0) + 0.1 * rng.normal(size=n_steps)

    incident = np.zeros(n_steps, dtype=int)

    # inject a few "bad" regions
    starts = rng.integers(200, n_steps - 100, size=18)
    lengths = rng.integers(8, 30, size=len(starts))

    for s, length in zip(starts, lengths):
        e = min(s + length, n_steps)
        x[s:e] += rng.uniform(1.8, 2.8)
        incident[s:e] = 1

    return x, incident


def make_windows(series, incident, window_size=50, horizon=10):
    X = []
    y = []

    last_start = len(series) - window_size - horizon

    for start in range(last_start):
        end = start + window_size
        future_end = end + horizon

        x_window = series[start:end]
        future_incident = incident[end:future_end]

        label = 1 if np.any(future_incident > 0) else 0

        X.append(x_window)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def chronological_split(X, y, train_frac=0.6, val_frac=0.2):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test
