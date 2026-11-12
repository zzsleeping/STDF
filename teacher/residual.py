"""Conditional residual correction of the base forecast (Eq. 2 of the paper).

Shallow CatBoost MultiRMSE models predict ``Y - B_oof`` for one variable group
and one 24-step block from features of the most recent 96 observations and of
the base forecast itself.  Every feature is available at the forecast origin.
"""
import numpy as np

LOOKBACK = 96
TREE_SPECS = (
    {'name': 'depth3', 'depth': 3, 'iterations': 300, 'learning_rate': 0.05, 'l2_leaf_reg': 5.0},
    {'name': 'depth4', 'depth': 4, 'iterations': 300, 'learning_rate': 0.05, 'l2_leaf_reg': 5.0},
)
ETAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)


def _slopes(array):
    """Least-squares change per step of [rows, time, channels]."""
    steps = np.arange(array.shape[1], dtype=np.float64)
    steps -= steps.mean()
    return np.einsum('ntc,t->nc', array, steps) / float(np.square(steps).sum())


def build_features(values, origins, base_block):
    """Past-only features phi(X^r, B) for one group and one forecast block.

    values: [T, c] observations of the group; origins: [N]; base_block: [N, h, c].
    Local levels, trends, lags and variability of the last 96 steps, plus the
    locally normalized base forecast and its summary statistics.
    """
    origins = np.asarray(origins, dtype=np.int64)
    forecast = np.asarray(base_block, dtype=np.float64)
    if origins.min() < LOOKBACK:
        raise ValueError('Residual features need 96 past observations')
    recent = np.asarray(values[origins[:, None] + np.arange(-LOOKBACK, 0)[None, :]], dtype=np.float64)
    center = recent.mean(axis=1)
    centered = recent - center[:, None, :]
    scale = np.sqrt(np.mean(np.square(centered), axis=1) + 1e-6)
    normalized = centered / scale[:, None, :]
    n, c = len(origins), recent.shape[2]
    pooled = normalized.reshape(n, 12, 8, c).mean(axis=2)
    last24 = normalized[:, -24:, :]
    stats = [center, scale, normalized[:, -1, :], _slopes(last24),
             last24[:, -1, :] - last24[:, 0, :], last24.std(axis=1)]
    nf = (forecast - center[:, None, :]) / scale[:, None, :]
    forecast_stats = [nf.mean(axis=1), nf.std(axis=1), nf[:, -1, :] - nf[:, 0, :]]
    features = np.concatenate(
        [pooled.reshape(n, -1), normalized[:, -8:, :].reshape(n, -1)] + stats
        + [nf.reshape(n, -1)] + forecast_stats, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Non-finite residual features')
    return features


def residual_targets(values, origins, first, end, base_block):
    """Y - B for forecast steps [first, end), flattened to [N, h*c]."""
    origins = np.asarray(origins, dtype=np.int64)
    target = np.asarray(values[origins[:, None] + np.arange(first, end)[None, :]], dtype=np.float64)
    return (target - np.asarray(base_block, dtype=np.float64)).reshape(len(origins), -1).astype(np.float32)


class _ConstantModel:
    """Residual mean; used when CatBoost cannot split (constant inputs/targets)."""

    def __init__(self, mean):
        self.mean = np.asarray(mean, dtype=np.float32)

    def predict(self, features):
        return np.broadcast_to(self.mean, (len(features), len(self.mean))).copy()


class _TreeModel:
    def __init__(self, model, targets):
        self.model, self.targets = model, targets

    def predict(self, features):
        out = np.asarray(self.model.predict(np.asarray(features, dtype=np.float32)), dtype=np.float32)
        return out.reshape(len(features), self.targets)


def fit_corrector(x, y, spec, seed=2021, device='cpu', cpu_threads=8):
    """Fixed-budget shallow trees on all block/channel outputs jointly (no early stopping)."""
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if bool(np.all(x == x[0])) or bool(np.all(y == y[0])):
        return _ConstantModel(y.mean(axis=0))
    from catboost import CatBoostRegressor
    cuda = str(device).startswith('cuda')
    params = {'loss_function': 'MultiRMSE', 'depth': int(spec['depth']),
              'iterations': int(spec['iterations']), 'learning_rate': float(spec['learning_rate']),
              'l2_leaf_reg': float(spec['l2_leaf_reg']), 'random_seed': int(seed),
              'thread_count': int(cpu_threads), 'verbose': False, 'allow_writing_files': False,
              'bootstrap_type': 'No', 'boosting_type': 'Plain', 'has_time': True,
              'task_type': 'GPU' if cuda else 'CPU'}
    if cuda:
        params['devices'] = str(device).split(':', 1)[1] if ':' in str(device) else '0'
    else:
        params['random_strength'] = 0.0
    model = CatBoostRegressor(**params)
    model.fit(x, y)
    return _TreeModel(model, y.shape[1])


class Scores:
    """Summed squared/absolute errors; disjoint blocks and groups add exactly."""

    def __init__(self):
        self.square, self.absolute, self.count = 0.0, 0.0, 0

    def update(self, prediction, target, ratio=None):
        delta = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        if ratio is not None:
            delta = delta * ratio
        self.square += float(np.square(delta).sum())
        self.absolute += float(np.abs(delta).sum())
        self.count += delta.size

    @property
    def mse(self):
        return self.square / self.count

    @property
    def mae(self):
        return self.absolute / self.count


def select_correction(candidates):
    """Pick (depth, eta) from forward-validation scores.

    candidates: {(name, eta): {fold: (base Scores, corrected Scores)}}.  A
    non-zero weight is eligible only if it improves both the mean relative MSE
    and the mean relative MAE over the base forecast; eta = 0 is the fallback.
    """
    rows = []
    for (name, eta), folds in candidates.items():
        mse = float(np.mean([c.mse / b.mse for b, c in folds.values()]))
        mae = float(np.mean([c.mae / b.mae for b, c in folds.values()]))
        rows.append({'model_name': name, 'eta': float(eta), 'mse_ratio': mse, 'mae_ratio': mae,
                     'score': (mse + mae) / 2, 'eligible': eta == 0 or (mse < 1 and mae < 1)})
    chosen = min((r for r in rows if r['eligible']), key=lambda r: (r['score'], r['eta'], r['model_name']))
    return {**chosen, 'candidates': rows}
