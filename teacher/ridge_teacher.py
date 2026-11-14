"""History-adaptive base forecaster of the STDF statistical teacher.

Channels are split, in their original order, into consecutive groups of at most
``series_group`` variables and the horizon into contiguous ``forecast_group``
step blocks.  Every (group, block) unit owns a lookback, a local-statistics
window and a Ridge penalty; windows of all channels in a group are pooled to
fit one shared multi-output map (Eq. 1 of the paper).

Two fitting protocols are provided:

* ``fit_teacher``            full data. The configuration of each unit is
  searched with TPE by three-fold expanding-window validation *inside* the
  fitting interval and then refitted on the whole interval.
* ``fit_teacher_few_shot``   few-shot. The prefix is too short for internal
  folds, so each unit is selected by the official validation split.
"""
import random

import numpy as np
import torch

from teacher import searchcast_ridge as sc


def _alphas(device):
    return torch.logspace(-6, 4, 21, device=device, dtype=torch.float64)


def expanding_boundaries(fit_end, n_folds):
    bounds = [int(fit_end * (i + 1) // (n_folds + 1)) for i in range(n_folds + 1)]
    if min(np.diff([0] + bounds)) < 2:
        raise ValueError('Fitting interval is too short for the requested folds')
    return bounds


def plan_units(channels, horizon, fit_end, series_group=7, forecast_group=24,
               min_lookback=32, max_lookback=2048, n_folds=3):
    """Enumerate (variable group, forecast block) units and their lookback range."""
    bounds = expanding_boundaries(fit_end, n_folds)
    units = []
    for c0 in range(0, channels, series_group):
        for h0 in range(0, horizon, forecast_group):
            h1 = min(h0 + forecast_group, horizon)
            # every fold needs one training window and one complete label block
            upper = min(max_lookback, bounds[0] - h1)
            if upper < min_lookback:
                raise ValueError(f'First fold ends at {bounds[0]}: too short for H={h1}, L>={min_lookback}')
            if any(b - a < h1 for a, b in zip(bounds[:-1], bounds[1:])):
                raise ValueError(f'A validation fold is shorter than H={h1}')
            units.append({'series': list(range(c0, min(channels, c0 + series_group))),
                          'h_start': h0, 'h_end': h1,
                          'min_lookback': min_lookback, 'max_lookback': upper})
    return bounds, units


class _StrictObjective(sc.SingleObjectiveWrapper):
    """Every candidate must be scored on the same folds."""

    def _evaluate_fold(self, *args, **kwargs):
        result = super()._evaluate_fold(*args, **kwargs)
        if result is None or not torch.isfinite(result).all():
            raise sc.optuna.TrialPruned()
        return result


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_teacher(values, fit_end, horizon, device='cpu', n_trials=20, n_folds=3,
                series_group=7, forecast_group=24, min_lookback=32, max_lookback=2048,
                seed=2021, log=print):
    """Search every unit inside ``values[:fit_end]`` and refit it on that interval.

    ``values`` is [T, C] in the coordinates the forecasts should live in.
    Nothing at or after ``fit_end`` is read.
    """
    device = torch.device(device)
    bounds, units = plan_units(values.shape[1], horizon, fit_end, series_group, forecast_group,
                               min_lookback, max_lookback, n_folds)
    train = torch.from_numpy(np.asarray(values[:fit_end], dtype=np.float32).copy())
    alphas = _alphas(device)
    cells, weights = [], {}
    for index, unit in enumerate(units):
        unit_seed = int(seed) + index
        _seed_all(unit_seed)
        steps = list(range(unit['h_start'] + 1, unit['h_end'] + 1))
        objective = _StrictObjective(
            data=train, series_idx=unit['series'],
            lookbacks=[unit['min_lookback'], unit['max_lookback']],
            horizon=steps, alphas=alphas, device=device,
            train_ratio=bounds[0] / fit_end, test_ratio=0.0, n_folds=n_folds,
            scaler_scope='local', scaler_method='mean', pool_series=True)
        objective.fold_boundaries = bounds
        objective.split_idx_1, objective.split_idx_2 = bounds[0], bounds[-1]
        study = sc.optuna.create_study(direction='minimize',
                                       sampler=sc.optuna.samplers.TPESampler(seed=unit_seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_trial
        lookback = int(best.params['lookback'])
        local_ratio = float(best.params.get('local_ratio', 1.0))
        alpha = float(best.user_attrs['best_alpha'])
        noise = best.params.get('noise_type', 'none')
        aug = {'noise_type': noise, 'sigma': float(best.params.get('aug_sigma', 0.0)) if noise != 'none' else 0.0}
        # refit on the complete fitting interval, pooled over the group's channels
        x, y = sc.get_context_and_horizons(train[:, unit['series']].transpose(0, 1), lookback, steps)
        scaler = sc.LocalNormScaler(sc.StandardStrategy(), lookback, max(1, int(lookback * local_ratio)))
        theta = sc.RidgeSolver(device).solve(
            x.reshape(-1, lookback), y.reshape(-1, len(steps)),
            torch.tensor([alpha], device=device, dtype=torch.float64), scaler=scaler, aug_config=aug)
        theta = theta.detach().cpu().numpy()
        if not np.isfinite(theta).all():
            raise RuntimeError('Non-finite Ridge weights')
        key = f'cell_{index:05d}'
        weights[key] = theta
        cells.append({**unit, 'weight_key': key, 'lookback': lookback, 'alpha': alpha,
                      'local_ratio': local_ratio, 'augmentation': aug, 'cv_mse': float(best.value)})
        log(f'  ridge unit {index + 1}/{len(units)} C={unit["series"][0]}..{unit["series"][-1]} '
            f'H={steps[0]}..{steps[-1]} L={lookback} alpha={alpha:.4g} cv_mse={best.value:.6f}')
    return RidgeTeacher(cells, weights, values.shape[1], horizon, device)


def _pooled_contexts(values, origins, lookback):
    x = np.asarray(values)[origins[:, None] + np.arange(-lookback, 0)[None, :]]
    return np.ascontiguousarray(x.transpose(2, 0, 1).reshape(-1, lookback), dtype=np.float32)


def _pooled_targets(values, origins, first, end):
    y = np.asarray(values)[origins[:, None] + np.arange(first, end)[None, :]]
    return np.ascontiguousarray(y.transpose(2, 0, 1).reshape(-1, end - first), dtype=np.float32)


def _predict_path(values, origins, theta, lookback, local_ratio, device, batch_size):
    """[K, F, h] weights -> [K, N, h, C] forecasts."""
    channels, width = values.shape[1], theta.shape[-1]
    out = np.empty((len(theta), len(origins), width, channels), np.float32)
    solver = sc.RidgeSolver(device)
    for start in range(0, len(origins), batch_size):
        current = origins[start:start + batch_size]
        scaler = sc.LocalNormScaler(sc.StandardStrategy(), lookback, max(1, int(lookback * local_ratio)))
        pred = solver.predict(torch.from_numpy(_pooled_contexts(values, current, lookback)),
                              torch.from_numpy(np.asarray(theta)), scaler=scaler,
                              chunk_size=max(1, batch_size * channels))
        pred = pred.detach().cpu().numpy().reshape(len(theta), channels, len(current), width)
        out[:, start:start + len(current)] = pred.transpose(0, 2, 3, 1)
    return out


def fit_teacher_few_shot(values, fit_end, train_origins, val_origins, horizon, score_ratio=None,
                         device='cpu', n_trials=20, series_group=7, forecast_group=24,
                         min_lookback=32, max_lookback=2048, min_fit_origins=16, seed=2021,
                         batch_size=256, log=print):
    """Few-shot base forecaster: units are selected on the official validation split.

    Fitting reads ``values[:fit_end]`` only.  ``score_ratio`` ([C]) rescales
    validation errors to the reporting units; it never touches model inputs.
    """
    device = torch.device(device)
    channels = values.shape[1]
    ratio = np.ones(channels) if score_ratio is None else np.asarray(score_ratio, np.float64)
    last = int(train_origins[-1])
    cap = min(max_lookback, last - min_fit_origins + 1)
    if cap < min_lookback:
        raise ValueError('Few-shot prefix is too short for the statistical teacher')
    alphas = np.logspace(-6, 4, 21)
    alpha_t = torch.tensor(alphas, dtype=torch.float64, device=device)
    cells, weights = [], {}
    for c0 in range(0, channels, series_group):
        group = list(range(c0, min(c0 + series_group, channels)))
        prefix = np.asarray(values[:fit_end, group], dtype=np.float32).copy()
        val_values = np.asarray(values[:, group])
        for h0 in range(0, horizon, forecast_group):
            h1 = min(h0 + forecast_group, horizon)
            target = val_values[val_origins[:, None] + np.arange(h0, h1)[None, :]]
            study = sc.optuna.create_study(direction='minimize', sampler=sc.optuna.samplers.TPESampler(
                seed=seed + h0 + c0 * horizon))
            study.enqueue_trial({'lookback': min(96, cap), 'local_ratio': 1.0})
            best = {}

            def objective(trial):
                lookback = trial.suggest_int('lookback', min_lookback, cap)
                local_ratio = trial.suggest_float('local_ratio', 1e-3, 1.0, log=True)
                fit = train_origins[train_origins >= lookback]
                scaler = sc.LocalNormScaler(sc.StandardStrategy(), lookback, max(1, int(lookback * local_ratio)))
                theta = sc.RidgeSolver(device).solve(
                    torch.from_numpy(_pooled_contexts(prefix, fit, lookback)),
                    torch.from_numpy(_pooled_targets(prefix, fit, h0, h1)), alpha_t, scaler=scaler,
                    aug_config={'noise_type': 'none', 'sigma': 0.0}).detach().cpu().numpy()
                squares, count = np.zeros(len(theta)), 0
                for start in range(0, len(val_origins), batch_size):
                    current = val_origins[start:start + batch_size]
                    pred = _predict_path(val_values, current, theta, lookback, local_ratio, device, batch_size)
                    error = (pred.astype(np.float64) - target[start:start + len(current)][None]) * ratio[group]
                    squares += np.square(error).sum(axis=(1, 2, 3))
                    count += error[0].size
                scores = squares / count
                k = int(np.argmin(scores))
                if not best or scores[k] < best['val_mse']:
                    best.update(val_mse=float(scores[k]), lookback=lookback, local_ratio=local_ratio,
                                alpha=float(alphas[k]), theta=theta[k:k + 1].copy())
                return float(scores[k])

            study.optimize(objective, n_trials=n_trials)
            key = f'cell_{len(cells):05d}'
            weights[key] = best.pop('theta')
            cells.append({'series': group, 'h_start': h0, 'h_end': h1, 'weight_key': key, **best})
            log(f'  ridge unit C={group[0]}..{group[-1]} H={h0 + 1}..{h1} L={best["lookback"]} '
                f'alpha={best["alpha"]:.4g} val_mse={best["val_mse"]:.6f}')
    return RidgeTeacher(cells, weights, channels, horizon, device)


class RidgeTeacher:
    """A frozen collection of fitted (group, block) Ridge units."""

    def __init__(self, cells, weights, channels, horizon, device='cpu'):
        self.cells, self.weights = cells, weights
        self.channels, self.horizon = channels, horizon
        self.device = torch.device(device)
        self.min_origin = max(int(cell['lookback']) for cell in cells)
        self.solver = sc.RidgeSolver(self.device)

    def predict(self, values, origins, batch_size=256):
        """Forecast [N, H, C]; only values strictly before each origin are read."""
        origins = np.asarray(origins, dtype=np.int64)
        if len(origins) and (origins.min() < self.min_origin or origins.max() > len(values)):
            raise ValueError('Origin lacks teacher history or exceeds the data')
        out = np.empty((len(origins), self.horizon, self.channels), np.float32)
        for start in range(0, len(origins), batch_size):
            current = origins[start:start + batch_size]
            for cell in self.cells:
                lookback = int(cell['lookback'])
                index = current[:, None] + np.arange(-lookback, 0)[None, :]
                x = np.asarray(values[index])[:, :, cell['series']].transpose(2, 0, 1)
                x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(-1, lookback)
                scaler = sc.LocalNormScaler(sc.StandardStrategy(), lookback,
                                            max(1, int(lookback * cell['local_ratio'])))
                pred = self.solver.predict(x, torch.from_numpy(self.weights[cell['weight_key']]),
                                           scaler=scaler, chunk_size=batch_size)
                width = cell['h_end'] - cell['h_start']
                pred = pred.detach().cpu().numpy().reshape(len(cell['series']), len(current), width)
                for local_c, global_c in enumerate(cell['series']):
                    out[start:start + len(current), cell['h_start']:cell['h_end'], global_c] = pred[local_c]
        if not np.isfinite(out).all():
            raise RuntimeError('Non-finite teacher forecast')
        return out

    def save(self, path, **extra):
        np.savez(path, **self.weights)
        return {'cells': self.cells, 'channels': self.channels, 'horizon': self.horizon,
                'teacher_min_origin': self.min_origin, **extra}
