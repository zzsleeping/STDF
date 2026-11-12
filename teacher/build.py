"""Offline construction of the frozen statistical teacher and its guidance cache.

The teacher is fitted on the training interval only and frozen before student
training.  The student never executes it: it only reads

    train_predictions.npy   [N_train, H, C]  teacher forecasts T (float32)
    train_valid.npy         [N_train]        origins with sufficient teacher history

Full-data protocol
    1. base forecaster B^full: grouped Ridge searched and refitted on the
       training interval (teacher/ridge_teacher.py);
    2. chronological out-of-fold forecasts: Ridge teachers refitted on the first
       40%, 60% and 80% of the interval, each repeating the search and the
       normalization within its own prefix, forecast the following 20%;
    3. CatBoost correctors are fitted to the OOF residuals; tree depth and the
       weight eta are chosen by forward validation over the OOF intervals;
    4. T = B^full + eta * f(phi(X^r, B^full)).

Few-shot protocol
    The prefix is too short for OOF folds: the teacher is the base forecaster
    only (eta = 0), selected on the official validation split.
"""
import json
import os
import time

import numpy as np

from teacher import residual
from teacher.ridge_teacher import fit_teacher, fit_teacher_few_shot


def _prefix_scaler(raw):
    mean, var = raw.mean(0), raw.var(0)
    eps = np.finfo(np.float64).eps
    scale = np.sqrt(var)
    scale[var <= len(raw) * eps * var + (len(raw) * mean * eps) ** 2] = 1.0
    return mean, scale


def _blocks(horizon, width):
    return [(h0, min(h0 + width, horizon)) for h0 in range(0, horizon, width)]


def _groups(channels, width):
    return [(c0, min(c0 + width, channels)) for c0 in range(0, channels, width)]


def _targets(values, origins, first, end):
    return np.asarray(values[origins[:, None] + np.arange(first, end)[None, :]])


def _restrict(teacher, group):
    """The frozen Ridge units that cover one variable group."""
    first, end = group
    cells = []
    for cell in teacher.cells:
        series = [c - first for c in cell['series'] if first <= c < end]
        if series:
            cells.append({**cell, 'series': series})
    return type(teacher)(cells, teacher.weights, end - first, teacher.horizon, teacher.device)


def _identity(info, args):
    return {'data': args.data, 'data_path': args.data_path, 'seq_len': args.seq_len,
            'pred_len': args.pred_len, 'percent': args.percent, 'train_end': int(info['train_end']),
            'columns': info['columns'], 'mean': np.asarray(info['mean']).tolist(),
            'scale': np.asarray(info['scale']).tolist(), 'teacher_seed': args.teacher_seed,
            'teacher_trials': args.teacher_trials, 'teacher_max_lookback': args.teacher_max_lookback,
            'teacher_residual': int(args.teacher_residual and args.percent == 100)}


def teacher_dir(args):
    name = '{}_sl{}_pl{}'.format(os.path.splitext(args.data_path)[0], args.seq_len, args.pred_len)
    if args.percent != 100:
        name += '_p{}'.format(args.percent)
    return os.path.join(args.teacher_path, name)


def load_teacher_cache(info, args):
    """Return (predictions memmap, valid mask) or None if no matching cache exists."""
    folder = teacher_dir(args)
    meta_path = os.path.join(folder, 'metadata.json')
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta['identity'] != _identity(info, args):
        raise ValueError('Teacher cache {} was built with a different configuration; '
                         'remove it or change --teacher_path'.format(folder))
    predictions = np.load(os.path.join(folder, 'train_predictions.npy'), mmap_mode='r')
    valid = np.load(os.path.join(folder, 'train_valid.npy'))
    return predictions, valid


def _oof_fold(index, cut, end, info, args, folder, device, log):
    """Refit the Ridge teacher on raw[:cut] and forecast origins in [cut, end)."""
    raw, horizon = info['raw'], args.pred_len
    mean, scale = _prefix_scaler(raw[:cut])
    prefix = ((raw[:cut] - mean) / scale).astype(np.float32)
    log('OOF fold {}: fit on [0, {}), forecast [{}, {})'.format(index + 1, cut, cut, end))
    teacher = fit_teacher(prefix, cut, horizon, device=device, n_trials=args.teacher_trials,
                          max_lookback=args.teacher_max_lookback, seed=args.teacher_seed + index * 100000,
                          log=log)
    first = max(cut, residual.LOOKBACK, teacher.min_origin)
    origins = np.arange(first, end - horizon + 1, dtype=np.int64)
    values = ((raw[:end] - mean) / scale).astype(np.float32)
    predictions = np.lib.format.open_memmap(
        os.path.join(folder, 'oof_{}.npy'.format(index)), mode='w+', dtype=np.float32,
        shape=(len(origins), horizon, raw.shape[1]))
    for start in range(0, len(origins), args.teacher_batch_size):
        block = teacher.predict(values, origins[start:start + args.teacher_batch_size], args.teacher_batch_size)
        # prefix coordinates -> coordinates of the full training interval
        block = (block.astype(np.float64) * scale + mean - info['mean']) / info['scale']
        predictions[start:start + len(block)] = block.astype(np.float32)
    predictions.flush()
    return {'origins': origins, 'predictions': predictions}


def _fit_arrays(folds, values, group, block):
    (c0, c1), (h0, h1) = group, block
    xs, ys = [], []
    for fold in folds:
        base = np.asarray(fold['predictions'][:, h0:h1, c0:c1])
        xs.append(residual.build_features(values, fold['origins'], base))
        ys.append(residual.residual_targets(values, fold['origins'], h0, h1, base))
    return np.concatenate(xs), np.concatenate(ys)


def _select_correction(folds, info, args, device, log):
    """Forward validation: fit on earlier OOF intervals, score on the next one."""
    values = info['values'][:info['train_end']]
    groups = _groups(values.shape[1], args.teacher_series_group)
    blocks = _blocks(args.pred_len, args.teacher_forecast_group)
    candidates = {}
    for v in (1, 2):
        validation, fitting = folds[v], folds[:v]
        for gi, group in enumerate(groups):
            group_values = np.asarray(values[:, group[0]:group[1]])
            for bi, block in enumerate(blocks):
                x, y = _fit_arrays(fitting, group_values, group, block)
                base = np.asarray(validation['predictions'][:, block[0]:block[1], group[0]:group[1]])
                xv = residual.build_features(group_values, validation['origins'], base)
                truth = _targets(group_values, validation['origins'], block[0], block[1])
                for spec in residual.TREE_SPECS:
                    model = residual.fit_corrector(
                        x, y, spec, seed=args.teacher_seed + v * 1000 + gi * len(blocks) + bi,
                        device=device, cpu_threads=args.teacher_threads)
                    correction = model.predict(xv).reshape(base.shape)
                    for eta in residual.ETAS:
                        scores = candidates.setdefault((spec['name'], eta), {}).setdefault(
                            v, (residual.Scores(), residual.Scores()))
                        scores[0].update(base, truth)
                        scores[1].update(base + np.float32(eta) * correction, truth)
            log('  correction validation fold {} group {}/{}'.format(v, gi + 1, len(groups)))
    return residual.select_correction(candidates)


def build_teacher(info, args, device='cpu', log=print):
    """Fit the teacher for one dataset/horizon and write the guidance cache."""
    folder = teacher_dir(args)
    os.makedirs(folder, exist_ok=True)
    started = time.time()
    horizon, seq_len = args.pred_len, args.seq_len
    values, train_end = info['values'], info['train_end']
    channels = values.shape[1]
    ratio = np.asarray(info['scale'], np.float64) / np.asarray(info['score_scale'], np.float64)
    train_origins = np.arange(seq_len, train_end - horizon + 1, dtype=np.int64)
    val_origins = np.arange(info['val_start'], info['val_end'] - horizon + 1, dtype=np.int64)
    test_origins = np.arange(info['val_end'], info['test_end'] - horizon + 1, dtype=np.int64)
    few_shot = args.percent != 100
    use_residual = bool(args.teacher_residual) and not few_shot

    log('>>> building statistical teacher in {}'.format(folder))
    if few_shot:
        teacher = fit_teacher_few_shot(
            values, train_end, train_origins, val_origins, horizon, score_ratio=ratio, device=device,
            n_trials=args.teacher_trials, series_group=args.teacher_series_group,
            forecast_group=args.teacher_forecast_group, max_lookback=args.teacher_max_lookback,
            seed=args.teacher_seed, batch_size=args.teacher_batch_size, log=log)
    else:
        teacher = fit_teacher(
            values, train_end, horizon, device=device, n_trials=args.teacher_trials,
            series_group=args.teacher_series_group, forecast_group=args.teacher_forecast_group,
            max_lookback=args.teacher_max_lookback, seed=args.teacher_seed, log=log)
    valid = train_origins >= teacher.min_origin
    if not valid.any():
        raise ValueError('No training window has enough history for the teacher')
    eligible = train_origins[valid]

    selection = {'model_name': None, 'eta': 0.0}
    folds = None
    if use_residual:
        bounds = [int(train_end * k // 5) for k in (2, 3, 4, 5)]
        folds = [_oof_fold(i, bounds[i], bounds[i + 1], info, args, folder, device, log) for i in range(3)]
        selection = _select_correction(folds, info, args, device, log)
        log('selected correction: {} eta={} (relative MSE {:.4f}, MAE {:.4f})'.format(
            selection['model_name'], selection['eta'], selection['mse_ratio'], selection['mae_ratio']))
    eta = float(selection['eta'])
    spec = next((s for s in residual.TREE_SPECS if s['name'] == selection['model_name']), None)

    # Guidance targets, one variable group at a time (bounded memory).
    temporary = os.path.join(folder, 'train_predictions.partial.npy')
    cache = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                      shape=(len(train_origins), horizon, channels))
    cache[:] = 0.0
    rows = np.flatnonzero(valid)
    groups = _groups(channels, args.teacher_series_group)
    blocks = _blocks(horizon, args.teacher_forecast_group)
    scores = {split: {'base': residual.Scores(), 'teacher': residual.Scores()} for split in ('val', 'test')}
    size = args.teacher_batch_size
    for gi, group in enumerate(groups):
        c0, c1 = group
        sub = _restrict(teacher, group)
        group_values = np.asarray(values[:, c0:c1])
        train_values = group_values[:train_end]
        models = {}
        if eta > 0:
            for bi, block in enumerate(blocks):
                x, y = _fit_arrays(folds, train_values, group, block)
                models[block[0]] = residual.fit_corrector(
                    x, y, spec, seed=args.teacher_seed + 10000 + gi * len(blocks) + bi,
                    device=device, cpu_threads=args.teacher_threads)

        def forecast(source, origins):
            base = sub.predict(source, origins, size)
            corrected = base.copy()
            for h0, h1 in blocks if eta > 0 else ():
                x = residual.build_features(source, origins, base[:, h0:h1])
                corrected[:, h0:h1] += np.float32(eta) * models[h0].predict(x).reshape(len(origins), h1 - h0, -1)
            return base, corrected

        for start in range(0, len(eligible), size):
            _, corrected = forecast(train_values, eligible[start:start + size])
            cache[rows[start:start + size], :, c0:c1] = corrected
        for split, origins in (('val', val_origins), ('test', test_origins)):
            origins = origins[origins >= teacher.min_origin]
            for start in range(0, len(origins), size):
                current = origins[start:start + size]
                base, corrected = forecast(group_values, current)
                truth = _targets(group_values, current, 0, horizon)
                scores[split]['base'].update(base, truth, ratio[c0:c1])
                scores[split]['teacher'].update(corrected, truth, ratio[c0:c1])
        log('  guidance targets: group {}/{}'.format(gi + 1, len(groups)))
    cache.flush()
    del cache
    os.replace(temporary, os.path.join(folder, 'train_predictions.npy'))
    np.save(os.path.join(folder, 'train_origins.npy'), train_origins)
    np.save(os.path.join(folder, 'train_valid.npy'), valid)
    if folds is not None and not args.teacher_keep_oof:
        for index in range(len(folds)):
            folds[index]['predictions'] = None
            os.remove(os.path.join(folder, 'oof_{}.npy'.format(index)))

    metrics = {split: {name: {'mse': s.mse, 'mae': s.mae} for name, s in parts.items()}
               for split, parts in scores.items()}
    lookbacks = [int(cell['lookback']) for cell in teacher.cells]
    meta = teacher.save(os.path.join(folder, 'ridge_weights.npz'), identity=_identity(info, args),
                        selection={k: v for k, v in selection.items() if k != 'candidates'},
                        correction_candidates=selection.get('candidates', []), metrics=metrics,
                        train_windows=int(len(train_origins)), eligible_windows=int(valid.sum()),
                        lookback_min=min(lookbacks), lookback_median=float(np.median(lookbacks)),
                        lookback_max=max(lookbacks), build_seconds=time.time() - started)
    with open(os.path.join(folder, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log('teacher done in {:.1f}s | eta={} | guidance windows {}/{} | test mse:{:.4f} mae:{:.4f}'.format(
        time.time() - started, eta, int(valid.sum()), len(train_origins),
        metrics['test']['teacher']['mse'], metrics['test']['teacher']['mae']))
    return load_teacher_cache(info, args)
