import csv
import os

import numpy as np
from torch.utils.data import Dataset

ETT_DATA = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')


def split_boundaries(data, n_rows):
    """Standard chronological train/val/test ends."""
    if data in ETT_DATA:
        factor = 4 if data.startswith('ETTm') else 1
        return 12 * 30 * 24 * factor, 16 * 30 * 24 * factor, 20 * 30 * 24 * factor
    num_train, num_test = int(n_rows * 0.7), int(n_rows * 0.2)
    return num_train, n_rows - num_test, n_rows


def population_stats(raw):
    """Mean/std as sklearn's StandardScaler (population variance, constant columns -> 1)."""
    mean, var = raw.mean(0), raw.var(0)
    eps = np.finfo(np.float64).eps
    scale = np.sqrt(var)
    scale[var <= len(raw) * eps * var + (len(raw) * mean * eps) ** 2] = 1.0
    return mean, scale


def read_csv(root_path, data_path, data, target='OT'):
    """[T, C] float64 observations; custom datasets move the target to the last column."""
    with open(os.path.join(root_path, data_path), newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = header[1:]
        order = list(range(len(columns)))
        if data not in ETT_DATA and target in columns:
            order.remove(columns.index(target))
            order.append(columns.index(target))
        rows = [[float(row[1 + i]) for i in order] for row in reader if row]
    raw = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('{} contains missing or non-finite values'.format(data_path))
    return raw, [columns[i] for i in order]


def load_forecast_data(root_path, data_path, data, seq_len=96, percent=100, target='OT'):
    """Read one benchmark and standardize it with training statistics.

    percent < 100 selects the few-shot protocol of CALF: only the first
    ``seq_len + (N - seq_len) * percent // 100`` time steps of the training
    split are available.  Inputs are then standardized with that prefix, while
    ``score_scale`` (statistics of the full training split) only defines the
    units in which errors are reported.
    """
    raw, columns = read_csv(root_path, data_path, data, target)
    train_end, val_end, test_end = split_boundaries(data, len(raw))
    raw = raw[:test_end]
    fit_end = train_end if percent == 100 else seq_len + (train_end - seq_len) * percent // 100
    mean, scale = population_stats(raw[:fit_end])
    _, score_scale = population_stats(raw[:train_end])
    return {'data': data, 'columns': columns, 'raw': raw,
            'values': ((raw - mean) / scale).astype(np.float32),
            'mean': mean, 'scale': scale, 'score_scale': score_scale,
            'train_end': fit_end, 'val_start': train_end, 'val_end': val_end, 'test_end': test_end}


class Dataset_Forecast(Dataset):
    """Sliding windows indexed by forecast origin.

    Each item is (history [L, C], future [H, C], index).  The index addresses
    the precomputed teacher forecast of the same training window.
    """

    def __init__(self, info, flag='train', size=None):
        self.seq_len, self.pred_len = size
        first, end = {'train': (self.seq_len, info['train_end']),
                      'val': (info['val_start'], info['val_end']),
                      'test': (info['val_end'], info['test_end'])}[flag]
        if end - self.pred_len < first:
            raise ValueError('{} split is too short for L={} H={}'.format(flag, self.seq_len, self.pred_len))
        self.values = info['values']
        self.origins = np.arange(first, end - self.pred_len + 1, dtype=np.int64)

    def __getitem__(self, index):
        t = int(self.origins[index])
        return self.values[t - self.seq_len:t].copy(), self.values[t:t + self.pred_len].copy(), index

    def __len__(self):
        return len(self.origins)
