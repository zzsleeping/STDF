import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Forecast, load_forecast_data

_cache = {}


def get_info(args, root_path=None, data_path=None, data=None):
    """Load (once) the standardized series of a benchmark."""
    key = (root_path or args.root_path, data_path or args.data_path, data or args.data,
           args.seq_len, args.percent, args.target)
    if key not in _cache:
        _cache[key] = load_forecast_data(*key)
    return _cache[key]


def data_provider(args, flag):
    info = get_info(args)
    data_set = Dataset_Forecast(info, flag=flag, size=[args.seq_len, args.pred_len])
    if flag == 'train':
        # Few-shot budgets can be smaller than one batch: keep every window.
        few_shot = args.percent != 100
        batch_size = min(args.batch_size, len(data_set)) if few_shot else args.batch_size
        shuffle_flag, drop_last = True, not few_shot
    else:
        batch_size, shuffle_flag, drop_last = args.eval_batch_size, False, False
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag,
                             num_workers=args.num_workers, drop_last=drop_last,
                             generator=torch.Generator().manual_seed(args.seed))
    return data_set, data_loader
