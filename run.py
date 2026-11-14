import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STDF')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, default='STDF', help='model name, options: [STDF]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1',
                        help='dataset type, options:[ETTh1, ETTh2, ETTm1, ETTm2, custom]')
    parser.add_argument('--root_path', type=str, default='./datasets/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str, default='OT', help='column moved last in custom datasets')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='student input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='unused; kept for CALF-compatible scripts')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times; run i uses seed + i')
    parser.add_argument('--seed', type=int, default=2021, help='student seed')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='batch size of validation and test')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--tmax', type=int, default=20, help='T_max of cosine annealing')
    parser.add_argument('--des', type=str, default='test', help='exp description')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # gpt
    parser.add_argument('--gpt_layers', type=int, default=6, help='number of GPT-2 blocks')
    parser.add_argument('--gpt_path', type=str, default='gpt2', help='name or local path of pretrained GPT-2')

    # lora
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # frozen memory interface
    parser.add_argument('--use_memory', type=int, default=1, help='1: query the frozen PCA memory')
    parser.add_argument('--word_embedding_path', type=str, default='wte_pca_500.pt', help='PCA memory from pca.py')
    parser.add_argument('--pca_components', type=int, default=500)
    parser.add_argument('--cross_rank', type=int, default=64, help='cross-attention width (heads x 16)')
    parser.add_argument('--cross_heads', type=int, default=4)

    # statistical teacher (fitted offline, frozen, never executed at inference)
    parser.add_argument('--use_teacher', type=int, default=1, help='0: supervised only')
    parser.add_argument('--teacher_path', type=str, default='./teacher_cache/', help='teacher forecast caches')
    parser.add_argument('--teacher_residual', type=int, default=1, help='1: add the residual correction')
    parser.add_argument('--teacher_trials', type=int, default=20, help='TPE trials per (group, block) unit')
    parser.add_argument('--teacher_max_lookback', type=int, default=2048, help='longest teacher history')
    parser.add_argument('--teacher_series_group', type=int, default=7, help='variables per group')
    parser.add_argument('--teacher_forecast_group', type=int, default=24, help='steps per forecast block')
    parser.add_argument('--teacher_seed', type=int, default=2021)
    parser.add_argument('--teacher_device', type=str, default='cpu', help='device of the Ridge solver')
    parser.add_argument('--teacher_threads', type=int, default=8, help='CatBoost CPU threads')
    parser.add_argument('--teacher_batch_size', type=int, default=256)
    parser.add_argument('--teacher_keep_oof', type=int, default=0, help='1: keep out-of-fold forecasts on disk')

    # guidance loss weight: L = MAE(S,Y) + con_w * MSE(S,T) + rob_w * MAE(S,T)
    parser.add_argument('--con_w', type=float, default=0.4, help='lambda, weight of the consistency term')
    parser.add_argument('--rob_w', type=float, default=0.4, help='beta, weight of the robust guidance term')

    # few shot percentage
    parser.add_argument('--percent', type=int, default=100, help='few shot percentage')

    # zero shot
    parser.add_argument('--zero_shot', type=int, default=0, help='1: test on target_data')
    parser.add_argument('--target_data', type=str, default='ETTm1',
                        help='target data for zero shot (comma-separated; files <name>.csv)')
    parser.add_argument('--target_root_path', type=str, default='', help='default: root_path')
    parser.add_argument('--time_scale_aug', type=int, default=0, help='1: time-scale augmentation in training')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    base_seed = args.seed

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    for ii in range(args.itr):
        args.seed = base_seed + ii
        # private stream of the memory cross-attention initialization
        args.cross_seed = 400000 + args.seed
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_pl{}_gpt{}_p{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.gpt_layers,
            args.percent,
            args.des,
            args.seed)

        exp = Exp(args)  # set experiments
        if args.is_training:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
