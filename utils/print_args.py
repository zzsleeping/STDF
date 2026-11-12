def _row(a, av, b=None, bv=None):
    line = f'  {a + ":":<20}{str(av):<24}'
    if b is not None:
        line += f'{b + ":":<20}{str(bv):<24}'
    print(line)


def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    _row('Task Name', args.task_name, 'Is Training', args.is_training)
    _row('Model ID', args.model_id, 'Model', args.model)
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    _row('Data', args.data, 'Root Path', args.root_path)
    _row('Data Path', args.data_path, 'Checkpoints', args.checkpoints)
    _row('Percent', args.percent, 'Zero Shot', args.zero_shot)
    print()

    print("\033[1m" + "Forecasting Task" + "\033[0m")
    _row('Seq Len', args.seq_len, 'Pred Len', args.pred_len)
    print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    _row('GPT Layers', args.gpt_layers, 'GPT Path', args.gpt_path)
    _row('LoRA r', args.r, 'LoRA alpha', args.lora_alpha)
    _row('LoRA dropout', args.lora_dropout, 'Use Memory', args.use_memory)
    _row('PCA Memory', args.word_embedding_path, 'PCA Components', args.pca_components)
    print()

    print("\033[1m" + "Statistical Teacher" + "\033[0m")
    _row('Use Teacher', args.use_teacher, 'Teacher Path', args.teacher_path)
    _row('Residual', args.teacher_residual, 'Trials', args.teacher_trials)
    _row('Max Lookback', args.teacher_max_lookback, 'Teacher Seed', args.teacher_seed)
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    _row('Num Workers', args.num_workers, 'Itr', args.itr)
    _row('Train Epochs', args.train_epochs, 'Batch Size', args.batch_size)
    _row('Patience', args.patience, 'Learning Rate', args.learning_rate)
    _row('Tmax', args.tmax, 'Seed', args.seed)
    _row('Des', args.des, 'Time-scale Aug', args.time_scale_aug)
    print()

    print("\033[1m" + "Guidance Loss Weight" + "\033[0m")
    _row('Con Weight', args.con_w, 'Rob Weight', args.rob_w)
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    _row('Use GPU', args.use_gpu, 'GPU', args.gpu)
    print()
