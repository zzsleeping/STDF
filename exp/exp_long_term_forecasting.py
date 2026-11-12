from data_provider.data_factory import data_provider, get_info
from exp.exp_basic import Exp_Basic
from utils.augmentation import choose_scale, time_scale
from utils.losses import GuidanceLoss
from utils.tools import EarlyStopping, set_seed
import torch
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # The statistical teacher is fitted (or its cache is read) and frozen
        # before the student exists; the student only sees its forecasts.
        self.teacher_pred, self.teacher_valid = None, None
        if args.is_training and args.use_teacher:
            self._prepare_teacher(args)
        set_seed(args.seed)
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _prepare_teacher(self, args):
        # Imported here so that testing a trained student needs no teacher code
        # (Optuna, CatBoost) and is not touched by its global settings.
        from teacher.build import build_teacher, load_teacher_cache
        info = get_info(args)
        cache = load_teacher_cache(info, args)
        if cache is None:
            cache = build_teacher(info, args, device=args.teacher_device)
        else:
            print('Use cached teacher forecasts')
        self.teacher_pred, self.teacher_valid = cache
        print('teacher guidance windows: {}/{}'.format(int(self.teacher_valid.sum()), len(self.teacher_valid)))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.teacher_pred is None:
            return GuidanceLoss(0.0, 0.0)
        return GuidanceLoss(self.args.con_w, self.args.rob_w)

    def _teacher_batch(self, index):
        """Precomputed teacher forecasts of the windows in this minibatch."""
        if self.teacher_pred is None:
            return None, None
        index = index.numpy()
        valid = torch.from_numpy(self.teacher_valid[index]).to(self.device)
        teacher = torch.from_numpy(np.asarray(self.teacher_pred[index], dtype=np.float32)).to(self.device)
        return teacher, valid

    def _score_ratio(self, info):
        # Errors are reported in units of the full training split. In the
        # full-data setting these are the model's own coordinates (ratio 1).
        ratio = np.asarray(info['scale'], np.float64) / np.asarray(info['score_scale'], np.float64)
        return torch.as_tensor(ratio, dtype=torch.float64, device=self.device)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if self.teacher_pred is not None and len(self.teacher_pred) != len(train_data):
            raise ValueError('Teacher cache does not match the training windows')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                teacher, valid = self._teacher_batch(index)

                if self.args.time_scale_aug:
                    q = choose_scale(self.args.seed, epoch + 1, i)
                    batch_x, batch_y, teacher = time_scale(batch_x, batch_y, teacher, valid, q)

                outputs = self.model(batch_x)['outputs_time']

                loss, _ = criterion(outputs, batch_y, teacher, valid)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss, _ = self.vali(vali_data, vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

            # The checkpoint is selected by validation MSE only.
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader):
        ratio = self._score_ratio(get_info(self.args))
        squared, absolute, count = 0.0, 0.0, 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)['outputs_time']

                error = (outputs.double() - batch_y.double()) * ratio
                squared += error.square().sum().item()
                absolute += error.abs().sum().item()
                count += error.numel()
        self.model.train()

        return squared / count, absolute / count

    def _zero_shot_test(self, target_data):
        """Evaluate the source-trained student on another dataset without any update.

        Inputs are standardized with the source training statistics stored with
        the experiment; target statistics only define the error units.
        """
        args = self.args
        source = get_info(args)
        target = get_info(args, args.target_root_path or args.root_path, '{}.csv'.format(target_data), target_data)
        raw = target['raw']
        if raw.shape[1] != len(source['mean']):
            raise ValueError('Source and target must have the same variables')
        origins = np.arange(target['val_end'], target['test_end'] - args.pred_len + 1)
        steps = np.arange(args.pred_len)
        squared, absolute, count = 0.0, 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(origins), args.eval_batch_size):
                current = origins[start:start + args.eval_batch_size]
                history = raw[current[:, None] + np.arange(-args.seq_len, 0)[None, :]]
                x = ((history - source['mean']) / source['scale']).astype(np.float32)
                outputs = self.model(torch.from_numpy(x).to(self.device))['outputs_time'].cpu().numpy()
                forecast = outputs.astype(np.float64) * source['scale'] + source['mean']
                error = (forecast - raw[current[:, None] + steps[None, :]]) / target['score_scale']
                squared += float(np.square(error).sum())
                absolute += float(np.abs(error).sum())
                count += error.size
        return squared / count, absolute / count

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
                                                  map_location=self.device))

        if self.args.zero_shot:
            # one source model, any number of comma-separated targets
            for target_data in self.args.target_data.split(','):
                mse, mae = self._zero_shot_test(target_data)
                self._save_result(setting + '_to_' + target_data, mse, mae)
            return

        test_data, test_loader = self._get_data(flag='test')
        mse, mae = self.vali(test_data, test_loader)
        self._save_result(setting, mse, mae)
        return

    def _save_result(self, setting, mse, mae):
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse]))

        return
