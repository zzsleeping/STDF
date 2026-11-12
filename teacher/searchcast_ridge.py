"""Grouped Ridge forecaster with searched lookback, local normalization and penalty.

This file is a subset of ``optuna_ridge.py`` from SearchCast
(Huang, Xu and Darlow, "How Good Can Linear Models Be for Time-Series
Forecasting?", https://github.com/SakanaAI/SearchCast), licensed under the
Apache License, Version 2.0.  Changes for STGF: the command-line entry point,
the global-baseline search and the plotting/reporting utilities were removed,
together with the imports only they needed.  The scalers, augmentations, Ridge
solver, windowing helpers and the single-objective search wrapper below are
unchanged.
"""

import os

import torch
import optuna

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. Scaling Strategies & Scalers
# ==========================================
class ScalerStrategy:
    """Base class for computing center and scale statistics separately."""

    def compute_center(self, tensor, dim):
        raise NotImplementedError

    def compute_scale(self, tensor, dim):
        raise NotImplementedError


class StandardStrategy(ScalerStrategy):
    """Mean for center, std for scale."""

    def compute_center(self, tensor, dim):
        return torch.mean(tensor, dim=dim, keepdim=True)

    def compute_scale(self, tensor, dim):
        # Biased variance (correction=0) to match numpy/reference
        var = torch.var(tensor, dim=dim, keepdim=True, correction=0)
        return torch.sqrt(var + 1e-5)


class RobustStrategy(ScalerStrategy):
    """Median for center, IQR for scale."""

    def __init__(self, q_min=0.25, q_max=0.75):
        self.q_min = q_min
        self.q_max = q_max

    def compute_center(self, tensor, dim):
        return torch.quantile(tensor, 0.50, dim=dim, keepdim=True)

    def compute_scale(self, tensor, dim):
        q_l = torch.quantile(tensor, self.q_min, dim=dim, keepdim=True)
        q_h = torch.quantile(tensor, self.q_max, dim=dim, keepdim=True)
        return (q_h - q_l).clamp_min(1e-5)


class IdentityStrategy(ScalerStrategy):
    """No normalization."""

    def compute_center(self, tensor, dim):
        return torch.zeros(1, device=tensor.device)

    def compute_scale(self, tensor, dim):
        return torch.ones(1, device=tensor.device)


class GlobalScaler:
    """
    Global normalization: (X - center) / scale.

    Both center and scale are computed on the original (non-centered) data.
    Computes global (scalar) statistics over all data for dataset-level preprocessing.
    """

    def __init__(self, strategy: ScalerStrategy):
        self.strategy = strategy
        self.center = None
        self.scale = None

    def fit(self, X: torch.Tensor):
        """Compute global center and scale statistics.

        X: (S, T) series × time
        Computes scalar statistics over all values.
        """
        # Flatten to compute global statistics
        X_flat = X.reshape(-1, 1)  # (S*T, 1)
        self.center = self.strategy.compute_center(X_flat, dim=0)  # (1, 1) scalar
        self.scale = self.strategy.compute_scale(X_flat, dim=0)  # (1, 1) scalar

    def transform(self, X: torch.Tensor):
        """Apply (X - center) / scale."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        if isinstance(self.strategy, IdentityStrategy):
            return X
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        return (X - center) / scale

    def inv_transform(self, X: torch.Tensor):
        """Reverse: X * scale + center."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        return X * scale + center


class LocalNormScaler:
    """
    Local normalization (instance-norm style).

    Center is computed on original data, scale is computed on centered data.
    Transform subtracts center and appends scale as a feature.

    Args:
        strategy: ScalerStrategy (Standard, Robust, or Identity)
        lookback: context length
        last_k: trailing samples to compute stats from (default: lookback)
    """

    def __init__(self, strategy: ScalerStrategy, lookback: int, last_k: int = None):
        self.strategy = strategy
        self.lookback = lookback
        self.last_k = last_k or lookback
        self.center = None
        self.scale = None

    def fit(self, X: torch.Tensor):
        """Compute center on original data, scale on centered data."""
        wins = X[:, -self.last_k:] if X.dim() == 2 and X.shape[-1] >= self.last_k else X
        self.center = self.strategy.compute_center(wins, dim=-1)
        # Compute scale on centered data
        centered = wins - self.center
        # self.scale = self.strategy.compute_scale(centered, dim=-1)
        # the scale is std deviation regardless of strategy for local norm
        var = torch.var(centered, dim=-1, keepdim=True, correction=0)
        self.scale = torch.sqrt(var + 1e-5)

    def transform(self, X: torch.Tensor):
        """Subtract center, append scale as feature. Returns (N, L+1)."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        X_centered = X - center
        return torch.cat([X_centered, scale], dim=-1)

    def transform_target(self, Y: torch.Tensor):
        """Subtract center from target (no scale appending)."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(Y.device, non_blocking=True)
        return Y - center

    def inv_transform(self, pred: torch.Tensor):
        """Add center back to predictions."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(pred.device, non_blocking=True)
        return pred + center

    @property
    def output_dim(self):
        """Feature dimension after transform (lookback + 1)."""
        return self.lookback + 1


class Augmentor:
    def apply(self, X): return X

class TimeDomainNoise(Augmentor):
    def __init__(self, sigma): self.sigma = sigma
    def apply(self, X):
        if self.sigma <= 0: return X
        return X.add_(torch.randn_like(X), alpha=self.sigma)

class FreqDomainNoise(Augmentor):
    def __init__(self, sigma, mode='amplitude'):
        """
        mode: 'amplitude' (scale magnitude) or 'phase' (shift phase)
        """
        self.sigma = sigma
        self.mode = mode

    def apply(self, X):
        # X shape: (Batch, Time, Feat) or (Batch, Time)
        # 1. FFT
        # rfft computes the real-input FFT (faster, gives only positive freqs)
        X_freq = torch.fft.rfft(X, dim=1) 
        
        # 2. Perturb
        if self.mode == 'amplitude':
            # Perturb magnitude: Multiply by random scale ~ N(1, sigma)
            # Create noise for (Batch, Freq_Bins, Feat)
            noise = torch.randn_like(X_freq.abs()) * self.sigma
            # Apply to amplitude, keep phase
            # New_Complex = (Old_Abs + Noise) * e^(i * Old_Phase)
            # Efficient way: scale real and imaginary parts uniformly? 
            # Better: Scale magnitude directly
            scale = 1.0 + noise
            X_freq = X_freq * scale
            
        elif self.mode == 'phase':
            # Perturb phase: Add random angle ~ N(0, sigma)
            phase_noise = torch.randn_like(X_freq.angle()) * self.sigma
            # New_Complex = Old_Complex * e^(i * phase_noise)
            # Euler's formula: e^(ix) = cos(x) + i*sin(x)
            rotation = torch.polar(torch.ones_like(X_freq.abs()), phase_noise)
            X_freq = X_freq * rotation

        # 3. Inverse FFT
        X_aug = torch.fft.irfft(X_freq, n=X.shape[1], dim=1)
        return X_aug


def apply_augmentation(X, config=None):
    if config is None: return X
    X_aug = X.clone()
    if config['noise_type'] == 'time':
        X_aug = TimeDomainNoise(config['sigma']).apply(X_aug)
    elif config['noise_type'] == 'freq':
        X_aug = FreqDomainNoise(config['sigma']).apply(X_aug)
    return X_aug

# ==========================================
# 2. Solver
# ==========================================
class RidgeSolver:
    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def solve(self, X_train, Y_train, alphas, scaler=None, chunk_size=50000, aug_config=None, dtype=torch.float64):
        """
        Solve ridge regression for multiple alpha values.

        Args:
            X_train: (N, L) input features
            Y_train: (N, H) targets
            alphas: (K,) regularization strengths
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
                    - LocalNormScaler: per-sample normalization, appends scale feature, no intercept
                    - GlobalScaler: dataset-level normalization, adds intercept
                    - None: no normalization, adds intercept
            chunk_size: batch size for memory-efficient computation
            aug_config: augmentation configuration
            dtype: computation dtype

        Returns:
            Theta: (K, F, H) weight matrices for each alpha
        """
        N, L = X_train.shape
        _, H = Y_train.shape
        K = alphas.shape[0]

        # Determine feature dimension and intercept based on scaler type
        use_local_norm = isinstance(scaler, LocalNormScaler)
        if use_local_norm:
            F = L + 1  # context features + scale feature (no intercept)
            fit_intercept = False
        else:
            F = L + 1  # features + intercept
            fit_intercept = True

        XTX = torch.zeros(F, F, device=self.device, dtype=dtype)
        XTY = torch.zeros(F, H, device=self.device, dtype=dtype)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_chunk = X_train[start:end].to(dtype=dtype, device=self.device, non_blocking=True)
            Y_chunk = Y_train[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

            if use_local_norm:
                # LocalNormScaler: fit per-chunk, transform X (appends scale), transform target Y
                scaler.fit(X_chunk)
                X_chunk = scaler.transform(X_chunk)
                Y_chunk = scaler.transform_target(Y_chunk)
                # Data augmentation (on normalized features, excluding appended scale)
                X_chunk_features = X_chunk[:, :-1]
                X_chunk_features = apply_augmentation(X_chunk_features, aug_config)
                X_chunk = torch.cat([X_chunk_features, X_chunk[:, -1:]], dim=-1)
            else:
                # GlobalScaler or None: transform if scaler provided
                if scaler is not None:
                    X_chunk = scaler.transform(X_chunk)
                    Y_chunk = scaler.transform(Y_chunk)
                # Data augmentation
                X_chunk = apply_augmentation(X_chunk, aug_config)
                # Add intercept term
                X_chunk = torch.cat([torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device), X_chunk], dim=1)

            # Accumulate the gram matrix
            XTX.add_(X_chunk.T @ X_chunk)
            XTY.add_(X_chunk.T @ Y_chunk)

        XTX_expanded = XTX.unsqueeze(0).expand(K, -1, -1)

        # Create regularization matrix
        if fit_intercept:
            # Don't regularize the intercept term
            reg_diags = alphas.unsqueeze(1).expand(-1, F).clone()
            reg_diags[:, 0] = 0.0
        else:
            # Regularize all features (LocalNormScaler case)
            reg_diags = alphas.unsqueeze(1).expand(-1, F)
        I_reg = torch.diag_embed(reg_diags)

        A = XTX_expanded + I_reg
        B = XTY.unsqueeze(0).expand(K, -1, -1)

        # Solve Linear System
        try:
            Lchol = torch.linalg.cholesky(A)
            Theta = torch.cholesky_solve(B, Lchol)  # (K, F, H)
        except RuntimeError:
            try:
                Theta = torch.linalg.solve(A, B)
            except RuntimeError:
                XTX_inv = torch.linalg.pinv(A)
                Theta = torch.einsum('kij,kjh->kih', XTX_inv, B)

        torch.cuda.empty_cache()
        return Theta

    @torch.no_grad()
    def predict(self, X, theta, scaler=None, chunk_size=50000, dtype=torch.float64):
        """
        Make predictions using trained weights.

        Args:
            X: (N, L) input features (raw, before any normalization)
            theta: (K, F, H) weight matrices
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
                    - LocalNormScaler: fit, transform (appends scale), predict, inv_transform
                    - GlobalScaler: transform only, predict (caller handles inv_transform)
                    - None: add intercept, predict
            chunk_size: batch size for memory-efficient prediction
            dtype: computation dtype

        Returns:
            Y_pred: (K, N, H) predictions
        """
        N = X.shape[0]
        K, _, H = theta.shape
        theta = theta.to(dtype=dtype, device=self.device, non_blocking=True)

        use_local_norm = isinstance(scaler, LocalNormScaler)

        # Process in chunks for memory efficiency
        Y_pred_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_chunk = X[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

            if use_local_norm:
                # LocalNormScaler: fit, transform, predict, then inv_transform
                scaler.fit(X_chunk)
                X_transformed = scaler.transform(X_chunk)
                Y_pred_norm = torch.einsum('nf, kfh -> knh', X_transformed, theta)
                Y_pred_chunk = scaler.inv_transform(Y_pred_norm)
            else:
                # GlobalScaler or None: add intercept and predict
                if scaler is not None:
                    X_chunk = scaler.transform(X_chunk)
                ones = torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device)
                X_chunk = torch.cat([ones, X_chunk], dim=1)
                Y_pred_chunk = torch.einsum('nf, kfh -> knh', X_chunk, theta)

            Y_pred_chunks.append(Y_pred_chunk.cpu())

        Y_pred = torch.cat(Y_pred_chunks, dim=1).to(self.device)
        return Y_pred

    @torch.no_grad()
    def solve_batched(self, X_train_batch, Y_train_batch, alphas, scaler=None, chunk_size=50000, aug_config=None, dtype=torch.float64):
        """
        Solve ridge regression for multiple series simultaneously (batched).

        Each series gets its own separate Ridge model with unique weights.
        Adaptively batches over series and alphas to fit GPU memory.

        Args:
            X_train_batch: (S, N_s, L) input features for S series
            Y_train_batch: (S, N_s, H) targets for S series
            alphas: (K,) regularization strengths
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
            chunk_size: batch size for memory-efficient computation per series
            aug_config: augmentation configuration
            dtype: computation dtype

        Returns:
            Theta: (K, S, F, H) weight matrices on CPU
        """
        S, N_s, L = X_train_batch.shape
        _, _, H = Y_train_batch.shape
        K = alphas.shape[0]

        # Determine feature dimension and intercept based on scaler type
        use_local_norm = isinstance(scaler, LocalNormScaler)
        if use_local_norm:
            F = L + 1  # context features + scale feature (no intercept)
            fit_intercept = False
        else:
            F = L + 1  # features + intercept
            fit_intercept = True

        # Theta on CPU — results are read by predict_batched which also works per-series
        Theta = torch.zeros(K, S, F, H, dtype=dtype)

        # Determine series_batch: must fit Gram matrices + at least 1 alpha solve
        # Gram: Sb * (F*F + F*H) * 8 bytes for XTX + XTY
        # Solve (min): 1 * Sb * F*F * 8 * 3 bytes for A, L, workspace
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(self.device)
        gpu_budget = int(free_mem * 0.7)
        bytes_per_series = (F * F + F * H + F * F * 3) * 8  # Gram + 1-alpha solve
        series_batch = max(1, min(S, gpu_budget // bytes_per_series))

        for s_start in range(0, S, series_batch):
            s_end = min(s_start + series_batch, S)
            Sb = s_end - s_start

            # --- Phase 1: Accumulate Gram matrices for this series batch ---
            XTX = torch.zeros(Sb, F, F, device=self.device, dtype=dtype)
            XTY = torch.zeros(Sb, F, H, device=self.device, dtype=dtype)

            for si, s in enumerate(range(s_start, s_end)):
                X_s = X_train_batch[s]  # (N_s, L)
                Y_s = Y_train_batch[s]  # (N_s, H)

                for start in range(0, N_s, chunk_size):
                    end = min(start + chunk_size, N_s)
                    X_chunk = X_s[start:end].to(dtype=dtype, device=self.device, non_blocking=True)
                    Y_chunk = Y_s[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

                    if use_local_norm:
                        scaler.fit(X_chunk)
                        X_chunk = scaler.transform(X_chunk)
                        Y_chunk = scaler.transform_target(Y_chunk)
                        X_chunk_features = X_chunk[:, :-1]
                        X_chunk_features = apply_augmentation(X_chunk_features, aug_config)
                        X_chunk = torch.cat([X_chunk_features, X_chunk[:, -1:]], dim=-1)
                    else:
                        if scaler is not None:
                            X_chunk = scaler.transform(X_chunk)
                            Y_chunk = scaler.transform(Y_chunk)
                        X_chunk = apply_augmentation(X_chunk, aug_config)
                        X_chunk = torch.cat([torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device), X_chunk], dim=1)

                    XTX[si].add_(X_chunk.T @ X_chunk)
                    XTY[si].add_(X_chunk.T @ Y_chunk)

            # --- Phase 2: Solve for all alphas, adaptively batched ---
            # Re-query free memory after Gram allocation
            torch.cuda.empty_cache()
            free_mem2, _ = torch.cuda.mem_get_info(self.device)
            solve_budget = int(free_mem2 * 0.7)
            # Each alpha in this batch needs Sb * F * F * 8 * 3 bytes
            solve_bytes_per_alpha = Sb * F * F * 8 * 3
            alpha_batch = max(1, min(K, solve_budget // max(1, solve_bytes_per_alpha)))

            for k_start in range(0, K, alpha_batch):
                k_end = min(k_start + alpha_batch, K)
                Kb = k_end - k_start
                alpha_chunk = alphas[k_start:k_end]

                diag_vals = alpha_chunk.view(Kb, 1, 1).expand(Kb, Sb, F).clone()
                if fit_intercept:
                    diag_vals[:, :, 0] = 0.0
                I_reg = torch.diag_embed(diag_vals)

                A = XTX.unsqueeze(0).expand(Kb, -1, -1, -1) + I_reg  # (Kb, Sb, F, F)
                B = XTY.unsqueeze(0).expand(Kb, -1, -1, -1).clone()  # (Kb, Sb, F, H)

                try:
                    L = torch.linalg.cholesky(A)
                    Theta[k_start:k_end, s_start:s_end] = torch.cholesky_solve(B, L).cpu()
                except RuntimeError:
                    try:
                        Theta[k_start:k_end, s_start:s_end] = torch.linalg.solve(A, B).cpu()
                    except RuntimeError:
                        A_inv = torch.linalg.pinv(A)
                        Theta[k_start:k_end, s_start:s_end] = torch.einsum('ksij,ksjh->ksih', A_inv, B).cpu()

                del A, B, I_reg, diag_vals

            del XTX, XTY
            torch.cuda.empty_cache()

        return Theta

    @torch.no_grad()
    def predict_batched(self, X_batch, theta, scaler=None, chunk_size=50000, dtype=torch.float64):
        """
        Make predictions using batched trained weights (one model per series).

        Args:
            X_batch: (S, N_s, L) input features for S series
            theta: (K, S, F, H) weight matrices
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
            chunk_size: batch size for memory-efficient prediction per series
            dtype: computation dtype

        Returns:
            Y_pred: (K, S, N_s, H) predictions
        """
        S, N_s, L = X_batch.shape
        K, _, _, H = theta.shape
        # theta stays on CPU; load per-series to GPU to avoid large GPU allocation

        use_local_norm = isinstance(scaler, LocalNormScaler)

        Y_pred_all = []

        # Process each series
        for s in range(S):
            X_s = X_batch[s]  # (N_s, L)
            theta_s = theta[:, s, :, :].to(dtype=dtype, device=self.device, non_blocking=True)  # (K, F, H)

            # Process in chunks for memory efficiency
            Y_pred_chunks = []
            for start in range(0, N_s, chunk_size):
                end = min(start + chunk_size, N_s)
                X_chunk = X_s[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

                if use_local_norm:
                    # LocalNormScaler: fit, transform, predict, then inv_transform
                    scaler.fit(X_chunk)
                    X_transformed = scaler.transform(X_chunk)
                    Y_pred_norm = torch.einsum('nf, kfh -> knh', X_transformed, theta_s)
                    Y_pred_chunk = scaler.inv_transform(Y_pred_norm)
                else:
                    # GlobalScaler or None: add intercept and predict
                    if scaler is not None:
                        X_chunk = scaler.transform(X_chunk)
                    ones = torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device)
                    X_chunk = torch.cat([ones, X_chunk], dim=1)
                    Y_pred_chunk = torch.einsum('nf, kfh -> knh', X_chunk, theta_s)

                Y_pred_chunks.append(Y_pred_chunk.cpu())

            Y_pred_s = torch.cat(Y_pred_chunks, dim=1)  # (K, N_s, H)
            Y_pred_all.append(Y_pred_s)

        Y_pred = torch.stack(Y_pred_all, dim=1)  # (K, S, N_s, H) — stays on CPU
        return Y_pred

# ==========================================
# 3. Data Pipeline
# ==========================================
def get_context_and_horizons(data, lookback, horizons):
    if isinstance(horizons, int):
        H_max = horizons
        idx_tensor = torch.tensor([horizons], dtype=torch.long)
    else:
        H_max = max(horizons)
        idx_tensor = torch.tensor(horizons, dtype=torch.long)

    # Dynamic Window: lookback + max_horizon required
    window_size = lookback + H_max
    if data.shape[1] < window_size:
        return None, None

    data_windowed = data.unfold(1, window_size, 1)
    X = data_windowed[:, :, :lookback]
    target_indices = lookback + idx_tensor - 1
    Y = torch.index_select(data_windowed, dim=2, index=target_indices)
    return X, Y


def create_strategy(method: str, scaler_config: dict) -> ScalerStrategy:
    """Create a ScalerStrategy based on method name."""
    if method == "mean":
        return StandardStrategy()
    elif method == "robust":
        q_min = scaler_config.get('q_min', 0.25)
        q_max = scaler_config.get('q_max', 0.75)
        return RobustStrategy(q_min, q_max)
    elif method == "none":
        return IdentityStrategy()
    else:
        raise ValueError(f"Unknown scaler method: {method}")


def create_scaler(strategy: ScalerStrategy, scope: str, lookback: int,
                  scaler_config: dict):
    """Create GlobalScaler or LocalNormScaler based on scope."""
    if scope == 'global':
        return GlobalScaler(strategy)
    elif scope == 'local':
        ratio = scaler_config.get('local_ratio', 1.0)
        last_k = max(1, int(lookback * ratio))
        return LocalNormScaler(strategy, lookback, last_k)
    else:
        raise ValueError(f"Unknown scope: {scope}")


def maybe_transform(X: torch.Tensor, scaler, force_no_fit: bool = False):
    """
    Apply scaler transformation, fitting if needed for LocalNormScaler.

    For GlobalScaler: assumes already fitted, just transforms.
    For LocalNormScaler: fits on X then transforms (unless force_no_fit).
    """
    if X is None:
        return None
    if isinstance(scaler, LocalNormScaler):
        if not force_no_fit:
            scaler.fit(X)
        return scaler.transform(X)
    elif isinstance(scaler, GlobalScaler):
        return scaler.transform(X)
    else:
        return X


def get_prepared_data(series_data, lookback, horizons, split_idx_1, split_idx_2,
                      scaler_config):
    """
    Prepare train/val/test data with appropriate scalers.

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers_dict
    """
    if not isinstance(horizons, (list, tuple)):
        horizons = [horizons]

    method = scaler_config['method']
    scope = scaler_config['scope']

    strategy = create_strategy(method, scaler_config)

    # Slice boundaries
    test_slice_start = max(0, split_idx_2 - lookback)

    # Get windows
    X_train, Y_train = get_context_and_horizons(
        series_data[:, :split_idx_1], lookback, horizons)

    if split_idx_2 > split_idx_1:
        X_val, Y_val = get_context_and_horizons(
            series_data[:, split_idx_1 - lookback:split_idx_2],
            lookback, horizons)
    else:
        X_val, Y_val = None, None

    X_test, Y_test = get_context_and_horizons(
        series_data[:, test_slice_start:], lookback, horizons)

    # Create scalers
    if scope == 'global':
        scaler_train = GlobalScaler(strategy)
        scaler_train.fit(series_data[:, :split_idx_1])
        scaler_val = scaler_test = scaler_train
    elif scope == 'local':
        ratio = scaler_config.get('local_ratio', 1.0)
        last_k = max(1, int(lookback * ratio))
        scaler_train = LocalNormScaler(strategy, lookback, last_k)
        scaler_val = LocalNormScaler(strategy, lookback, last_k)
        scaler_test = LocalNormScaler(strategy, lookback, last_k)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    scalers = {'train': scaler_train, 'val': scaler_val, 'test': scaler_test}
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers

# ==========================================
# 5. Objective Wrappers
# ==========================================
class SingleObjectiveWrapper:
    def __init__(self, data, series_idx, lookbacks, horizon, alphas, device,
                 train_ratio=0.4, test_ratio=0.2, split_starts=None, split_ends=None,
                 n_folds=1, fold_reg_lambda=0.0,
                 scaler_scope="search", scaler_method="search",
                 fixed_local_ratio=None, fixed_noise_type=None, fixed_aug_sigma=None,
                 pool_series=False):
        """
        Args:
            data: (T, S) time series data
            series_idx: index of target series, or None for all series
            lookbacks: list of valid lookback values for Optuna to search over
            horizon: forecast horizon(s)
            alphas: regularization strengths
            device: torch device
            train_ratio: fraction of data for training during search
            test_ratio: fraction of data for test (not used during search)
            split_starts: list of split start indices
            split_ends: list of split end indices
            n_folds: number of folds for expanding window CV (1 = single split)
            fold_reg_lambda: regularization weight for fold variance (0 = no regularization)
            scaler_scope: "global", "local", or "search" (default: "search")
            scaler_method: "mean", "robust", or "search" (default: "search")
            fixed_local_ratio: if set, use this value instead of searching
            fixed_noise_type: if set, use this augmentation type instead of searching
            fixed_aug_sigma: if set, use this augmentation sigma instead of searching
            pool_series: if True, pool training data across series (single model per group)
        """
        if series_idx is None:
            self.data = data.transpose(0, 1)
            self.is_batched = True
            self.n_series = data.shape[1]
        elif isinstance(series_idx, (list, tuple)):
            self.data = data[:, series_idx].transpose(0, 1)
            self.is_batched = len(series_idx) > 1
            self.n_series = len(series_idx)
        else:
            self.data = data[:, series_idx].unsqueeze(0)
            self.is_batched = False
            self.n_series = 1
        # Store lookback bounds for log-scale search
        self.min_lookback = min(lookbacks)
        self.max_lookback = max(lookbacks)
        self.horizon = horizon
        self.alphas = alphas
        self.device = device
        self.solver = RidgeSolver(device)
        self.n_folds = n_folds
        self.fold_reg_lambda = fold_reg_lambda
        self.pool_series = pool_series

        # Ablation controls
        self.scaler_scope = scaler_scope
        self.scaler_method = scaler_method
        self.fixed_local_ratio = fixed_local_ratio
        self.fixed_noise_type = fixed_noise_type
        self.fixed_aug_sigma = fixed_aug_sigma

        T = self.data.shape[1]

        # first train_ratio data is used as training set during search
        self.split_idx_1 = int(T * train_ratio)
        # the final test_ratio is test, which is not used during search
        self.split_idx_2 = int(T * (1 - test_ratio))

        # use default 0.7/0.1/0.2 split for benchmark if the splits is not provided
        self.split_starts = split_starts
        self.split_ends = split_ends

        # Precompute fold boundaries for expanding window k-fold CV
        if n_folds > 1:
            trainval_len = self.split_idx_2
            fold_size = trainval_len // (n_folds + 1)
            # Boundaries: [fold_size, 2*fold_size, ..., (n_folds+1)*fold_size]
            self.fold_boundaries = [fold_size * (i + 1) for i in range(n_folds + 1)]
        else:
            self.fold_boundaries = None

    def _evaluate_fold(self, train_end, val_end, scaler_config, aug_config, lookback):
        """
        Evaluate a single train/val split.

        Args:
            train_end: end index of training data
            val_end: end index of validation data
            scaler_config: scaler configuration dict
            aug_config: augmentation configuration dict
            lookback: context length for this evaluation

        Returns:
            mse_per_alpha: (K,) tensor of MSE for each alpha, or None if fold is invalid
        """
        try:
            X_train_w, Y_train, X_val_w, Y_val, _, _, scalers = \
                get_prepared_data(self.data, lookback, self.horizon,
                                  train_end, val_end, scaler_config)
        except ValueError:
            return None

        if X_train_w is None or X_val_w is None:
            return None

        if self.is_batched and not self.pool_series:
            # Batched mode: keep series separate (S, N_s, L)
            # X_train_w shape: (S, N_windows, L)
            # Y_train shape: (S, N_windows, H)

            # Solve with batched solver (each series gets own model)
            Theta = self.solver.solve_batched(
                X_train_w, Y_train, self.alphas,
                scaler=scalers['train'],
                aug_config=aug_config
            )  # (K, S, F, H)

            # Predict per-series (returns CPU tensor to avoid OOM)
            Y_pred = self.solver.predict_batched(
                X_val_w, Theta, scaler=scalers['val']
            )  # (K, S, N_val, H) on CPU

            if isinstance(scalers['val'], GlobalScaler):
                Y_pred = scalers['val'].inv_transform(Y_pred)

            # Compute MSE on CPU to avoid large GPU allocation
            # Y_val shape: (S, N_val, H)
            # Y_pred shape: (K, S, N_val, H)
            diff = Y_pred - Y_val.cpu().unsqueeze(0)  # (K, S, N_val, H)
            mse_per_series = (diff ** 2).mean(dim=(-2, -1))  # (K, S)
            mse_per_alpha = mse_per_series.mean(dim=1)  # (K,) - average across series
        else:
            # Single series mode: concatenate windows
            X_train = X_train_w.reshape(-1, lookback)
            Y_train = Y_train.reshape(-1, Y_train.shape[-1])
            X_val = X_val_w.reshape(-1, lookback)
            Y_val = Y_val.reshape(-1, Y_val.shape[-1])

            # Solve
            Theta = self.solver.solve(
                X_train, Y_train, self.alphas,
                scaler=scalers['train'],
                aug_config=aug_config
            )

            # Predict
            Y_pred = self.solver.predict(X_val, Theta, scaler=scalers['val'])
            if isinstance(scalers['val'], GlobalScaler):
                Y_pred = scalers['val'].inv_transform(Y_pred)
            Y_pred = Y_pred.cpu()

            # Compute MSE per alpha
            diff = Y_pred - Y_val.unsqueeze(0)  # (K, N_window, H)
            mse_per_alpha = (diff ** 2).mean(dim=(-2, -1))  # (K,)

        return mse_per_alpha

    def __call__(self, trial):
        # Suggest lookback using log-scale search (more efficient for context length)
        lookback = trial.suggest_int("lookback", self.min_lookback, self.max_lookback, log=True)

        # Scaler scope (global vs local)
        if self.scaler_scope == "search":
            scaler_scope = trial.suggest_categorical("scaler_scope", ["global", "local"])
        else:
            scaler_scope = self.scaler_scope

        # Scaler method (mean vs robust)
        if self.scaler_method == "search":
            scaler_method = trial.suggest_categorical("scaler_method", ["mean", "robust"])
        else:
            scaler_method = self.scaler_method

        # Local ratio (only relevant for local normalization)
        if scaler_scope == "local":
            if self.fixed_local_ratio is not None:
                local_ratio = self.fixed_local_ratio
            else:
                local_ratio = trial.suggest_float("local_ratio", 1e-3, 1.0, log=True)
        else:
            local_ratio = 1.0  # Not used for global, but needed for config dict

        scaler_config = {
            "method": scaler_method, "scope": scaler_scope,
            "local_ratio": local_ratio, "q_min": 0.25, "q_max": 0.75
        }

        # Augmentation hyperparameters
        if self.fixed_noise_type is not None:
            noise_type = self.fixed_noise_type
        else:
            noise_type = trial.suggest_categorical("noise_type", ["none", "time", "freq"])

        if noise_type == "none":
            sigma = 0.0
        else:
            if self.fixed_aug_sigma is not None:
                sigma = self.fixed_aug_sigma
            else:
                # Log-scale search for sigma (small values matter more)
                sigma = trial.suggest_float("aug_sigma", 1e-3, 0.5, log=True)
        aug_config = {"noise_type": noise_type, "sigma": sigma}

        if self.n_folds == 1:
            # Single split evaluation
            mse_per_alpha = self._evaluate_fold(
                self.split_idx_1, self.split_idx_2, scaler_config, aug_config, lookback)
            if mse_per_alpha is None:
                raise optuna.TrialPruned()
        else:
            # Expanding window k-fold CV
            fold_mses = []
            for fold_idx in range(self.n_folds):
                train_end = self.fold_boundaries[fold_idx]
                val_end = self.fold_boundaries[fold_idx + 1]
                mse = self._evaluate_fold(
                    train_end, val_end, scaler_config, aug_config, lookback)
                if mse is not None:
                    fold_mses.append(mse)

            if not fold_mses:
                raise optuna.TrialPruned()

            # Stack fold MSEs: (n_folds, K)
            fold_mses_tensor = torch.stack(fold_mses)
            mean_mse = fold_mses_tensor.mean(dim=0)  # (K,)

            # Apply fold variance regularization if lambda > 0
            if self.fold_reg_lambda > 0 and len(fold_mses) > 1:
                std_mse = fold_mses_tensor.std(dim=0)  # (K,)
                mse_per_alpha = mean_mse + self.fold_reg_lambda * std_mse
                # Log fold statistics for analysis
                best_mean_idx = mean_mse.argmin()
                trial.set_user_attr("fold_mean", mean_mse[best_mean_idx].item())
                trial.set_user_attr("fold_std", std_mse[best_mean_idx].item())
            else:
                mse_per_alpha = mean_mse

        # Find best alpha from (averaged) MSE
        mse_per_alpha = mse_per_alpha.squeeze()
        best_val_mse, best_idx = torch.min(mse_per_alpha, dim=0)

        trial.set_user_attr("best_alpha", self.alphas[best_idx].item())
        return best_val_mse.item()

    def refit_test(self, best_params, best_alpha_val, use_train_val=False):
        """
        Returns DICTIONARY containing raw predictions/targets for full test set.
        """
        # Get hyperparameters: use fixed values if set, otherwise from best_params
        lookback = best_params.get("lookback", self.min_lookback)

        # Scaler config: use fixed values if set, otherwise from best_params
        if self.scaler_method == "search":
            scaler_method = best_params.get("scaler_method", "mean")
        else:
            scaler_method = self.scaler_method

        if self.scaler_scope == "search":
            scaler_scope = best_params.get("scaler_scope", "local")
        else:
            scaler_scope = self.scaler_scope

        if self.fixed_local_ratio is not None:
            local_ratio = self.fixed_local_ratio
        else:
            local_ratio = best_params.get("local_ratio", 1.0)

        scaler_config = {
            "method": scaler_method, "scope": scaler_scope,
            "local_ratio": local_ratio, "q_min": 0.25, "q_max": 0.75
        }

        # Augmentation config: use fixed values if set, otherwise from best_params
        if self.fixed_noise_type is not None:
            noise_type = self.fixed_noise_type
        else:
            noise_type = best_params.get("noise_type", "none")

        if noise_type == "none":
            sigma = 0.0
        else:
            if self.fixed_aug_sigma is not None:
                sigma = self.fixed_aug_sigma
            else:
                sigma = best_params.get("aug_sigma", 0.0)
        aug_config = {"noise_type": noise_type, "sigma": sigma}

        # Use Train+Val (split_idx_2)
        start = self.split_ends[1] if use_train_val else self.split_starts[1]
        end = self.split_ends[1]
        X_train_w, Y_train, _, _, X_test_w, Y_test, scalers = \
            get_prepared_data(self.data, lookback, self.horizon,
                              start, end, scaler_config)

        S = self.n_series
        N_test = X_test_w.shape[1] if X_test_w.dim() == 3 else X_test_w.shape[0]

        X_train = X_train_w.reshape(-1, lookback)
        Y_train = Y_train.reshape(-1, Y_train.shape[-1])
        X_test = X_test_w.reshape(-1, lookback)
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        single_alpha = torch.tensor([best_alpha_val], device=self.device)

        # Solve with scaler
        Theta = self.solver.solve(
            X_train, Y_train, single_alpha,
            scaler=scalers['test'],
            aug_config=aug_config,
        )

        # Predict with scaler
        Y_pred = self.solver.predict(X_test, Theta, scaler=scalers['test'])
        Y_pred = Y_pred.reshape_as(Y_test)  # (S*N_test, H) or (N_test, H)

        # For GlobalScaler, apply inv_transform to get back to original scale
        if isinstance(scalers['test'], GlobalScaler):
            Y_pred = scalers['test'].inv_transform(Y_pred)
        Y_pred = Y_pred.to(torch.float32).cpu()

        if self.pool_series and self.is_batched:
            # Reshape to per-series: (S*N_test, H) -> (S, N_test, H)
            H = Y_pred.shape[-1]
            Y_pred = Y_pred.reshape(S, N_test, H)
            Y_test_per = Y_test.reshape(S, N_test, H)
            per_series_mse = ((Y_pred - Y_test_per)**2).mean(dim=(-2, -1))  # (S,)
            return {
                'test_mse': per_series_mse.mean().item(),
                'per_series_mse': per_series_mse,  # (S,)
                'raw_preds': Y_pred,  # (S, N_test, H)
            }

        # Calculate simple mean for logging
        mse = ((Y_pred - Y_test)**2).mean().item()
        return {
            'test_mse': mse,
            'raw_preds': Y_pred,
        }
