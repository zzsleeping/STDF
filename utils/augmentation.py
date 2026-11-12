"""Time-scale augmentation used for source-domain training in zero-shot transfer.

One scale q in {1, 1/2, 1/4} (probabilities 1/2, 1/4, 1/4) is drawn for every
training minibatch.  The history, the observed future and the cached teacher
forecast are linearly resampled on a grid of step q anchored at the last
observed value, so all three stay aligned.  The history grid reads the history
only; q = 1 is the identity.  No target-domain data is involved.
"""
import hashlib

import torch

SCALES = (1.0, 1.0, 0.5, 0.25)


def choose_scale(seed, epoch, batch_index):
    """Deterministic in (seed, epoch, batch): independent of every global RNG."""
    payload = 'stgf_zeromultiscale_resample_v1:%d:%d:%d' % (seed, epoch, batch_index)
    return SCALES[hashlib.sha256(payload.encode('ascii')).digest()[0] & 3]


def _interpolate(sequence, positions):
    left = torch.floor(positions).long()
    right = torch.clamp(left + 1, max=sequence.shape[1] - 1)
    alpha = (positions - left).to(sequence.dtype).view(1, -1, 1)
    a, b = sequence.index_select(1, left), sequence.index_select(1, right)
    return a + (b - a) * alpha


def time_scale(x, y, teacher, valid, q):
    """x: [B, L, C]; y, teacher: [B, H, C]; valid: bool [B]. Returns resampled copies."""
    if teacher is not None:
        teacher = torch.where(valid.view(-1, 1, 1), teacher, torch.zeros_like(teacher))
    if q == 1.0:
        return x, y, teacher
    L, H = x.shape[1], y.shape[1]
    history = (L - 1) - q * torch.arange(L - 1, -1, -1, dtype=torch.float64, device=x.device)
    future = q * torch.arange(1, H + 1, dtype=torch.float64, device=x.device)
    anchor = x[:, -1:, :]
    xq = _interpolate(x, history)
    yq = _interpolate(torch.cat((anchor, y), dim=1), future)
    tq = None
    if teacher is not None:
        tq = _interpolate(torch.cat((anchor, teacher), dim=1), future)
        tq = torch.where(valid.view(-1, 1, 1), tq, torch.zeros_like(tq))
    return xq, yq, tq
