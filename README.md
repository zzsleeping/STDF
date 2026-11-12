# STGF: Long-Term Time Series Forecasting with Large Language Models via Statistical Teacher Guidance

STGF guides a LoRA-adapted GPT-2 forecaster with forecasts from an independently
fitted and frozen **statistical teacher**. The teacher is a grouped Ridge forecaster
with searched lookbacks (up to 2,048 steps) plus a gradient-boosted residual
correction. It is fitted on training data only, frozen before student training and
**never executed at inference**: the deployed student forecasts from a 96-step input.

```
L = MAE(S, Y) + λ · MSE(S, sg(T)) + β · MAE(S, sg(T)),   λ = β = 0.4
```

`S` is the student forecast, `Y` the observed future and `T` the precomputed
teacher forecast; the two guidance terms cover the training windows whose origin
has enough history for the teacher.

This repository is built on the code base of [CALF](https://github.com/Hank0626/CALF)
and keeps its layout.

## Structure

```text
run.py                               entry point (arguments, seeds, train/test)
pca.py                               builds the frozen PCA memory wte_pca_500.pt
exp/exp_long_term_forecasting.py     teacher preparation, training, validation, testing
data_provider/                       chronological splits, standardization, windows
models/STGF.py                       student: numerical encoder, PCA memory interface,
                                     GPT-2 (6 blocks, LoRA), prediction head
models/GPT2_arch.py                  GPT-2 wrapper from CALF
teacher/ridge_teacher.py             history-adaptive grouped Ridge base forecaster
teacher/residual.py                  CatBoost residual correction and its selection
teacher/build.py                     offline teacher construction -> guidance cache
teacher/searchcast_ridge.py          Ridge solver/search from SearchCast (Apache-2.0)
utils/losses.py                      guidance loss
utils/augmentation.py                time-scale augmentation (zero-shot)
scripts/long_term_forecasting/       ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic
scripts/few_shot_forecasting/        10% few-shot on ETT
scripts/zero_shot_forecasting/       ETTh1/ETTh2 -> ETTm1/ETTm2
```

## Prerequisites

```bash
pip install -r requirements.txt
```

Datasets follow the usual layout (`datasets/ETT/ETTh1.csv`, `datasets/weather/weather.csv`,
`datasets/electricity/electricity.csv`, `datasets/traffic/traffic.csv`).

Build the frozen PCA memory once (500 components of the GPT-2 word embeddings):

```bash
python pca.py
```

## Long-term forecasting

```bash
sh scripts/long_term_forecasting/ETTh1.sh
```

Each run proceeds in two stages.

1. **Teacher** (offline, once per dataset and horizon). `teacher/build.py` searches and
   fits the Ridge base forecaster, produces chronological out-of-fold forecasts from
   prefixes of 40%, 60% and 80% of the training interval, selects the CatBoost depth and
   correction weight η by forward validation (η = 0 is the fallback), and writes the
   teacher forecasts of all training windows to `teacher_cache/<data>_sl96_pl<H>/`.
   Later runs and seeds reuse this cache. `metadata.json` there records the selected
   units, η and the teacher's standalone validation/test errors.
2. **Student.** The student is trained with the guidance loss, the checkpoint is selected
   by validation MSE, and test MSE/MAE are appended to `result_long_term_forecast.txt`.

`--itr 3 --seed 2021` trains seeds 2021, 2022 and 2023, as in the paper.

Useful switches:

| argument | meaning |
|---|---|
| `--use_teacher 0` | supervised only (observed targets) |
| `--con_w`, `--rob_w` | λ and β of the guidance loss |
| `--teacher_residual 0` | base Ridge forecaster only, much cheaper to build |
| `--teacher_max_lookback 96` | short-history teacher |
| `--teacher_device cuda` | run the Ridge solver on GPU |
| `--use_memory 0` | remove the PCA memory interface |
| `--seq_len 336` | longer student input |

Teacher construction is the main extra cost: it is cheap on ETT but takes hours on
Electricity and Traffic, most of it for the out-of-fold forecasts and residual correction.

## Few-shot forecasting

```bash
sh scripts/few_shot_forecasting/ETTh1.sh
```

With `--percent 10` only the first `L + ⌊0.1 (N − L)⌋` steps of the training split are
used (the data protocol of CALF). The teacher is refitted within this budget; the prefix
is too short for out-of-fold folds, so it is the base forecaster only, with its
configuration selected on the validation split. Inputs are standardized with the prefix
and errors are reported in the units of the full training split.

## Zero-shot forecasting

```bash
sh scripts/zero_shot_forecasting/ETTh1.sh
```

The student is trained on the source dataset with time-scale augmentation
(`--time_scale_aug 1`), which resamples histories, targets and cached teacher forecasts
together; model selection uses source validation data only. It is then evaluated on the
target datasets (`--target_data ETTm1,ETTm2`) without any update.

## Testing a trained model

```bash
python run.py --is_training 0 --model_id ETTh1_STGF_96_96 --data ETTh1 \
  --root_path ./datasets/ETT/ --data_path ETTh1.csv --pred_len 96
```

The setting string must match the training run (`model_id`, `seed`, `des`, ...).
No teacher is needed.

## Acknowledgement

- [CALF](https://github.com/Hank0626/CALF): code base, numerical interface and GPT-2 wrapper.
- [SearchCast](https://github.com/SakanaAI/SearchCast): Ridge search and solver used by the teacher (`teacher/searchcast_ridge.py`, Apache-2.0).
