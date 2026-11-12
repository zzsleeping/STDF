export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

seq_len=96
model=STGF

for pred_len in 96 192 336 720
do

python run.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id electricity_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 32 \
    --eval_batch_size 8 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --patience 15 \
    --tmax 10 \
    --gpt_layers 6 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --con_w 0.4 \
    --rob_w 0.4 \
    --seed 2021 \
    --itr 3 \
    --des long_term \
    --model $model

echo '====================================================================================================================='
done
