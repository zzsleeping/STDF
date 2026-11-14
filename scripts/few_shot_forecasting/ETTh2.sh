export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

seq_len=96
model=STDF
percent=10

for pred_len in 96 192 336 720
do

python run.py \
    --root_path ./datasets/ETT/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh2_$model'_'$seq_len'_'$pred_len'_'$percent \
    --data ETTh2 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --patience 5 \
    --tmax 20 \
    --gpt_layers 6 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --con_w 0.4 \
    --rob_w 0.4 \
    --seed 2021 \
    --itr 3 \
    --des few_shot \
    --model $model \
    --percent $percent

echo '====================================================================================================================='
done
