# temp
CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/temp_seed_5/seed_1' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='temp' \
    --outlier_strategy=0 \
    --seed=1

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/temp_seed_5/seed_2' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='temp' \
    --outlier_strategy=0 \
    --seed=2

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/temp_seed_5/seed_3' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='temp' \
    --outlier_strategy=0 \
    --seed=3

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/temp_seed_5/seed_4' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='temp' \
    --outlier_strategy=0 \
    --seed=4

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/temp_seed_5/seed_5' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='temp' \
    --outlier_strategy=0 \
    --seed=5

# wind
CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/wind_seed_5/seed_1' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='wind' \
    --outlier_strategy=0 \
    --seed=1

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/wind_seed_5/seed_2' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='wind' \
    --outlier_strategy=0 \
    --seed=2

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/wind_seed_5/seed_3' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='wind' \
    --outlier_strategy=0 \
    --seed=3

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/wind_seed_5/seed_4' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='wind' \
    --outlier_strategy=0 \
    --seed=4

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/wind_seed_5/seed_5' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --pred_var='wind' \
    --outlier_strategy=0 \
    --seed=5


