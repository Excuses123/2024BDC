# 联合训练(10折)
CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_1' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=1

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_2' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=2

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_3' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=3

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_4' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=4

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_5' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=5

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_6' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=6

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_7' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=7

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_8' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=8

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_9' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=9

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/all_seed_10/seed_10' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=10

# 训练temp模型(5折)
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

# 训练wind模型(5折)
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


