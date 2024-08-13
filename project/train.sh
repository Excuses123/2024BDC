# 联合训练(10折)
CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./dataset/global' \
    --model_path='./dataset/other/all_seed_10/seed_1' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=1

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_2' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=2

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_3' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=3

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_4' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=4

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_5' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=5

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_6' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=6

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_7' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=7

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_8' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=8

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_9' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=9

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/all_seed_10/seed_10' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --seed=10


# 训练temp模型(5折)
CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/temp_seed_5/seed_1' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='temp' \
    --seed=11

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/temp_seed_5/seed_2' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='temp' \
    --seed=12

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/temp_seed_5/seed_3' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='temp' \
    --seed=13

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/temp_seed_5/seed_4' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='temp' \
    --seed=14

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/temp_seed_5/seed_5' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='temp' \
    --seed=15

# 训练wind模型(5折)
CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/wind_seed_5/seed_1' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='wind' \
    --seed=11

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/wind_seed_5/seed_2' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='wind' \
    --seed=12

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/wind_seed_5/seed_3' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='wind' \
    --seed=13

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/wind_seed_5/seed_4' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='wind' \
    --seed=14

CUDA_VISIBLE_DEVICES=0 python project/run_itransformer.py \
    --data_path='./data/global' \
    --model_path='./dataset/other/wind_seed_5/seed_5' \
    --batch_size=10240 \
    --learning_rate=3e-4 \
    --max_epochs=2 \
    --num_workers=7 \
    --pred_var='wind' \
    --seed=15


