# 线上成绩: 0.9442316500614263
CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_1' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=1

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_2' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=5 \
    --seed=2

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_3' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=3

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_4' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=5 \
    --seed=4

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_5' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=5

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_6' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=5 \
    --seed=6

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_7' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=7

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_8' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=5 \
    --seed=8

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_9' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=6 \
    --seed=9

CUDA_VISIBLE_DEVICES=0,2 python project/run_itransformer.py \
    --data_path='./data' \
    --model_path='./project/checkpoint/seed_10' \
    --batch_size=10240 \
    --learning_rate=5e-4 \
    --max_epochs=2 \
    --num_workers=5 \
    --seed=10