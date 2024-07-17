import os
import random
import numpy as np

era5 = np.load("./data/global_data.npy")  # (5848, 4, 9, 3850)
temp = np.load("./data/temp.npy")         # (17544, 3850, 1)
wind = np.load("./data/wind.npy")         # (17544, 3850, 1)
total_time, num_station, _ = temp.shape

for fold in range(5):
    print(f"split data of fold: {fold}")
    os.makedirs(f"./fold_data/fold_{fold}", exist_ok=True)

    # 随机15%的station作文验证
    n = int(num_station * 0.15)
    sta = random.sample(range(num_station), n)
    sta_train = np.array([i for i in range(num_station) if i not in sta])
    np.save('./test_data/sample_station.npy', sta)

    # 随机71个窗口
    win = np.array([random.randint(0, total_time - 168 - 24) for _ in range(71)])
    np.save('./test_data/sample_win.npy', win)

    train_era5 = era5[:, :, :, sta_train]
    train_temp = temp[:, sta_train, :]
    train_wind = wind[:, sta_train, :]
    print(f"train_era5 shape: {train_era5.shape}")
    print(f"train_temp shape: {train_temp.shape}")
    print(f"train_wind shape: {train_wind.shape}")

    val_era5 = np.concatenate([era5[None, i//3: i//3+56+8, :, :, sta] for i in win], axis=0) # [71, 56+8, 4, 9, 60]
    val_temp = np.concatenate([temp[None, i: i+168+24, sta, :] for i in win], axis=0)  # [71, 168+24, 60, 1]
    val_wind = np.concatenate([wind[None, i: i+168+24, sta, :] for i in win], axis=0)  # [71, 168+24, 60, 1]
    print(f"val_era5 shape: {val_era5.shape}")
    print(f"val_temp shape: {val_temp.shape}")
    print(f"val_wind shape: {val_wind.shape}")

    np.save(f'./fold_data/fold_{fold}/global_data.npy', train_era5)
    np.save(f'./fold_data/fold_{fold}/temp.npy', train_temp)
    np.save(f'./fold_data/fold_{fold}/wind.npy', train_wind)

    np.save(f'./fold_data/fold_{fold}/cenn_data.npy', val_era5[:, :56, :, :, :])
    np.save(f'./fold_data/fold_{fold}/temp_lookback.npy', val_temp[:, :168, :, :])
    np.save(f'./fold_data/fold_{fold}/wind_lookback.npy', val_wind[:, :168, :, :])

    np.save(f'./fold_data/fold_{fold}/temp_lookback_label.npy', val_temp[:, 168:, :, :])
    np.save(f'./fold_data/fold_{fold}/wind_lookback_label.npy', val_wind[:, 168:, :, :])

