import random
import numpy as np

a = np.load("./data/global_data.npy")
b = np.load("./data/temp.npy")
c = np.load("./data/wind.npy")

n = int(a.shape[0] * 0.75)
total_time, num_station = b.shape[:2]

# 随机71个窗口
win = np.array([random.randint(0, total_time - n * 3 - 168 - 24) for _ in range(71)])
# 随机60个station
sta = np.array([random.randint(0, num_station) for _ in range(60)])

a_train = a[:n]
b_train = b[:n*3]
c_train = c[:n*3]

a_val = a[n:]       # [1462, 4, 9, 3850]
b_val = b[n*3:]     # [4386, 3850, 1]
c_val = c[n*3:]     # [4386, 3850, 1]

a_val = np.concatenate([a_val[None, i//3 : i//3+56+8, :, :, sta] for i in win], axis=0) # [71, 56+8, 4, 9, 60]
b_val = np.concatenate([b_val[None, i: i+168+24, sta, :] for i in win], axis=0)  # [71, 168+24, 60, 1]
c_val = np.concatenate([c_val[None, i: i+168+24, sta, :] for i in win], axis=0)  # [71, 168+24, 60, 1]

np.save('./test_data/global_data.npy', a_train)
np.save('./test_data/temp.npy', b_train)
np.save('./test_data/wind.npy', c_train)

np.save('./test_data/cenn_data.npy', a_val[:, :56, :, :, :])
np.save('./test_data/temp_lookback.npy', b_val[:, :168, :, :])
np.save('./test_data/wind_lookback.npy', c_val[:, :168, :, :])

np.save('./test_data/temp_lookback_label.npy', b_val[:, 168:, :, :])
np.save('./test_data/wind_lookback_label.npy', c_val[:, 168:, :, :])

