import os
import random
import numpy as np
import argparse


def cmd_args():
    parser = argparse.ArgumentParser(description="Model Args")
    parser.add_argument("--seed", type=int, default=1024, help="random seed.")
    parser.add_argument('--val_rate', type=float, default=0.25, help='验证比例')
    parser.add_argument("--num_sta", type=int, default=300, help="验证的station个数")
    parser.add_argument("--data_path", type=str, default='./eval_data')

    return parser.parse_args()


if __name__ == '__main__':

    args = cmd_args()
    random.seed(args.seed)

    print('验证集比例: ', args.val_rate)
    print('验证集站点个数: ', args.num_sta)

    os.makedirs(args.data_path, exist_ok=True)

    a = np.load("./data/global_data.npy")
    b = np.load("./data/temp.npy")
    c = np.load("./data/wind.npy")

    n = int(a.shape[0] * (1 - args.val_rate))
    total_time, num_station = b.shape[:2]

    # 随机71个窗口
    win = np.array([random.randint(0, total_time - n * 3 - 168 - 72) for _ in range(71)])
    np.save(os.path.join(args.data_path, 'sample_win.npy'), win)
    # 随机240个station进行验证
    sta = np.array([random.randint(0, num_station) for _ in range(240)])
    np.save(os.path.join(args.data_path, 'sample_station.npy'), sta)

    a_train = a[:n]
    b_train = b[:n*3]
    c_train = c[:n*3]

    a_val = a[n:]       # [1462, 4, 9, 3850]
    b_val = b[n*3:]     # [4386, 3850, 1]
    c_val = c[n*3:]     # [4386, 3850, 1]

    a_val = np.concatenate([a_val[None, i//3: i//3+56+24, :, :, sta] for i in win], axis=0)  # [71, 56+24, 4, 9, 240]
    b_val = np.concatenate([b_val[None, i: i+168+72, sta, :] for i in win], axis=0)  # [71, 168+72, 240, 1]
    c_val = np.concatenate([c_val[None, i: i+168+72, sta, :] for i in win], axis=0)  # [71, 168+72, 240, 1]

    print(a_train.shape)
    print(b_train.shape)
    print(c_train.shape)

    print(a_val.shape)
    print(b_val.shape)
    print(c_val.shape)

    np.save(os.path.join(args.data_path, 'global_data.npy'), a_train)
    np.save(os.path.join(args.data_path, 'temp.npy'), b_train)
    np.save(os.path.join(args.data_path, 'wind.npy'), c_train)

    np.save(os.path.join(args.data_path, 'cenn_data.npy'), a_val[:, :56, :, :, :])
    np.save(os.path.join(args.data_path, 'temp_lookback.npy'), b_val[:, :168, :, :])
    np.save(os.path.join(args.data_path, 'wind_lookback.npy'), c_val[:, :168, :, :])

    np.save(os.path.join(args.data_path, 'temp_lookback_label.npy'), b_val[:, 168:, :, :])
    np.save(os.path.join(args.data_path, 'wind_lookback_label.npy'), c_val[:, 168:, :, :])

