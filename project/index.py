import os
import torch
import numpy as np
from itransformer import ITransformer
from fredformer import FredFormer
from utils import DictToClass, todevice
from data_helper import load_test_data
from sklearn.metrics import mean_squared_error

bsz = 2048
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def invoke(inputs):
    save_path = '/home/mw/project'

    data = load_test_data(inputs)['x']  # (N * S, L, -1) # 加载测试数据

    # 待推理的模型路径&融合权重
    models = {
        "./checkpoint/all_seed_10/seed_1/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_2/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_3/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_4/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_5/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_6/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_7/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_8/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_9/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/all_seed_10/seed_10/all_0": {'ind': [0, 1], 'weight': 1},

        "./checkpoint/temp_seed_5/seed_1/temp_0": {'ind': 0, 'weight': 1},
        "./checkpoint/wind_seed_5/seed_1/wind_0": {'ind': 1, 'weight': 1},
        "./checkpoint/temp_seed_5/seed_2/temp_0": {'ind': 0, 'weight': 1},
        "./checkpoint/wind_seed_5/seed_2/wind_0": {'ind': 1, 'weight': 1},
        "./checkpoint/temp_seed_5/seed_3/temp_0": {'ind': 0, 'weight': 1},
        "./checkpoint/wind_seed_5/seed_3/wind_0": {'ind': 1, 'weight': 1},
        "./checkpoint/temp_seed_5/seed_4/temp_0": {'ind': 0, 'weight': 1},
        "./checkpoint/wind_seed_5/seed_4/wind_0": {'ind': 1, 'weight': 1},
        "./checkpoint/temp_seed_5/seed_5/temp_0": {'ind': 0, 'weight': 1},
        "./checkpoint/wind_seed_5/seed_5/wind_0": {'ind': 1, 'weight': 1},
    }

    result_temp, result_wind = 0, 0
    num_temp, num_wind = 0, 0
    for model_path, info in models.items():
        preds = inference(model_path, data)  # 遍历模型集合进行推理，模型同时输出temp和wind，根据ind中的0,1进行切片获取对应结果
        if info['ind'] == 0:
            result_temp += preds[info['ind']] * info['weight']
            num_temp += info['weight']
        elif info['ind'] == 1:
            result_wind += preds[info['ind']] * info['weight']
            num_wind += info['weight']
        else:
            result_temp += preds[info['ind'][0]] * info['weight']
            num_temp += info['weight']
            result_wind += preds[info['ind'][1]] * info['weight']
            num_wind += info['weight']

    result_temp = result_temp / num_temp    # temp根据权重加权取平均
    result_wind = result_wind / num_wind    # wind根据权重加权取平均

    np.save(os.path.join(save_path, "temp_predict.npy"), result_temp)
    np.save(os.path.join(save_path, "wind_predict.npy"), result_wind)


def invoke_eval(inputs, models):
    """
    离线验证用
    """
    data = load_test_data(inputs)['x']  # (N * S, L, -1)

    result_temp, result_wind = 0, 0
    num_temp, num_wind = 0, 0
    for model_path, info in models.items():
        preds = inference(model_path, data)
        if info['ind'] == 0:
            result_temp += preds[info['ind']] * info['weight']
            num_temp += info['weight']
        elif info['ind'] == 1:
            result_wind += preds[info['ind']] * info['weight']
            num_wind += info['weight']
        else:
            result_temp += preds[info['ind'][0]] * info['weight']
            num_temp += info['weight']
            result_wind += preds[info['ind'][1]] * info['weight']
            num_wind += info['weight']

    result_temp = result_temp / num_temp
    result_wind = result_wind / num_wind

    label_temp = np.load(os.path.join(inputs, "temp_lookback_label.npy"))
    label_wind = np.load(os.path.join(inputs, "wind_lookback_label.npy"))

    temp_var = np.var(label_temp)
    wind_var = np.var(label_wind)
    mse_temp = mean_squared_error(result_temp.flatten(), label_temp.flatten())
    mse_wind = mean_squared_error(result_wind.flatten(), label_wind.flatten())
    mse = mse_temp / temp_var * 10 + mse_wind / wind_var

    return mse, mse_temp, mse_wind


def inference(model_path, data):
    """
    模型推理
    """
    checkpoint = torch.load(f"{model_path}/model.bin", map_location='cpu')  # 加载模型文件
    args = DictToClass(checkpoint['args'])  # 参数格式转换

    # 根据模型名称初始化模型
    if args.model_name == 'itransformer':
        model = ITransformer(args).to(device)
    elif args.model_name == 'fredformer':
        model = FredFormer(args).to(device)
    else:
        raise Exception(f"model_name: {args.model_name} is not supported!")

    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
    model.eval()

    step = data.shape[0] // bsz + 1
    pred_temp, pred_wind = [], []
    with torch.no_grad():
        for i in range(step):
            batch = {'x': todevice(data[i * bsz: (i+1) * bsz, :, :], device)}
            p_temp, p_wind = model(batch, inference=True)  # [batch, pred_len, 1]  分批预测

            pred_temp.append(p_temp.detach().cpu().numpy())  # (N * S, P, 1)
            pred_wind.append(p_wind.detach().cpu().numpy())  # (N * S, P, 1)

    pred_temp = np.concatenate(pred_temp, axis=0)  # temp预测结果聚合
    pred_wind = np.concatenate(pred_wind, axis=0)  # wind预测结果聚合
    pred_wind = np.abs(pred_wind)  # 后处理：风速不为负

    P = pred_temp.shape[1]

    pred_temp = pred_temp.reshape((-1, 120, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    pred_wind = pred_wind.reshape((-1, 120, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)

    return pred_temp, pred_wind


