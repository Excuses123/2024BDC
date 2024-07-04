import torch
import random
import numpy as np


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def todevice(inputs, device):
    if isinstance(inputs, dict):
        return {k: todevice(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, (tuple, list)):
        return (todevice(v, device) for v in inputs)
    elif isinstance(inputs, torch.Tensor):
        kwargs = {"device": device}
        return inputs.to(**kwargs)


class DictToClass(object):
    def __init__(self, _obj=None):
        if _obj:
            self.__dict__.update(_obj)


def save_model(args, model, epoch, step, batch, save_path):
    model_dict = {
        "args": {k: v for k, v in args.__dict__.items()},
        "epoch": epoch,
        "step": step,
        "sample_batch": batch
    }
    if isinstance(model, (torch.nn.parallel.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
        model_dict['model_state_dict'] = model.module.state_dict()
    else:
        model_dict['model_state_dict'] = model.state_dict()

    torch.save(model_dict, save_path)


