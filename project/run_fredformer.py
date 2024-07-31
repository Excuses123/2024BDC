import os
import time
import torch
import argparse
from data_helper import create_dataloaders, load_test_data
from fredformer import FredFormer
from torch.nn.parallel import DataParallel
from utils import todevice, setup_seed, save_model


def validate(model, eval_data, device):
    bsz = 2048
    x = eval_data['x']
    step = x.shape[0] // bsz + 1
    pred_temp, pred_wind = [], []
    with torch.no_grad():
        for i in range(step):
            batch = todevice({'x': x[i * bsz: (i + 1) * bsz, :, :]}, device)
            p_temp, p_wind = model(batch, inference=True)  # [bsz, pred_len, 1]

            pred_temp.append(p_temp.detach().cpu())
            pred_wind.append(p_wind.detach().cpu())

    pred_temp = torch.cat(pred_temp, dim=0)
    pred_wind = torch.cat(pred_wind, dim=0)
    pred_wind = torch.abs(pred_wind)  # 后处理：风速不为负

    temp_var = eval_data['label_temp'].var()
    wind_var = eval_data['label_wind'].var()
    temp_mse = torch.nn.functional.mse_loss(eval_data['label_temp'], pred_temp) / temp_var
    wind_mse = torch.nn.functional.mse_loss(eval_data['label_wind'], pred_wind) / wind_var
    mse = temp_mse * 10 + wind_mse

    model.train()
    return mse, temp_mse, wind_mse


def train(args):
    train_dataloader = create_dataloaders(args)

    if args.do_eval:
        eval_data = load_test_data(args.data_path, label=True)
    else:
        eval_data = None

    model = FredFormer(args)
    parameters = [{'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, eps=args.epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    model.to(args.device)
    if args.device == 'cuda' and args.n_gpu >= 2:
        model = DataParallel(model)

    step = 0
    best_mse = 9999
    stop_train = False
    start_time = time.time()
    end_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    model.train()
    for epoch in range(1, args.max_epochs + 1):
        for batch in train_dataloader:
            batch = todevice(batch, args.device)
            optimizer.zero_grad()
            all_loss, temp_loss, wind_loss = model(batch)
            if args.pred_var == 'all':
                loss = all_loss.mean()
            elif args.pred_var == 'temp':
                loss = temp_loss.mean()
            else:
                loss = wind_loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]

            step += 1
            if step % args.print_steps == 0:
                now_time = time.time()
                cost = now_time - end_time
                end_time = now_time
                time_per_step = (now_time - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f"训练: Epoch {epoch} step {step} cur_lr {cur_lr:.5f} eta {remaining_time}: loss {loss:.4f}, all_loss {all_loss.mean():.4f}, "
                      f"temp_loss {temp_loss.mean():.4f}, wind_loss {wind_loss.mean():.4f}, 耗时 {cost:.3f}s")

            if args.do_eval and step % args.eval_step == 0:
                eval_all_mse, eval_temp_mse, eval_wind_mse = validate(model, eval_data, args.device)
                print(f"\n验证: Epoch {epoch} Step {step}, all_mse {eval_all_mse:.4f}, temp_mse {eval_temp_mse:.4f}, wind_mse {eval_wind_mse:.4f}\n")

                if args.pred_var == 'all':
                    eval_mse = eval_all_mse
                elif args.pred_var == 'temp':
                    eval_mse = eval_temp_mse
                else:
                    eval_mse = eval_wind_mse

                if eval_mse < best_mse:
                    best_mse = eval_mse
                    best_all_mse = eval_all_mse
                    best_temp_mse = eval_temp_mse
                    best_wind_mse = eval_wind_mse
                    best_step = step
                    print(f"保存模型: step {best_step} mse {best_mse:.4f} all_mse {best_all_mse:.4f} temp_mse {best_temp_mse:.4f} wind_mse {best_wind_mse:.4f}\n")
                    save_model(args, model, epoch, step, f'{args.model_path}/model.bin')

            if step >= args.max_steps:  # 达到设定的最大步数，保存模型并退出当前训练
                save_model(args, model, epoch, step, f'{args.model_path}/model.bin')
                stop_train = True
                break

        if stop_train:  # 退出一级循环
            break

        if not args.do_eval:  # 非验证模式，每个epoch保存一次模型
            save_model(args, model, epoch, step, f'{args.model_path}/model.bin')

    if args.do_eval:
        print(f"best_step {best_step} best_mse {best_mse:.4f} all_mse {best_all_mse:.4f} temp_mse {best_temp_mse:.4f} wind_mse {best_wind_mse:.4f}")


# 参数
def cmd_args():
    parser = argparse.ArgumentParser(description="Model Args")

    parser.add_argument("--seed", type=int, default=1024, help="random seed.")

    parser.add_argument('--model_name', type=str, default='fredformer', help='模型')
    parser.add_argument('--seq_len', type=int, default=168, help='输入窗口长度')
    parser.add_argument('--pred_len', type=int, default=72, help='预测窗口长度')
    parser.add_argument('--pred_var', type=str, default='all', help='预测哪个变量')
    parser.add_argument('--outlier_strategy', type=int, default=0,
                        help='异常值处理策略, 0: 不做处理, 1: 删除单变量的异常值, 2: 同时删除两个变量的异常值')
    parser.add_argument('--sample_rate', type=float, default=0.8, help='训练数据比例')

    # for fredformer
    parser.add_argument('--enc_in', type=int, default=24, help='encoder input size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')
    parser.add_argument('--patch_len', type=int, default=24, help='patch length')
    parser.add_argument('--stride', type=int, default=12, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

    parser.add_argument('--ablation', type=int, default=0)  # ablation study 012.
    parser.add_argument('--cf_dim', type=int, default=128)  # feature dimension
    parser.add_argument('--cf_drop', type=float, default=0.1)  # dropout
    parser.add_argument('--cf_depth', type=int, default=2)  # Transformer layer
    parser.add_argument('--cf_heads', type=int, default=4)  # number of multi-heads
    parser.add_argument('--cf_mlp', type=int, default=96)  # ff dimension
    parser.add_argument('--cf_head_dim', type=int, default=16)  # dimension for single head
    parser.add_argument('--d_model', type=int, default=64, help='dim')

    parser.add_argument('--use_nys', type=int, default=0)    #use nystrom

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--max_steps', type=int, default=50000, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=32, help='batch大小')
    parser.add_argument('--print_steps', type=int, default=100, help="多少步打印一次损失")
    parser.add_argument('--eval_step', type=int, default=1000, help="多少步进行一次验证")
    parser.add_argument('--eval_epoch', type=int, default=1, help="多少轮进行一次验证")
    parser.add_argument('--learning_rate', default=0.001, type=float, help='初始的学习率')
    parser.add_argument("--epsilon", default=1e-20, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="学习率衰减权重")
    parser.add_argument('--prefetch', default=16, type=int, help="训练时预加载的数据条数，加快训练速度")
    parser.add_argument('--num_workers', default=1, type=int, help="加载数据的进程数")

    # ========================= path Configs ==========================
    parser.add_argument("--data_path", type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./checkpoint')
    parser.add_argument('--out_path', type=str, default='./output')
    parser.add_argument('--do_eval', default=0, type=int, help="是否验证")

    return parser.parse_args()


def parse_args():
    args = cmd_args()

    setup_seed(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()

    args.model_path = os.path.join(args.model_path, f"{args.pred_var}_{args.outlier_strategy}")
    os.makedirs(args.model_path, exist_ok=True)

    return args


if __name__ == '__main__':

    t1 = time.time()

    args = parse_args()

    train(args)

    print(f"耗时: {time.time() - t1} s")

