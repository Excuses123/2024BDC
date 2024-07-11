import os
import time
import torch
import argparse
from data_helper import create_dataloaders, load_test_data
from pathformer import PathFormer
from torch.nn.parallel import DataParallel
from utils import todevice, setup_seed, save_model


def validate(model, eval_data):
    model.eval()
    with torch.no_grad():
        pred_temp, pred_wind = model(eval_data, inference=True)   # (N * S, P, 1)

        temp_var = eval_data['label_temp'].var()
        wind_var = eval_data['label_wind'].var()
        temp_mse = torch.nn.functional.mse_loss(eval_data['label_temp'], pred_temp)
        wind_mse = torch.nn.functional.mse_loss(eval_data['label_wind'], pred_wind)

        mse = temp_mse / temp_var * 10 + wind_mse / wind_var

    model.train()
    return mse, temp_mse, wind_mse


def train(args):
    # 加载训练数据
    train_dataloader = create_dataloaders(args)

    if args.do_eval:
        eval_data = load_test_data(args.data_path, label=True)
        eval_data = todevice(eval_data, args.device)
    else:
        eval_data = None

    model = PathFormer(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.epsilon)
    model.to(args.device)
    if args.device == 'cuda' and args.n_gpu >= 2:
        model = DataParallel(model)

    step = 0
    best_mse = 9999
    start_time = time.time()
    end_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    model.train()
    for epoch in range(1, args.max_epochs + 1):
        for batch in train_dataloader:
            batch = todevice(batch, args.device)
            optimizer.zero_grad()
            loss, temp_loss, wind_loss = model(batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.print_steps == 0:
                now_time = time.time()
                cost = now_time - end_time
                end_time = now_time
                time_per_step = (now_time - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f"训练: Epoch {epoch} step {step} eta {remaining_time}: mse_loss {loss:.4f}, "
                      f"temp_loss {temp_loss.mean():.4f}, wind_loss {wind_loss.mean():.4f}, 耗时 {cost:.3f}s")

            if args.do_eval and step % args.eval_step == 0:
                eval_mse, eval_temp_mse, eval_wind_mse = validate(model, eval_data)
                print(f"\n验证: Epoch {epoch} Step {step}, mse {eval_mse:.4f}, temp_mse {eval_temp_mse:.4f}, wind_mse {eval_wind_mse:.4f}\n")

                if eval_mse < best_mse:
                    best_mse = eval_mse
                    best_temp_mse = eval_temp_mse
                    best_wind_mse = eval_wind_mse
                    best_step = step
                    print(f"保存模型: step {best_step} mse {best_mse:.4f} temp_mse {best_temp_mse:.4f} wind_mse {best_wind_mse:.4f}\n")
                    save_model(args, model, epoch, step, f'{args.model_path}/model.bin')

        if not args.do_eval:
            # 非验证模式，每个epoch保存一次
            save_model(args, model, epoch, step, f'{args.model_path}/model.bin')

    if args.do_eval:
        print(f"best_step {best_step} best_mse {best_mse:.4f} temp_mse {best_temp_mse:.4f} wind_mse {best_wind_mse:.4f}")


# 参数
def cmd_args():
    parser = argparse.ArgumentParser(description="Model Args")

    parser.add_argument("--seed", type=int, default=1024, help="random seed.")

    parser.add_argument('--seq_len', type=int, default=168, help='输入窗口长度')
    parser.add_argument('--pred_len', type=int, default=24, help='预测窗口长度')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--d_model', type=int, default=168, help='dim')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--num_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', type=int, default=0, help='output_attention')

    # for pathformer
    parser.add_argument('--num_nodes', type=int, default=18)
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[[16,12,8],[32,12,8,6,4],[8,6,4,2]])
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='训练轮数')
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

    os.makedirs(args.model_path, exist_ok=True)

    return args


if __name__ == '__main__':

    args = parse_args()

    train(args)
