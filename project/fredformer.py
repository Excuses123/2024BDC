import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


class FredFormer(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2406.09009v4
    Github: https://github.com/chenzRG/Fredformer

    """
    def __init__(self, args):
        super(FredFormer, self).__init__()
        self.args = args
        output = 0
        self.model = Fredformer_backbone(ablation=args.ablation, use_nys=args.use_nys, output=output, c_in=args.enc_in,
                                         context_window=args.seq_len, target_window=args.pred_len,
                                         patch_len=args.patch_len, stride=args.stride, d_model=args.d_model,
                                         head_dropout=args.dropout, padding_patch=args.padding_patch,
                                         individual=args.individual, revin=args.revin, affine=args.affine,
                                         subtract_last=args.subtract_last, cf_dim=args.cf_dim, cf_depth=args.cf_depth,
                                         cf_heads=args.cf_heads, cf_mlp=args.cf_mlp, cf_head_dim=args.cf_head_dim,
                                         cf_drop=args.cf_drop)

    def forward(self, inputs, inference=False):
        """
        x: [batch, seq_len, feat]
        """
        x = inputs['x']
        x = x.permute(0, 2, 1)  # [batch, feat, seq_len]
        x = self.model(x)
        x = x.permute(0, 2, 1)  # [batch, pred_len, feat]

        pred_temp = x[:, -self.args.pred_len:, 0:1]  # [batch, pred_len, 1]
        pred_wind = x[:, -self.args.pred_len:, 1:2]  # [batch, pred_len, 1]

        if inference:
            return pred_temp, pred_wind
        else:
            temp_loss = F.mse_loss(pred_temp, inputs['label_temp']) / inputs['label_temp'].var().detach()
            wind_loss = F.mse_loss(pred_wind, inputs['label_wind']) / inputs['label_wind'].var().detach()
            loss = temp_loss * 10 + wind_loss
            return loss, temp_loss, wind_loss


class Fredformer_backbone(nn.Module):
    def __init__(self, ablation: int, use_nys: int, output: int, cf_dim: int,
                 cf_depth: int, cf_heads: int, cf_mlp: int, cf_head_dim: int, cf_drop: float, c_in: int,
                 context_window: int, target_window: int, patch_len: int, stride: int, d_model: int,
                 head_dropout=0, padding_patch=None, individual=False, revin=True, affine=True, subtract_last=False):

        super().__init__()
        self.use_nys = use_nys
        self.ablation = ablation
        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.output = output
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow = target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len) / stride + 1)
        self.norm = nn.LayerNorm(patch_len)
        # print("depth=",cf_depth)
        # Backbone
        self.re_attn = True
        # if self.use_nys == 0:
        self.fre_transformer = Trans_C(dim=cf_dim, depth=cf_depth, heads=cf_heads, mlp_dim=cf_mlp,
                                       dim_head=cf_head_dim, dropout=cf_drop, patch_dim=patch_len * 2,
                                       horizon=self.horizon * 2, d_model=d_model * 2)
        # else:
        #     self.fre_transformer = Trans_C_nys(dim=cf_dim, depth=cf_depth, heads=cf_heads, mlp_dim=cf_mlp,
        #                                        dim_head=cf_head_dim, dropout=cf_drop, patch_dim=patch_len * 2,
        #                                        horizon=self.horizon * 2, d_model=d_model * 2)

        # Head
        self.head_nf_f = d_model * 2 * patch_num  # self.horizon * patch_num#patch_len * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window,
                                    head_dropout=head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window,
                                    head_dropout=head_dropout)

        self.ircom = nn.Linear(self.targetwindow * 2, self.targetwindow)
        self.rfftlayer = nn.Linear(self.targetwindow * 2 - 2, self.targetwindow)
        self.final = nn.Linear(self.targetwindow * 2, self.targetwindow)

        # break up R&I:
        self.get_r = nn.Linear(d_model * 2, d_model * 2)
        self.get_i = nn.Linear(d_model * 2, d_model * 2)
        self.output1 = nn.Linear(target_window, target_window)

        # ablation
        self.input = nn.Linear(c_in, patch_len * 2)
        self.outpt = nn.Linear(d_model * 2, c_in)
        self.abfinal = nn.Linear(patch_len * patch_num, target_window)

    def forward(self, z):  # z: [bs x nvars x seq_len]

        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len,
                       step=self.stride)  # z2: [bs x nvars x patch_num x patch_len]

        # for channel-wise_1
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        # model shape
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_len = z1.shape[3]

        # proposed
        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))  # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))  # z: [bs * patch_num,nvars, patch_len]

        z = self.fre_transformer(torch.cat((z1, z2), -1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)  # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)  # z: [bs x nvars x target_window]
        z2 = self.head_f2(z2)  # z: [bs x nvars x target_window]

        z = torch.fft.ifft(torch.complex(z1, z2))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            # self.linears2 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                # self.linears2.append(nn.Linear(target_window, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)  # z: [bs x target_window]
                # z = self.linears2[i](z)                    # z: [target_window x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
            # x = self.linear1(x)
            # x = self.linear2(x) + x
            # x = self.dropout(x)
        return x


class Flatten_Head_t(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]

        x = self.flatten(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x

        x = self.linear3(x)
        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, d_model)  # horizon)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x, attn = self.transformer(x)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x  # ,attn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) / self.d_k

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn


class c_Transformer(nn.Module):
    # Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x_n, attn = attn(x)
            x = x_n + x
            x = ff(x) + x
        return x, attn


