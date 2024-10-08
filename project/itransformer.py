import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class ITransformer(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2310.06625
    Github: https://github.com/thuml/Time-Series-Library
    todo: hidden_size,rmse,F.huber_loss()
    """

    def __init__(self, args):
        super(ITransformer, self).__init__()
        self.pred_len = args.pred_len
        # Embedding
        self.enc_embedding = ITEmbedding(args.seq_len, args.d_model, args.dropout)
        # Encoder
        self.encoder = Encoder(args, norm_layer=torch.nn.LayerNorm(args.d_model))
        # Decoder
        self.birnn = nn.LSTM(input_size=args.d_model, hidden_size=args.lstm_hidden_size,
                             num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.projection = nn.Sequential(
            nn.Linear(args.lstm_hidden_size * 2, args.lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.lstm_hidden_size, args.pred_len)
        )

    def forward(self, inputs, inference=False):
        # Normalization from Non-stationary Transformer
        x_enc = inputs['x']   # [batch, seq, feat]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-20)
        x_enc /= stdev

        _, _, N = x_enc.shape  # [batch, time, feat]

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark=None)  # [batch, feat, dim]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [batch, feat, dim]

        lstm_out, _ = self.birnn(enc_out)
        lstm_out = self.dropout(lstm_out)
        dec_out = self.projection(lstm_out).permute(0, 2, 1)[:, :, :N]  # [batch, pred_len, feat]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        pred_temp = dec_out[:, -self.pred_len:, 0:1]  # [batch, pred_len, 1]
        pred_wind = dec_out[:, -self.pred_len:, 1:2]  # [batch, pred_len, 1]

        if inference:
            return pred_temp, pred_wind
        else:
            temp_loss = F.mse_loss(pred_temp, inputs['label_temp']) / inputs['label_temp'].var().detach()
            wind_loss = F.mse_loss(pred_wind, inputs['label_wind']) / inputs['label_wind'].var().detach()
            loss = temp_loss * 10 + wind_loss
            return loss, temp_loss, wind_loss


class ITEmbedding(nn.Module):
    """ for itransformer """
    def __init__(self, s_len, d_model, dropout=0.1):
        super(ITEmbedding, self).__init__()
        self.value_embedding = nn.Linear(s_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = x.permute(0, 2, 1)  # [batch, n_feat, s_len]
        if x_mark is None:
            x = self.value_embedding(x)  # [batch, n_feat, d_model]
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class TriangularCausalMask(object):
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=args.dropout, output_attention=args.output_attention),
            d_model=args.d_model, n_heads=args.n_heads
        )
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.hidden_size, out_channels=args.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = F.relu if args.activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, args, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.num_layers)])
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class MLP(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0.0,
                 use_bn=False, init_std=0.0001, seed=1024):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList([nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()
                                                for _ in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

