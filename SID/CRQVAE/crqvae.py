import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLPLayers
from .rq import ResidualVectorQuantizer


class CRQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=0.25,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 # --- SA-SID 核心修复：接收上层透传的情感与空间维度
                 senti_dim=5,
                 spatial_dim=2
        ):
        super(CRQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        # 挂载维度属性
        self.senti_dim = senti_dim
        self.spatial_dim = spatial_dim

        # 编码器
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        # 残差向量量化器
        self.rq = ResidualVectorQuantizer(
            num_emb_list, e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear
        )

        # 基于下游推荐任务进行损失估计，无需再使用重构损失
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, codes = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, codes

    
    # def compute_loss(self, quant_loss, out, xs=None):
    #     if self.loss_type == 'mse':
    #         loss_recon = F.mse_loss(out, xs, reduction='mean')
    #     elif self.loss_type == 'l1':
    #         loss_recon = F.l1_loss(out, xs, reduction='mean')
    #     else:
    #         raise ValueError('incompatible loss type')
    #
    #     loss_total = loss_recon + self.quant_loss_weight * quant_loss
    #
    #     return loss_total, quant_loss, loss_recon
    # def compute_loss(self, quant_loss, out, xs=None):
    #     # --- SA-SID 优化 1: 加权分块重建损失 (Weighted Block Recon Loss) ---
    #     if self.loss_type == 'mse':
    #         # 特征拼接顺序为: [geo(2维), cat(D维), time(27维), sent(5维)]
    #         # 无论你的 cat 是 64维 还是 768维，都可以通过这种切片精准分离
    #         loss_spatial = F.mse_loss(out[:, :2], xs[:, :2], reduction='mean')
    #         loss_other = F.mse_loss(out[:, 2:-5], xs[:, 2:-5], reduction='mean')
    #         loss_senti = F.mse_loss(out[:, -5:], xs[:, -5:], reduction='mean')
    #
    #         # 提高空间和情感特征的权重，强制网络学会重构这区区几维的核心数据
    #         loss_recon = 1.0 * loss_other + 20.0 * loss_spatial + 50.0 * loss_senti
    #
    #     elif self.loss_type == 'l1':
    #         loss_spatial = F.l1_loss(out[:, :2], xs[:, :2], reduction='mean')
    #         loss_other = F.l1_loss(out[:, 2:-5], xs[:, 2:-5], reduction='mean')
    #         loss_senti = F.l1_loss(out[:, -5:], xs[:, -5:], reduction='mean')
    #
    #         loss_recon = 1.0 * loss_other + 20.0 * loss_spatial + 50.0 * loss_senti
    #     else:
    #         raise ValueError('incompatible loss type')
    #
    #     loss_total = loss_recon + self.quant_loss_weight * quant_loss
    #
    #     return loss_total, quant_loss, loss_recon
    def compute_loss(self, quant_loss, out, xs=None):
        # --- SA-SID 优化 1: 动态加权分块重建损失 (Dynamic Weighted Block Recon Loss) ---
        sp_d = self.spatial_dim
        st_d = self.senti_dim

        if self.loss_type == 'mse':
            # 1. 提取最前方的空间特征 (如前 2 维)
            loss_spatial = F.mse_loss(out[:, :sp_d], xs[:, :sp_d], reduction='mean')

            if st_d > 0:
                # 2. 提取最后方的情感特征 (如最后 5 维)
                loss_senti = F.mse_loss(out[:, -st_d:], xs[:, -st_d:], reduction='mean')
                # 3. 提取中间的其他特征 (类别和时间)
                loss_other = F.mse_loss(out[:, sp_d:-st_d], xs[:, sp_d:-st_d], reduction='mean')
                # ⭐ 赋予核心特征极高的重构权重，抵抗维度霸权
                loss_recon = 1.0 * loss_other + 20.0 * loss_spatial + 50.0 * loss_senti
            else:
                # 兼容不带情感特征的回退模式
                loss_other = F.mse_loss(out[:, sp_d:], xs[:, sp_d:], reduction='mean')
                loss_recon = 1.0 * loss_other + 20.0 * loss_spatial

        elif self.loss_type == 'l1':
            loss_spatial = F.l1_loss(out[:, :sp_d], xs[:, :sp_d], reduction='mean')

            if st_d > 0:
                loss_senti = F.l1_loss(out[:, -st_d:], xs[:, -st_d:], reduction='mean')
                loss_other = F.l1_loss(out[:, sp_d:-st_d], xs[:, sp_d:-st_d], reduction='mean')
                loss_recon = 1.0 * loss_other + 20.0 * loss_spatial + 50.0 * loss_senti
            else:
                loss_other = F.l1_loss(out[:, sp_d:], xs[:, sp_d:], reduction='mean')
                loss_recon = 1.0 * loss_other + 20.0 * loss_spatial
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, quant_loss, loss_recon

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        x_q, _, (indices, scalars) = self.rq(x_e, use_sk=use_sk)
        # return indices.cpu(), scalars.cpu()  # [B, L], [B, L]
        x_q_cpu = x_q.cpu()
        indices_cpu = indices.cpu()
        return x_q_cpu, indices_cpu
    
    



    