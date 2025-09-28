import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from attrdict import AttrDict

from utils.misc import stack, logmeanexp
from utils.sampling import sample_subset
from models.modules import PoolingEncoder, Decoder

class NP(nn.Module):
    # 类初始化
    def __init__(self,
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            dim_lat=128,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        # 定义数据编码器
        self.denc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        # 定义潜在编码器
        self.lenc = PoolingEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                dim_lat=dim_lat,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        # 定义解码器
        self.dec = Decoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_enc=dim_hid+dim_lat,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, z=None, num_samples=None):
        # 从数据编码器获取数据的特征表示 theta ，并进行堆叠
        theta = stack(self.denc(xc, yc), num_samples)
        # 如果未提供潜在变量 z，将从潜在编码器中采样
        if z is None:
            # 获取数据的潜在分布
            pz = self.lenc(xc, yc)
            # 从潜在分布中采样，采样次数由num_samples决定
            z = pz.rsample() if num_samples is None \
                    else pz.rsample([num_samples])
        # 将数据特征 theta 和潜在变量 z 连接起来形成完整的编码
        encoded = torch.cat([theta, z], -1)
        # 对编码后的数据按 xt 的第二维度大小进行复制堆叠（如果需要）
        encoded = stack(encoded, xt.shape[-2], -2)

        # 将堆叠后的 xt 传给解码器进行最终的输出预测
        return self.dec(encoded, stack(xt, num_samples))

    def sample(self, xc, yc, xt, z=None, num_samples=None):
        # 获取预测分布
        pred_dist = self.predict(xc, yc, xt, z, num_samples)
        # 返回均值
        return pred_dist.loc

    def forward(self, batch, num_samples=None, reduce_ll=True):
        # 创建一个属性字典用来存储输出结果
        outs = AttrDict()
        # 训练模式
        if self.training:
            # 计算先验分布和后验分布
            pz = self.lenc(batch.xc, batch.yc)
            qz = self.lenc(batch.x, batch.y)
            # 从后验分布中采样潜在变量
            z = qz.rsample() if num_samples is None else \
                    qz.rsample([num_samples])
            # 使用采样的潜在变量和条件数据进行预测
            py = self.predict(batch.xc, batch.yc, batch.x,
                    z=z, num_samples=num_samples)

            if num_samples > 1:
                # 计算重构损失，多次采样情况下对每个样本的对数似然求和  K * B * N
                recon = py.log_prob(stack(batch.y, num_samples)).sum(-1) ####################################
                # 计算潜在变量的对数似然 K * B
                log_qz = qz.log_prob(z).sum(-1)
                log_pz = pz.log_prob(z).sum(-1)

                # 计算权重 K * B
                log_w = recon.sum(-1) + log_pz - log_qz
                # 计算损失，使用对数平均指数减小数值稳定性问题
                outs.loss = -logmeanexp(log_w).mean() / batch.x.shape[-2]
            else:
                # 单样本情况下，计算重构损失和KL散度
                outs.recon = py.log_prob(batch.y).sum(-1).mean()
                outs.kld = kl_divergence(qz, pz).sum(-1).mean()
                outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]
        # 评估模式
        else:
            # 使用训练好的模型进行预测
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            # 如果没有指定采样次数，直接计算每个目标点的对数似然
            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
            # 如果指定了采样次数，将目标数据重复 num_samples 次，以便计算每个样本的对数似然
            else:
                y = torch.stack([batch.y]*num_samples)
                if reduce_ll:
                    ll = logmeanexp(py.log_prob(y).sum(-1))
                else:
                    ll = py.log_prob(y).sum(-1)
            # 根据reduce_ll设置，计算上下文数据和目标数据的对数似然平均值或直接值
            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                outs.ctx_ll = ll[...,:num_ctx].mean()
                outs.tar_ll = ll[...,num_ctx:].mean()
            else:
                outs.ctx_ll = ll[...,:num_ctx]
                outs.tar_ll = ll[...,num_ctx:]
        return outs
