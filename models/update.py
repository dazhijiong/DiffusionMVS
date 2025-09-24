import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from tqdm.auto import tqdm
import math

class DepthHead(nn.Module):
    """深度预测头网络
    
    作用：从隐藏状态预测深度残差
    输入：隐藏状态特征 [B, hidden_dim, H, W]
    输出：深度残差 [B, 1, H, W]
    """
    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_d, act_fn=torch.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        return act_fn(out)

class ConvGRU_new(nn.Module):
    """改进的卷积门控循环单元
    
    支持外部控制信号（cz, cr, cq）的GRU变体
    """
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU_new, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h

class ConvGRU(nn.Module):
    """标准卷积门控循环单元"""
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    """分离卷积门控循环单元
    
    使用水平和垂直分离的卷积操作，减少参数量并保持感受野
    """
    def __init__(self, hidden_dim=128, input_dim=192+128+1):
        super(SepConvGRU, self).__init__()
        # 水平方向卷积 (1,5)
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        # 垂直方向卷积 (5,1)
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # 水平方向GRU更新
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # 垂直方向GRU更新
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class ProjectionInputDepth(nn.Module):
    """深度和代价体特征编码器
    
    作用：将当前深度图和局部代价体编码为统一特征表示
    输入：深度图 [B, 1, H, W] + 代价体 [B, cost_dim, H, W]
    输出：编码特征 [B, out_chs, H, W]
    """
    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        # 代价体编码分支
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        # 深度图编码分支
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        # 特征融合
        self.convd = nn.Conv2d(64+hidden_dim, out_chs - 1, 3, padding=1)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, depth, cost):
        # 代价体特征编码
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        # 深度图特征编码
        dfm = F.relu(self.convd1(depth))
        dfm = F.relu(self.convd2(dfm))
        
        # 特征融合
        cor_dfm = torch.cat([cor, dfm], dim=1)
        out_d = F.relu(self.convd(cor_dfm))
        
        if self.training and self.dropout is not None:
            out_d = self.dropout(out_d)
        return torch.cat([out_d, depth], dim=1)

class UpMaskNet(nn.Module):
    """上采样掩码生成网络
    
    作用：从隐藏状态生成用于凸组合上采样的掩码权重
    """
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        # 缩放掩码以平衡梯度
        mask = .25 * self.mask(feat)
        return mask

class BasicUpdateBlockDepth(nn.Module):
    """基础深度更新块（非扩散版本）
    
    作用：通过迭代GRU更新预测深度残差
    """
    def __init__(self, hidden_dim=128, cost_dim=256, ratio=8, context_dim=64 ,UpMask=False):
        super(BasicUpdateBlockDepth, self).__init__()

        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim+1)
        self.depth_head = DepthHead(hidden_dim, hidden_dim=hidden_dim, scale=False)
        self.UpMask = UpMask
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, net, depth_cost_func, inv_depth, context, seq_len=4, scale_inv_depth=None):
        inv_depth_list = [] 
        mask_list = []
        for i in range(seq_len):
            inv_depth = inv_depth.detach()

            input_features = self.encoder(inv_depth, depth_cost_func(scale_inv_depth(inv_depth)[1]))

            inp_i = torch.cat([context, input_features], dim=1)

            net = self.depth_gru(net, inp_i)

            delta_inv_depth = self.depth_head(net)

            inv_depth = inv_depth + delta_inv_depth
            inv_depth_list.append(inv_depth)
            if self.UpMask and i == seq_len - 1 :
                mask = .25 * self.mask(net)
                mask_list.append(mask)
            else:
                mask_list.append(inv_depth)
        return net, mask_list, inv_depth_list

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """从张量a中提取时间步t对应的值，并广播到x_shape形状"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """余弦β调度
    
    参考论文：https://openreview.net/forum?id=-NEXDKk8gZ
    生成平滑的β序列，避免线性调度的突变
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffUpdateBlockDepth(nn.Module):
    """扩散深度更新块
    
    核心组件：
    - 扩散过程：1000步余弦β调度，条件生成
    - 特征编码：深度图+代价体→统一特征
    - 循环更新：SepConvGRU迭代预测深度残差
    - 条件生成：上下文+局部代价体+当前深度作为条件
    """
    def __init__(self, args, dim=16, dim_mults=(1,2),
                 hidden_dim=128, cost_dim=256, context_dim=64, stage_idx=0, iters=3,ratio=8,UpMask=False):
        super(DiffUpdateBlockDepth, self).__init__()
        self.iters = iters
        # 特征编码与循环更新网络
        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=16, out_chs=16)
        self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim+1)
        self.depth_head = DepthHead(hidden_dim, hidden_dim=hidden_dim, scale=False)
        self.UpMask = UpMask
        self.mask = nn.Sequential(
            nn.Conv2d(context_dim, context_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_dim*2, ratio*ratio*9, 1, padding=0))

        # 构建扩散过程参数
        timesteps = 1000
        sampling_timesteps = 1
        self.timesteps = timesteps
        # 定义β调度
        betas = cosine_beta_schedule(timesteps=timesteps).float()
        # 采样相关参数
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.01
        self.scale = 1
        
        # 定义α相关参数
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        
        # 扩散过程q(x_t | x_{t-1})相关计算
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        
        # 后验q(x_{t-1} | x_t, x_0)相关计算
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        # 注册为buffer（不参与梯度更新）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：q(x_t | x_0)
        
        从x_0采样得到x_t，即添加噪声的过程
        """
        if noise is None:
            noise = self.scale*torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        """从x_t和x_0预测噪声
        
        用于DDIM采样中的噪声预测
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def forward(self, net, depth_cost_func, inv_depth, context, seq_len=4, scale_inv_depth=None,gt_inv_depth=None):
        """前向过程
        
        训练时：条件扩散生成
        推理时：DDIM采样生成
        
        参数：
        - net: 隐藏状态 [B, hidden_dim, H, W]
        - depth_cost_func: 代价体计算函数
        - inv_depth: 当前逆深度 [B, 1, H, W]
        - context: 上下文特征 [B, context_dim, H, W]
        - seq_len: 序列长度
        - scale_inv_depth: 深度缩放函数
        - gt_inv_depth: 训练时GT逆深度（可选）
        
        返回：
        - net: 更新后的隐藏状态
        - mask_list: 上采样掩码列表
        - inv_depth_list: 各次迭代的逆深度列表
        """
        inv_depth_list = [] 
        mask_list = []
        batch_size = inv_depth.shape[0]
        
        if self.training:
            # 训练时：条件扩散生成
            # 计算GT深度残差作为x_0
            gt_delta_inv_depth = gt_inv_depth - inv_depth

            # 处理无效值
            gt_delta_inv_depth = torch.where(torch.isinf(gt_delta_inv_depth), torch.zeros_like(gt_delta_inv_depth), gt_delta_inv_depth)
            gt_delta_inv_depth = gt_delta_inv_depth.detach()

            # 随机采样时间步和噪声
            t = torch.randint(0, self.timesteps, (batch_size,), device=inv_depth.device).long()
            noise = (self.scale*torch.randn_like(gt_delta_inv_depth)).float()

            # 前向扩散得到x_t
            delta_inv_depth = self.q_sample(x_start=gt_delta_inv_depth, t=t, noise=noise)
            inv_depth_new = inv_depth + delta_inv_depth
            inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
            
            # 迭代预测深度残差
            for i in range(self.iters):
                delta_inv_depth = delta_inv_depth.detach()
                inv_depth_new = inv_depth_new.detach()
                
                # 特征编码：深度图+代价体
                input_features = self.encoder(inv_depth_new, depth_cost_func(scale_inv_depth(inv_depth_new)[1]))
                
                # 条件组合：上下文+编码特征+深度残差
                inp_i = torch.cat([context, input_features, delta_inv_depth], dim=1)
                
                # GRU更新隐藏状态
                net = self.depth_gru(net, inp_i)
                
                # 预测深度残差
                delta_inv_depth = self.depth_head(net)
                
                # 更新逆深度
                inv_depth_new = inv_depth + delta_inv_depth
                inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                inv_depth_list.append(inv_depth_new)
            
                # 生成上采样掩码
                if self.UpMask and i == seq_len - 1 :
                    mask = .25 * self.mask(net)
                    mask_list.append(mask)
                else:
                    mask_list.append(inv_depth)
            return net, mask_list, inv_depth_list
        
        else:
            # 推理时：DDIM采样
            batch, device, total_timesteps, sampling_timesteps, eta = batch_size, inv_depth.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta

            # 构建采样时间步序列
            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))

            # 从噪声开始采样
            img = (self.scale*torch.randn_like(inv_depth)).float()

            for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                t = torch.full((batch,), time, device=device, dtype=torch.long)

                inv_depth_list = [] 
                mask_list = []

                delta_inv_depth = img
                inv_depth_new = inv_depth + delta_inv_depth
                inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                
                # 迭代预测深度残差
                for i in range(self.iters):
                    delta_inv_depth = delta_inv_depth.detach()
                    inv_depth_new = inv_depth_new.detach()
                    
                    # 特征编码和条件组合
                    input_features = self.encoder(inv_depth_new, depth_cost_func(scale_inv_depth(inv_depth_new)[1]))
                    inp_i = torch.cat([context, input_features, delta_inv_depth], dim=1)
                    
                    # GRU更新和残差预测
                    net = self.depth_gru(net, inp_i)
                    delta_inv_depth = self.depth_head(net)

                    inv_depth_new = inv_depth + delta_inv_depth
                    inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                    inv_depth_list.append(inv_depth_new)

                # DDIM采样更新
                pred_noise = self.predict_noise_from_start(img, t, delta_inv_depth)

                if time_next < 0:
                    delta_inv_depth = delta_inv_depth
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = (self.scale*torch.randn_like(inv_depth)).float()

                img = delta_inv_depth * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
                    
            # 生成最终掩码
            if self.UpMask and i == seq_len - 1 :
                mask = .25 * self.mask(net)
                mask_list.append(mask)
            else:
                mask_list.append(inv_depth)
            return net, mask_list, inv_depth_list
