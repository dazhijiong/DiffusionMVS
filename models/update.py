import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from tqdm.auto import tqdm
import math
class DepthHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x_d, act_fn=torch.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        return act_fn(out)

class ConvGRU_new(nn.Module):
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
    def __init__(self, hidden_dim=128, input_dim=192+128+1):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class ProjectionInputDepth(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convd = nn.Conv2d(64+hidden_dim, out_chs - 1, 3, padding=1)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, depth, cost):
        # print(cost.size())
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        dfm = F.relu(self.convd1(depth))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cor, dfm], dim=1)

        out_d = F.relu(self.convd(cor_dfm))
        if self.training and self.dropout is not None:
            out_d = self.dropout(out_d)
        return torch.cat([out_d, depth], dim=1)

class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask

class BasicUpdateBlockDepth(nn.Module):
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

            # TODO detach()
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
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffUpdateBlockDepth(nn.Module):
        def __init__(self, args, dim=16, dim_mults=(1,2),
                 hidden_dim=128, cost_dim=256, context_dim=64, stage_idx=0, iters=3,ratio=8,UpMask=False):
            super(DiffUpdateBlockDepth, self).__init__()
            self.iters = iters
            self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=16, out_chs=16)
            self.depth_gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim+1)
            self.depth_head = DepthHead(hidden_dim, hidden_dim=hidden_dim, scale=False)
            self.UpMask = UpMask
            self.mask = nn.Sequential(
                nn.Conv2d(context_dim, context_dim*2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(context_dim*2, ratio*ratio*9, 1, padding=0))
            # self.unet = Unet(dim=dim, hidden_dim=hidden_dim,
            #                 channels=(16+context_dim+1),  # [context, input_features, delta_inv_depth]
            #                 out_dim=1,
            #                 dim_mults=dim_mults)

            # build diffusion 
            timesteps = 1000
            sampling_timesteps = 1
            self.timesteps = timesteps
            # define beta schedule
            betas = cosine_beta_schedule(timesteps=timesteps).float()
            # sampling related parameters
            self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
            assert self.sampling_timesteps <= timesteps
            self.is_ddim_sampling = self.sampling_timesteps < timesteps
            self.ddim_sampling_eta = 0.01
            self.scale = 1
            # define alphas 
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
            sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
            sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
            # calculations for diffusion q(x_t | x_{t-1}) and others
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
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
            if noise is None:
                noise = self.scale*torch.randn_like(x_start)

            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        def predict_noise_from_start(self, x_t, t, x0):
            return (
                    (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            )
        def forward(self, net, depth_cost_func, inv_depth, context, seq_len=4, scale_inv_depth=None,gt_inv_depth=None):
            inv_depth_list = [] 
            mask_list = []
            batch_size = inv_depth.shape[0]
            if self.training:
            ### Diffusion forward 正向过程
                gt_delta_inv_depth = gt_inv_depth - inv_depth#compute the ground truth depth residual x0

                gt_delta_inv_depth = torch.where(torch.isinf(gt_delta_inv_depth), torch.zeros_like(gt_delta_inv_depth), gt_delta_inv_depth)
                gt_delta_inv_depth = gt_delta_inv_depth.detach()

                t = torch.randint(0, self.timesteps, (batch_size,), device=inv_depth.device).long()
                noise = (self.scale*torch.randn_like(gt_delta_inv_depth)).float()

                delta_inv_depth = self.q_sample(x_start=gt_delta_inv_depth, t=t, noise=noise)#x_t
                inv_depth_new = inv_depth + delta_inv_depth
                inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                for i in range(self.iters):
                    # delta_inv_depth = delta_inv_depth.detach()
                    delta_inv_depth = delta_inv_depth.detach()
                    # print(i, delta_inv_depth.size())
                    inv_depth_new = inv_depth_new.detach()
                    #we apply several 2D convolution layers on ¯Dt,k and Lt,krespectively to extract geometric features.
                    input_features = self.encoder(inv_depth_new, depth_cost_func(scale_inv_depth(inv_depth_new)[1]))#cost volume
                    #编码(inv_depth_new)，并把上下文，编码后的特征和深度差整合到一起。
                    # inp_i = torch.cat([context, input_features], dim=1)
                    inp_i = torch.cat([context, input_features, delta_inv_depth], dim=1)#The condition ct,k consists of: 
                    #(1) current depth map 
                    #(2) local cost volume 
                    #(3) reference context feature
                    net = self.depth_gru(net, inp_i)
                    delta_inv_depth = self.depth_head(net)
                    inv_depth_new = inv_depth + delta_inv_depth
                    inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                    inv_depth_list.append(inv_depth_new)
                
                    if self.UpMask and i == seq_len - 1 :
                        mask = .25 * self.mask(net)
                        mask_list.append(mask)
                    else:
                        mask_list.append(inv_depth)
                # print("inv_depth_list[-1]",inv_depth_list[-1].shape)
                return  net, mask_list, inv_depth_list
            
            else:
                print("not train")
                batch, device, total_timesteps, sampling_timesteps, eta = batch_size, inv_depth.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta

                times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
                times = list(reversed(times.int().tolist()))
                time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

                img = (self.scale*torch.randn_like(inv_depth)).float()
                

                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                    t = torch.full((batch,), time, device=device, dtype=torch.long)

                    inv_depth_list = [] 
                    mask_list = []

                    delta_inv_depth = img
                    inv_depth_new = inv_depth + delta_inv_depth
                    inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                    
                    for i in range(self.iters):
                        # delta_inv_depth = delta_inv_depth.float()
                        delta_inv_depth = delta_inv_depth.detach()
                        inv_depth_new = inv_depth_new.detach()
                        input_features = self.encoder(inv_depth_new, depth_cost_func(scale_inv_depth(inv_depth_new)[1]))
                        inp_i = torch.cat([context, input_features, delta_inv_depth], dim=1)
                        #inp_i就是ct,k
                        net = self.depth_gru(net, inp_i)
                        delta_inv_depth = self.depth_head(net)
                        # hidden, update = self.unet(inp_i, hidden, t)#unet预测更新值
                        # delta_inv_depth = delta_inv_depth + update#用unet更新delta_inv_depth为x_start
                        # fθ produces the update of depth residual

                        # delta_inv_depth_fea = self.unet_block[i](inp_i, t)
                        # delta_inv_depth = self.depth_block[i](delta_inv_depth_fea)

                        inv_depth_new = inv_depth + delta_inv_depth
                        inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                        # print("inv_depth_new testing",inv_depth_new.shape)
                        inv_depth_list.append(inv_depth_new)
                    
                        
                    # return net, mask_list, inv_depth_list

                    # pred_noise = self.predict_noise_from_start(x, t, x_start)
                    pred_noise = self.predict_noise_from_start(img, t, delta_inv_depth)

                    if time_next < 0:
                        delta_inv_depth = delta_inv_depth
                        continue

                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]

                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()#ddim
                    c = (1 - alpha_next - sigma ** 2).sqrt()

                    noise = (self.scale*torch.randn_like(inv_depth)).float()

                    img = delta_inv_depth * alpha_next.sqrt() + \
                        c * pred_noise + \
                        sigma * noise
                if self.UpMask and i == seq_len - 1 :
                    mask = .25 * self.mask(net)
                    mask_list.append(mask)
                else:
                    mask_list.append(inv_depth)
                return  net, mask_list, inv_depth_list


            # for i in range(seq_len):

            # # TODO detach()
            #     inv_depth = inv_depth.detach()

            #     input_features = self.encoder(inv_depth, depth_cost_func(scale_inv_depth(inv_depth)[1]))

            #     inp_i = torch.cat([context, input_features], dim=1)

            #     net = self.depth_gru(net, inp_i)

            #     delta_inv_depth = self.depth_head(net)

            #     inv_depth = inv_depth + delta_inv_depth
            #     inv_depth_list.append(inv_depth)
            # if self.UpMask and i == seq_len - 1 :
            #     mask = .25 * self.mask(net)
            #     mask_list.append(mask)
            # else:
            #     mask_list.append(inv_depth)
            # return net, mask_list, inv_depth_list
