import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .update import BasicUpdateBlockDepth,DiffUpdateBlockDepth
from functools import partial

Align_Corners_Range = False

class DepthNet(nn.Module):
    """多视图深度估计初始网络

    作用：
    - 构建参考帧与源帧在给定深度假设上的可微单应性对齐代价体（variance聚合）
    - 通过代价体正则化网络得到每个深度假设的得分，再做softmax回归得到初始深度
    - 同时输出光度置信度（基于概率体的局部聚合）
    """
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization):
        """前向过程
        参数：
        - features: List[Tensor]，每个视角的特征金字塔同层特征，shape: [B, C, H, W]
        - proj_matrices: Tensor，投影矩阵，shape: [B, N, 2, 4, 4]
        - depth_values: Tensor，深度假设列表（或视差的倒数），shape: [B, D, H, W] 或 [B, D]
        - num_depth: int，本层深度假设数量
        - cost_regularization: nn.Module，代价体正则化网络，输入[B, C, D, H, W]输出[B, 1, D, H, W]
        返回：
        - dict(depth: [B, H, W], photometric_confidence: [B, H, W])
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shape[1], num_depth)
        num_views = len(features)

        # 步骤1：取参考帧与源帧特征
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # 步骤2：可微单应性构建代价体（方差聚合）
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # 构造与当前尺度匹配的投影矩阵，并进行可微单应性变换
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping2(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # 推理阶段为节省显存，使用原地pow_等操作（会修改warped_volume内存）
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume
        # 聚合多个视角特征体的方差作为代价体
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # 步骤3：代价体正则化，得到每个深度的匹配置信度
        prob_volume_pre = cost_regularization(volume_variance).squeeze(1)

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        with torch.no_grad():
            # 光度置信度（对概率体在深度维做局部平均后，读取回归深度索引处的值）
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        return {"depth": depth,  "photometric_confidence": photometric_confidence}


def build_gwc_volume(refimg_fea, targetimg_fea, num_groups):
    """构建分组相关性体（group-wise correlation）
    参数：
    - refimg_fea: Tensor [B, C, H, W]
    - targetimg_fea: Tensor [B, C, D, H, W]
    - num_groups: 分组数量
    返回：Tensor [B, num_groups, D, H, W]
    """
    # B, C, H, W = refimg_fea.shape
    B, C, D, H, W = targetimg_fea.shape
    refimg_fea = refimg_fea.unsqueeze(2).repeat(1, 1, D, 1, 1)
    channels_per_group = C // num_groups
    volume = (refimg_fea * targetimg_fea).view([B, num_groups, channels_per_group, D, H, W]).mean(dim=2)
    volume = volume.contiguous()
    return volume


def disp_to_depth(disp, min_depth, max_depth):
    """将归一化视差映射为视差和值域内深度
    参数：
    - disp: 归一化视差 [0,1]
    - min_depth, max_depth: 深度范围
    返回：
    - scaled_disp: 缩放后的视差
    - depth: 对应深度
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    scaled_disp = scaled_disp.clamp(min = 1e-4)
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """将深度映射为归一化视差"""
    scaled_disp = 1 / depth
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = (scaled_disp - min_disp) / ((max_disp - min_disp))
    return disp


def upsample_depth(depth, mask, ratio=8):
    """利用可学习掩码做凸组合上采样
    输入：
    - depth: [N, 1, H/ratio, W/ratio]
    - mask:  [N, 9*ratio*ratio, H/ratio, W/ratio]
    输出：
    - 上采样后的[N, H, W]
    """
    N, _, H, W = depth.shape
    mask = mask.view(N, 1, 9, ratio, ratio, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(depth, [3, 3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, ratio * H, ratio * W)


class GetCost(nn.Module):
    """基于可微单应性构建局部深度范围的代价体（variance聚合）"""
    def __init__(self):
        super(GetCost, self).__init__()

    def forward(self, depth_values, features, proj_matrices, depth_interval, depth_max, depth_min, CostNum=4):
        """返回展平深度维后的代价体
        参数：
        - depth_values: [B, 1, H, W] 初始深度或视差倒数
        - features: List[Dict] 每个视角各stage的特征
        - proj_matrices: Dict 每个stage的投影矩阵
        - depth_interval: 当前stage深度采样步长
        - depth_max, depth_min: 深度边界（按当前尺度）
        - CostNum: 每个像素的局部深度采样数
        返回：Tensor [B, C*D, H, W]
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        num_views = len(features)

        # 取参考与源特征
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        depth_values = 1./depth_values

        # 构造局部深度采样范围（或直接使用给定深度）
        if CostNum > 1:
            depth_range_samples = get_depth_range_samples(cur_depth=depth_values.squeeze(1),
                                                          ndepth=CostNum,
                                                          depth_inteval_pixel=depth_interval.squeeze(1),
                                                          dtype=ref_feature[0].dtype,
                                                          device=ref_feature[0].device,
                                                          shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                                 ref_feature.shape[3]],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)
        else:
            depth_range_samples = depth_values
        depth_range_samples = 1./depth_range_samples
 
        # 可微单应性构建代价体（方差聚合）
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, depth_range_samples.shape[1], 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping2(src_fea, src_proj_new, ref_proj_new, depth_range_samples)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)
            del warped_volume

        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        b,c,d,h,w = volume_variance.shape
        volume_variance = volume_variance.view(b, c*d, h, w)
        return volume_variance


class Effi_MVS(nn.Module):
    """DiffusionMVS 主网络

    三阶段金字塔：
    - 特征提取：`P_1to8_FeatureNet` 生成代价体分辨率与上下文特征
    - 初始深度：`DepthNet` 在stage1进行粗深度估计
    - 迭代更新：每个stage使用 `DiffUpdateBlockDepth` 在条件（上下文、局部代价体、当前深度）下预测视差残差
    - 上采样重建：掩码引导的凸组合上采样得到最终深度
    """
    def __init__(self, args, depth_interals_ratio=[4,2,1], stage_channel=True):
        super(Effi_MVS, self).__init__()
        self.ndepths = args.ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.stage_channel = stage_channel
        seq_len = [int(e) for e in args.GRUiters.split(",")]
        self.seq_len = seq_len
        self.args = args
        self.num_stage = 3
        self.CostNum = args.CostNum
        self.GetCost = GetCost()
        self.hdim_stage = [64,32,16]
        self.cdim_stage = [64,32,16]
        self.context_feature = [128, 64, 32]
        self.feat_ratio = 8
        self.cost_dim_stage = [32, 16, 8]
        print("**********netphs:{}, depth_intervals_ratio:{}, hdim_stage:{}, cdim_stage:{}, context_feature:{}, cost_dim_stage:{}************".format(
            self.ndepths,depth_interals_ratio,self.hdim_stage,self.cdim_stage,self.context_feature,self.cost_dim_stage))
        # 特征与上下文网络
        self.feature = P_1to8_FeatureNet(base_channels=8, out_channel=self.cost_dim_stage, stage_channel=self.stage_channel)
        self.cnet_depth = P_1to8_FeatureNet(base_channels=8, out_channel=self.context_feature, stage_channel=self.stage_channel)
        # 扩散更新块（3个stage）
        self.update_block_depth1 = DiffUpdateBlockDepth(args=args,
           hidden_dim=self.hdim_stage[0], cost_dim=self.cost_dim_stage[0]*self.CostNum,
            context_dim=self.cdim_stage[0], stage_idx=0, UpMask=True)
        self.update_block_depth2 = DiffUpdateBlockDepth(args, 
            hidden_dim=self.hdim_stage[1], cost_dim=self.cost_dim_stage[1]*self.CostNum,
            context_dim=self.cdim_stage[1], stage_idx=1, UpMask=True)

        self.update_block_depth3 = DiffUpdateBlockDepth(args, 
             hidden_dim=self.hdim_stage[2], cost_dim=self.cost_dim_stage[2]*self.CostNum,
            context_dim=self.cdim_stage[2], stage_idx=2, UpMask=True)
        self.update_block = nn.ModuleList([self.update_block_depth1,self.update_block_depth2,self.update_block_depth3])
        # 初始深度与代价体正则化
        self.depthnet = DepthNet()
        self.cost_regularization = CostRegNet_small(in_channels=self.cost_dim_stage[0], base_channels=8)
        # 最终上采样
        self.up_ratio = 2
        self.upsample = nn.Sequential(
            nn.Conv2d(self.cost_dim_stage[2], 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.up_ratio*self.up_ratio*9, 1, stride=1, padding=0, dilation=1, bias=False)
        )


    def forward(self, imgs, proj_matrices, depth_values,depth_gt=None):
        """前向推理/训练
        - imgs: [B, N, 3, H, W]
        - proj_matrices: 每stage投影矩阵
        - depth_values: stage1用于初始估计的深度假设
        - depth_gt: 训练时分stage的GT字典（可选）
        返回：
        - dict(depth: List[Tensor], photometric_confidence: Tensor, mask_list: List[Tensor])
        """
        disp_min = depth_values[:, 0, None, None, None]
        disp_max = depth_values[:, -1, None, None, None]
        depth_max_ = 1. / disp_min
        depth_min_ = 1. / disp_max

        # 生成视差/深度互转函数（与当前样本的深度范围绑定）
        self.scale_inv_depth = partial(disp_to_depth, min_depth=depth_min_, max_depth=depth_max_)

        depth_interval = (disp_max - disp_min) / depth_values.size(1)
        # 特征提取（所有视角）
        features = []
        depth_predictions = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        cnet_depth = self.cnet_depth(imgs[:, 0])
        for stage_idx in range(self.num_stage):
            if self.training:
                # 训练时按stage取GT并处理无效值
                depth_gt_stage = depth_gt[f"stage{stage_idx+1}"].unsqueeze(1)
                _, _, H, W = depth_gt_stage.size()
                depth_maxs = depth_max_.view(-1,1,1,1).repeat(1,1,H,W)
                depth_gt_stage = torch.where(depth_gt_stage>1e-1, depth_gt_stage, depth_maxs)
                inv_depth_gt = depth_to_disp(depth_gt_stage, depth_min_, depth_max_)
            else:
                inv_depth_gt = None
            
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            ref_feature = features_stage[0]
            if stage_idx == 0:
                # 在stage1上构建全局深度假设并做初始深度估计
                depth_range_samples = get_depth_range_samples(cur_depth=depth_values,
                                                              ndepth=self.ndepths,
                                                              depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                              dtype=ref_feature[0].dtype,
                                                              device=ref_feature[0].device,
                                                              shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                                     ref_feature.shape[3]],
                                                              max_depth=disp_max,
                                                              min_depth=disp_min)
                depth_range_samples = 1./depth_range_samples

                init_depth = self.depthnet(features_stage, proj_matrices_stage, depth_values=depth_range_samples,
                                           num_depth=self.ndepths, cost_regularization=self.cost_regularization)

                photometric_confidence = init_depth["photometric_confidence"]
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),[ref_feature.shape[2]*8, ref_feature.shape[3]*8], mode='nearest')
                photometric_confidence = photometric_confidence.squeeze(1)
                init_depth = init_depth['depth']
                cur_depth = init_depth.unsqueeze(1)

                depth_predictions = [init_depth]
                depth_predictions.append(F.interpolate(init_depth.unsqueeze(1).detach(), scale_factor=2, mode='bilinear').squeeze(1))
                depth_predictions.append(F.interpolate(init_depth.unsqueeze(1).detach(), scale_factor=4, mode='bilinear').squeeze(1))
                inv_initial_depth = depth_to_disp(cur_depth, depth_min_, depth_max_)
                inv_cur_depth = inv_initial_depth
            else:
                # 后续stage使用上一stage的估计结果作为初始化
                inv_cur_depth=last_inv_depth.detach()
            
            cnet_depth_stage = cnet_depth["stage{}".format(stage_idx + 1)]

            hidden_d, inp_d = torch.split(cnet_depth_stage, [self.hdim_stage[stage_idx], self.cdim_stage[stage_idx]], dim=1)

            current_hidden_d = torch.tanh(hidden_d)

            inp_d = torch.relu(inp_d)

            depth_cost_func = partial(self.GetCost, features=features_stage, proj_matrices=proj_matrices_stage,
                                      depth_interval=depth_interval*self.depth_interals_ratio[stage_idx], depth_max=disp_max, depth_min=disp_min,
                                      CostNum=self.CostNum)

            current_hidden_d, up_mask_seqs, inv_depth_seqs = self.update_block[stage_idx](current_hidden_d, depth_cost_func,
                                                                             inv_cur_depth,
                                                                             inp_d, seq_len=self.seq_len[stage_idx],
                                                                             scale_inv_depth=self.scale_inv_depth,
                                                                             gt_inv_depth=inv_depth_gt)
            
            # 记录各次迭代的深度结果
            for inv_depth_i in inv_depth_seqs:
                depth_predictions.append(self.scale_inv_depth(inv_depth_i)[1].squeeze(1))
            last_mask = up_mask_seqs[-1]
            last_inv_depth = inv_depth_seqs[-1]
            if stage_idx<2:
                last_inv_depth = F.interpolate(last_inv_depth,
                                    scale_factor=2, mode='bilinear')
        # 最终上采样到更高分辨率
        mask = .25 * self.upsample(ref_feature)
        depth_upsampled = upsample_depth(last_inv_depth, mask, self.up_ratio).unsqueeze(1)
        depth_predictions.append(self.scale_inv_depth(depth_upsampled)[1].squeeze(1))
        return {"depth": depth_predictions, "photometric_confidence": photometric_confidence,"mask_list":up_mask_seqs}
