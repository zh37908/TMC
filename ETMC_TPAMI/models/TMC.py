import torch
import torch.nn as nn
from models.image import ImageEncoder
import torch.nn.functional as F
from models.channel import Channel
import numpy as np

# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))


class TMC(nn.Module):
    def __init__(self, args):
        super(TMC, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        self.clf_depth = nn.ModuleList()
        self.clf_rgb = nn.ModuleList()
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(depth_last_size, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_depth.append(nn.Linear(depth_last_size, args.n_classes))

        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden
        self.clf_rgb.append(nn.Linear(rgb_last_size, args.n_classes))

    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.args.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        depth_evidence, rgb_evidence = F.softplus(depth_out), F.softplus(rgb_out)
        depth_alpha, rgb_alpha = depth_evidence+1, rgb_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha


class ETMC(TMC):
    def __init__(self, args):
        super(ETMC, self).__init__(args)
        last_size = args.img_hidden_sz * args.num_image_embeds + args.img_hidden_sz * args.num_image_embeds
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden
        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        pseudo_out = torch.cat([rgb, depth], -1)
        for layer in self.clf:
            pseudo_out = layer(pseudo_out)

        depth_evidence, rgb_evidence, pseudo_evidence = F.softplus(depth_out), F.softplus(rgb_out), F.softplus(pseudo_out)
        depth_alpha, rgb_alpha, pseudo_alpha = depth_evidence+1, rgb_evidence+1, pseudo_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(self.DS_Combin_two(depth_alpha, rgb_alpha), pseudo_alpha)
        return depth_alpha, rgb_alpha, pseudo_alpha, depth_rgb_alpha


class TMC_base(nn.Module):
    """
    Baseline model using standard softmax outputs and cross-entropy loss.
    The depth-rgb prediction is produced by an additional classifier that
    takes the concatenation of depth and rgb image features.
    """
    def __init__(self, args):
        super(TMC_base, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)

        depth_feat_dim = args.img_hidden_sz * args.num_image_embeds
        rgb_feat_dim = args.img_hidden_sz * args.num_image_embeds
        comb_feat_dim = depth_feat_dim + rgb_feat_dim

        # depth classifier
        self.clf_depth = nn.ModuleList()
        last_dim = depth_feat_dim
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(last_dim, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth.append(nn.Linear(last_dim, args.n_classes))

        # rgb classifier
        self.clf_rgb = nn.ModuleList()
        last_dim = rgb_feat_dim
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_rgb.append(nn.Linear(last_dim, args.n_classes))

        # combined depth+rgb classifier
        self.clf_depth_rgb = nn.ModuleList()
        last_dim = comb_feat_dim
        for hidden in args.hidden:
            self.clf_depth_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_depth_rgb.append(nn.ReLU())
            self.clf_depth_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth_rgb.append(nn.Linear(last_dim, args.n_classes))

    def _forward_mlp(self, x, mlp):
        for layer in mlp:
            x = layer(x)
        return x

    def forward(self, rgb, depth):
        # extract visual features
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)

        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)

        # individual modality logits
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        # fused logits for depth-rgb
        comb_feat = torch.cat([depth_feat, rgb_feat], dim=-1)
        depth_rgb_logits = self._forward_mlp(comb_feat, self.clf_depth_rgb)

        # return raw logits (CrossEntropyLoss expects logits)
        return depth_logits, rgb_logits, depth_rgb_logits


class TMC_base(nn.Module):
    """
    Baseline model using standard softmax outputs and cross-entropy loss.
    The depth-rgb prediction is produced by an additional classifier that
    takes the concatenation of depth and rgb image features.
    """
    def __init__(self, args):
        super(TMC_base, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)

        depth_feat_dim = args.img_hidden_sz * args.num_image_embeds
        rgb_feat_dim = args.img_hidden_sz * args.num_image_embeds
        comb_feat_dim = depth_feat_dim + rgb_feat_dim

        # depth classifier
        self.clf_depth = nn.ModuleList()
        last_dim = depth_feat_dim
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(last_dim, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth.append(nn.Linear(last_dim, args.n_classes))

        # rgb classifier
        self.clf_rgb = nn.ModuleList()
        last_dim = rgb_feat_dim
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_rgb.append(nn.Linear(last_dim, args.n_classes))

        # combined depth+rgb classifier
        self.clf_depth_rgb = nn.ModuleList()
        last_dim = comb_feat_dim
        for hidden in args.hidden:
            self.clf_depth_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_depth_rgb.append(nn.ReLU())
            self.clf_depth_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth_rgb.append(nn.Linear(last_dim, args.n_classes))

    def _forward_mlp(self, x, mlp):
        for layer in mlp:
            x = layer(x)
        return x

    def forward(self, rgb, depth):
        # extract visual features
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)

        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)

        # individual modality logits
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        # fused logits for depth-rgb
        comb_feat = torch.cat([depth_feat, rgb_feat], dim=-1)
        depth_rgb_logits = self._forward_mlp(comb_feat, self.clf_depth_rgb)

        # return raw logits (CrossEntropyLoss expects logits)
        return depth_logits, rgb_logits, depth_rgb_logits
    


import torchvision

class ImageEncoder_no_pretrain(nn.Module):
    def __init__(self, args):
        super(ImageEncoder_no_pretrain, self).__init__()
        self.args = args

        model = torchvision.models.resnet18(pretrained=False)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048



class TMC_base_channel(nn.Module):
    """
    Baseline model using standard softmax outputs and cross-entropy loss.
    The depth-rgb prediction is produced by an additional classifier that
    takes the concatenation of depth and rgb image features.
    """
    def __init__(self, args):
        super(TMC_base_channel, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder_no_pretrain(args)
        
        self.depthenc = ImageEncoder_no_pretrain(args)  
        self.channel = Channel('awgn')

        depth_feat_dim = args.img_hidden_sz * args.num_image_embeds
        rgb_feat_dim = args.img_hidden_sz * args.num_image_embeds
        comb_feat_dim = depth_feat_dim + rgb_feat_dim
        last_dim = rgb_feat_dim

        self.rgbchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.rgbchannel_enc.append(nn.Linear(last_dim, hidden))
            self.rgbchannel_enc.append(nn.ReLU())
            self.rgbchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.rgbchannel_enc.append(nn.Linear(last_dim, args.channel_size))
        last_dim = depth_feat_dim
        self.depthchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.depthchannel_enc.append(nn.Linear(last_dim, hidden))
            self.depthchannel_enc.append(nn.ReLU())
            self.depthchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.depthchannel_enc.append(nn.Linear(last_dim, args.channel_size))
        # depth classifier
        self.clf_depth = nn.ModuleList()
        last_dim = args.channel_size
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(last_dim, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth.append(nn.Linear(last_dim, args.n_classes))

        # rgb classifier
        self.clf_rgb = nn.ModuleList()
        last_dim = args.channel_size
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_rgb.append(nn.Linear(last_dim, args.n_classes))

        # combined depth+rgb classifier
        self.clf_depth_rgb = nn.ModuleList()
        last_dim = 2*args.channel_size
        for hidden in args.hidden:
            self.clf_depth_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_depth_rgb.append(nn.ReLU())
            self.clf_depth_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth_rgb.append(nn.Linear(last_dim, args.n_classes))

    def _forward_mlp(self, x, mlp):
        for layer in mlp:
            x = layer(x)
        return x

    def forward(self, rgb, depth):
        # extract visual features
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        depth_feat = self.channel(depth_feat, self.args.channel_snr)

        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        rgb_feat = self.channel(rgb_feat, self.args.channel_snr)

        # individual modality logits
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        # fused logits for depth-rgb
        comb_feat = torch.cat([depth_feat, rgb_feat], dim=-1)
        depth_rgb_logits = self._forward_mlp(comb_feat, self.clf_depth_rgb)

        # return raw logits (CrossEntropyLoss expects logits)
        return depth_logits, rgb_logits, depth_rgb_logits


class TMC_channel(nn.Module):
    """
    Evidential TMC variant with channel encoders and a channel layer,
    mirroring the structural changes of TMC_base_channel relative to TMC_base.
    """
    def __init__(self, args):
        super(TMC_channel, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder_no_pretrain(args)
        self.depthenc = ImageEncoder_no_pretrain(args)
        self.channel = Channel('awgn')
        # 支持动态信道范围，与 TMC_channel_dynamic/TMC_channel_snr 一致
        self.snr_min: float = float(getattr(args, "snr_min", 0.0))
        self.snr_max: float = float(getattr(args, "snr_max", 20.0))

        depth_feat_dim = args.img_hidden_sz * args.num_image_embeds
        rgb_feat_dim = args.img_hidden_sz * args.num_image_embeds

        # channel encoders to map visual features to compact channel_size
        last_dim = rgb_feat_dim
        self.rgbchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.rgbchannel_enc.append(nn.Linear(last_dim, hidden))
            self.rgbchannel_enc.append(nn.ReLU())
            self.rgbchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.rgbchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        last_dim = depth_feat_dim
        self.depthchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.depthchannel_enc.append(nn.Linear(last_dim, hidden))
            self.depthchannel_enc.append(nn.ReLU())
            self.depthchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.depthchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        # classifiers consume channel-encoded features
        self.clf_depth = nn.ModuleList()
        last_dim = args.channel_size
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(last_dim, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth.append(nn.Linear(last_dim, args.n_classes))

        self.clf_rgb = nn.ModuleList()
        last_dim = args.channel_size
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_rgb.append(nn.Linear(last_dim, args.n_classes))

    def _forward_mlp(self, x: torch.Tensor, mlp: nn.ModuleList) -> torch.Tensor:
        for layer in mlp:
            x = layer(x)
        return x

    def DS_Combin_two(self, alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        alpha = {0: alpha1, 1: alpha2}
        b, S, E, u = {}, {}, {}, {}
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        S_a = self.args.n_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, snr: torch.Tensor = None):
        # 兼容两种用法：
        # - 显式传入 snr（[B] 或 [B,1]），两模态共用该批次的 SNR 标量
        # - 不传入 snr（None）时，深度与 RGB 分别在 [snr_min, snr_max] 内独立采样（动态信道）
        use_dynamic = snr is None
        if use_dynamic:
            snr_scalar_depth = float(torch.empty((), device=rgb.device).uniform_(self.snr_min, self.snr_max).item())
            snr_scalar_rgb = float(torch.empty((), device=rgb.device).uniform_(self.snr_min, self.snr_max).item())
        else:
            snr_input = snr.view(-1, 1) if snr.dim() == 1 else snr
            snr_scalar = float(torch.mean(snr_input).item())
            snr_scalar_depth = snr_scalar
            snr_scalar_rgb = snr_scalar

        # extract and channel-encode depth features
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        depth_feat = self.channel(depth_feat, snr_scalar_depth)

        # extract and channel-encode rgb features
        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        rgb_feat = self.channel(rgb_feat, snr_scalar_rgb)

        # per-modality logits then evidences
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        depth_evidence = F.softplus(depth_logits)
        rgb_evidence = F.softplus(rgb_logits)
        depth_alpha = depth_evidence + 1
        rgb_alpha = rgb_evidence + 1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha


class TMC_base_channel_dynamic(TMC_base_channel):
    """
    与 TMC_base_channel 结构相同，但在每次 forward 内分别为 depth 与 rgb 独立采样 SNR∈[0,20] dB。
    """
    def __init__(self, args):
        super(TMC_base_channel_dynamic, self).__init__(args)
        # store snr range from args if provided
        self.snr_min: float = float(getattr(args, "snr_min", 0.0))
        self.snr_max: float = float(getattr(args, "snr_max", 20.0))

    def forward(self, rgb, depth):
        # extract visual features
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        # sample independent SNR for depth
        snr_depth = float(np.random.uniform(self.snr_min, self.snr_max))
        depth_feat = self.channel(depth_feat, snr_depth)

        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        # sample independent SNR for rgb
        snr_rgb = float(np.random.uniform(self.snr_min, self.snr_max))
        rgb_feat = self.channel(rgb_feat, snr_rgb)

        # individual modality logits
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        # fused logits for depth-rgb
        comb_feat = torch.cat([depth_feat, rgb_feat], dim=-1)
        depth_rgb_logits = self._forward_mlp(comb_feat, self.clf_depth_rgb)

        return depth_logits, rgb_logits, depth_rgb_logits


class TMC_channel_dynamic(TMC_channel):
    """
    与 TMC_channel 结构相同，但在每次 forward 内分别为 depth 与 rgb 独立采样 SNR∈[0,20] dB。
    """
    def __init__(self, args):
        super(TMC_channel_dynamic, self).__init__(args)
        self.snr_min: float = float(getattr(args, "snr_min", 0.0))
        self.snr_max: float = float(getattr(args, "snr_max", 20.0))

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # extract and channel-encode depth features
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        snr_depth = float(np.random.uniform(self.snr_min, self.snr_max))
        depth_feat = self.channel(depth_feat, snr_depth)

        # extract and channel-encode rgb features
        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        snr_rgb = float(np.random.uniform(self.snr_min, self.snr_max))
        rgb_feat = self.channel(rgb_feat, snr_rgb)

        # per-modality logits then evidences
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        depth_evidence = F.softplus(depth_logits)
        rgb_evidence = F.softplus(rgb_logits)
        depth_alpha = depth_evidence + 1
        rgb_alpha = rgb_evidence + 1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha


class TMC_channel_snr(nn.Module):
    """
    TMC + channel 结构，并显式引入 SNR 条件（无 pseudo 分支）。

    通过 `args.snr_input_method` 指定融合方式：
    - "concat": 将 SNR 嵌入向量拼接到每个模态的 channel 特征后，再送入各自分类头；
    - "add": 将 SNR 嵌入投影到与 channel 特征同维度后做逐元素相加；
    - "mlp": 将 [feat, snr_embed] 连接后经一层 MLP 融合为与 channel_size 同维度；
    - "none": 不将 SNR 嵌入输入到 decoder/分类头（仅用于通道扰动）。

    forward 接受 snr（形状 [B] 或 [B,1]），用于条件化特征；为与已有 Channel 接口保持兼容，通道扰动仍使用 batch 平均 SNR 的标量。
    """
    def __init__(self, args):
        super(TMC_channel_snr, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder_no_pretrain(args)
        self.depthenc = ImageEncoder_no_pretrain(args)
        self.channel = Channel(args.channel_type)
        # 支持动态信道范围（与 TMC_channel_dynamic 一致）
        self.snr_min: float = float(getattr(args, 'snr_min', 0.0))
        self.snr_max: float = float(getattr(args, 'snr_max', 20.0))

        depth_feat_dim = args.img_hidden_sz * args.num_image_embeds
        rgb_feat_dim = args.img_hidden_sz * args.num_image_embeds

        # 将视觉特征编码到 channel 空间
        last_dim = rgb_feat_dim
        self.rgbchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.rgbchannel_enc.append(nn.Linear(last_dim, hidden))
            self.rgbchannel_enc.append(nn.ReLU())
            self.rgbchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.rgbchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        last_dim = depth_feat_dim
        self.depthchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.depthchannel_enc.append(nn.Linear(last_dim, hidden))
            self.depthchannel_enc.append(nn.ReLU())
            self.depthchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.depthchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        # SNR 嵌入与融合
        self.snr_embed_dim: int = int(getattr(args, 'snr_embed_dim', 64))
        self.snr_input_method: str = str(getattr(args, 'snr_input_method', 'concat'))

        self.snr_embed = nn.Sequential(
            nn.Linear(1, self.snr_embed_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

        # 按融合策略配置分类头的输入维度与必要的投影/融合层
        if self.snr_input_method == 'concat':
            depth_clf_in = args.channel_size + self.snr_embed_dim
            rgb_clf_in = args.channel_size + self.snr_embed_dim
            self.fuse_depth = None
            self.fuse_rgb = None
            self.snr_to_channel = None
        elif self.snr_input_method == 'add':
            depth_clf_in = args.channel_size
            rgb_clf_in = args.channel_size
            self.snr_to_channel = nn.Linear(self.snr_embed_dim, args.channel_size)
            self.fuse_depth = None
            self.fuse_rgb = None
        elif self.snr_input_method == 'mlp':
            depth_clf_in = args.channel_size
            rgb_clf_in = args.channel_size
            self.fuse_depth = nn.Sequential(
                nn.Linear(args.channel_size + self.snr_embed_dim, args.channel_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.fuse_rgb = nn.Sequential(
                nn.Linear(args.channel_size + self.snr_embed_dim, args.channel_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.snr_to_channel = None
        else:  # 'none'
            depth_clf_in = args.channel_size
            rgb_clf_in = args.channel_size
            self.fuse_depth = None
            self.fuse_rgb = None
            self.snr_to_channel = None

        # 分类头（evidential）
        self.clf_depth = nn.ModuleList()
        last_dim = depth_clf_in
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(last_dim, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth.append(nn.Linear(last_dim, args.n_classes))

        self.clf_rgb = nn.ModuleList()
        last_dim = rgb_clf_in
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_rgb.append(nn.Linear(last_dim, args.n_classes))

    def _forward_mlp(self, x: torch.Tensor, mlp: nn.ModuleList) -> torch.Tensor:
        for layer in mlp:
            x = layer(x)
        return x

    def DS_Combin_two(self, alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        alpha = {0: alpha1, 1: alpha2}
        b, S, E, u = {}, {}, {}, {}
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        bb_sum = torch.sum(bb, dim=(1, 2))
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        S_a = self.args.n_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def _fuse_with_snr(self, feat: torch.Tensor, snr_embed: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'concat':
            return torch.cat([feat, snr_embed], dim=-1)
        if mode == 'add':
            assert self.snr_to_channel is not None
            snr_proj = self.snr_to_channel(snr_embed)
            return feat + snr_proj
        if mode == 'none':
            return feat
        # mlp 由上游处理
        return feat

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, snr: torch.Tensor):
        # 支持两种模式：
        # 1) 传入 snr（[B] 或 [B,1]），两模态使用同一批次 SNR；
        # 2) 未传入 snr 时，按 [snr_min, snr_max] 为 depth/rgb 分别采样批量共享的随机 SNR（动态信道）。

        use_dynamic = snr is None
        B = rgb.shape[0]

        if use_dynamic:
            # torch 随机（避免 numpy 往返与不同随机源）
            snr_scalar_depth = torch.empty((), device=rgb.device).uniform_(self.snr_min, self.snr_max).item()
            snr_scalar_rgb = torch.empty((), device=rgb.device).uniform_(self.snr_min, self.snr_max).item()
            snr_input_depth = torch.full((B, 1), snr_scalar_depth, dtype=torch.float32, device=rgb.device)
            snr_input_rgb = torch.full((B, 1), snr_scalar_rgb, dtype=torch.float32, device=rgb.device)
        else:
            snr_input = snr.view(-1, 1) if snr.dim() == 1 else snr
            snr_scalar = float(torch.mean(snr_input).item())
            snr_scalar_depth = snr_scalar
            snr_scalar_rgb = snr_scalar
            snr_input_depth = snr_input
            snr_input_rgb = snr_input

        # 视觉特征 -> channel 编码 -> 通道扰动（depth/rgb 可用不同 SNR）
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        depth_feat = self.channel(depth_feat, snr_scalar_depth)

        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        rgb_feat = self.channel(rgb_feat, snr_scalar_rgb)

        # SNR 嵌入与按策略融合 SNR 条件
        if self.snr_input_method == 'mlp':
            snr_embed_d = self.snr_embed(snr_input_depth)
            snr_embed_r = self.snr_embed(snr_input_rgb)
            assert hasattr(self, 'fuse_depth') and self.fuse_depth is not None
            assert hasattr(self, 'fuse_rgb') and self.fuse_rgb is not None
            depth_feat_fused = self.fuse_depth(torch.cat([depth_feat, snr_embed_d], dim=-1))
            rgb_feat_fused = self.fuse_rgb(torch.cat([rgb_feat, snr_embed_r], dim=-1))
        elif self.snr_input_method in ('concat', 'add'):
            snr_embed_d = self.snr_embed(snr_input_depth)
            snr_embed_r = self.snr_embed(snr_input_rgb)
            depth_feat_fused = self._fuse_with_snr(depth_feat, snr_embed_d, self.snr_input_method)
            rgb_feat_fused = self._fuse_with_snr(rgb_feat, snr_embed_r, self.snr_input_method)
        else:  # 'none'
            depth_feat_fused = depth_feat
            rgb_feat_fused = rgb_feat

        # per-modality logits -> evidences -> alphas
        depth_logits = self._forward_mlp(depth_feat_fused, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat_fused, self.clf_rgb)

        depth_evidence = F.softplus(depth_logits)
        rgb_evidence = F.softplus(rgb_logits)
        depth_alpha = depth_evidence + 1
        rgb_alpha = rgb_evidence + 1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha

class ETMC_channel(nn.Module):
    """
    Evidential ETMC variant with channel encoders and a channel layer.
    Mirrors ETMC (with extra pseudo branch) while adopting channel pathway like TMC_channel.
    """
    def __init__(self, args):
        super(ETMC_channel, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder_no_pretrain(args)
        self.depthenc = ImageEncoder_no_pretrain(args)
        self.channel = Channel('awgn')

        depth_feat_dim = args.img_hidden_sz * args.num_image_embeds
        rgb_feat_dim = args.img_hidden_sz * args.num_image_embeds

        # channel encoders map to compact channel_size
        last_dim = rgb_feat_dim
        self.rgbchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.rgbchannel_enc.append(nn.Linear(last_dim, hidden))
            self.rgbchannel_enc.append(nn.ReLU())
            self.rgbchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.rgbchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        last_dim = depth_feat_dim
        self.depthchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.depthchannel_enc.append(nn.Linear(last_dim, hidden))
            self.depthchannel_enc.append(nn.ReLU())
            self.depthchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.depthchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        # classifiers consume channel-encoded features
        self.clf_depth = nn.ModuleList()
        last_dim = args.channel_size
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(last_dim, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth.append(nn.Linear(last_dim, args.n_classes))

        self.clf_rgb = nn.ModuleList()
        last_dim = args.channel_size
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_rgb.append(nn.Linear(last_dim, args.n_classes))

        # pseudo branch MLP, input is concatenated channel-encoded features
        comb_dim = args.channel_size + args.channel_size
        self.clf_pseudo = nn.ModuleList()
        last_dim = comb_dim
        for hidden in args.hidden:
            self.clf_pseudo.append(nn.Linear(last_dim, hidden))
            self.clf_pseudo.append(nn.ReLU())
            self.clf_pseudo.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_pseudo.append(nn.Linear(last_dim, args.n_classes))

    def _forward_mlp(self, x: torch.Tensor, mlp: nn.ModuleList) -> torch.Tensor:
        for layer in mlp:
            x = layer(x)
        return x

    def DS_Combin_two(self, alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        alpha = {0: alpha1, 1: alpha2}
        b, S, E, u = {}, {}, {}, {}
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        bb_sum = torch.sum(bb, dim=(1, 2))
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        S_a = self.args.n_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # extract and channel-encode
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        depth_feat = self.channel(depth_feat, self.args.channel_snr)

        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        rgb_feat = self.channel(rgb_feat, self.args.channel_snr)

        # per-modality logits -> evidences -> alphas
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        depth_evidence = F.softplus(depth_logits)
        rgb_evidence = F.softplus(rgb_logits)
        depth_alpha = depth_evidence + 1
        rgb_alpha = rgb_evidence + 1

        # pseudo branch on concatenated channel features
        pseudo_feat = torch.cat([rgb_feat, depth_feat], dim=-1)
        pseudo_logits = self._forward_mlp(pseudo_feat, self.clf_pseudo)
        pseudo_evidence = F.softplus(pseudo_logits)
        pseudo_alpha = pseudo_evidence + 1

        depth_rgb_alpha = self.DS_Combin_two(self.DS_Combin_two(depth_alpha, rgb_alpha), pseudo_alpha)
        return depth_alpha, rgb_alpha, pseudo_alpha, depth_rgb_alpha


class ETMC_channel_dynamic(ETMC_channel):
    """
    与 ETMC_channel 结构相同，但在每次 forward 内分别为 depth 与 rgb 独立采样 SNR∈[snr_min, snr_max] dB。
    """
    def __init__(self, args):
        super(ETMC_channel_dynamic, self).__init__(args)
        self.snr_min: float = float(getattr(args, "snr_min", 0.0))
        self.snr_max: float = float(getattr(args, "snr_max", 20.0))

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # depth path
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        snr_depth = float(np.random.uniform(self.snr_min, self.snr_max))
        depth_feat = self.channel(depth_feat, snr_depth)

        # rgb path
        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        snr_rgb = float(np.random.uniform(self.snr_min, self.snr_max))
        rgb_feat = self.channel(rgb_feat, snr_rgb)

        # heads
        depth_logits = self._forward_mlp(depth_feat, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat, self.clf_rgb)

        depth_evidence = F.softplus(depth_logits)
        rgb_evidence = F.softplus(rgb_logits)
        depth_alpha = depth_evidence + 1
        rgb_alpha = rgb_evidence + 1

        pseudo_feat = torch.cat([rgb_feat, depth_feat], dim=-1)
        pseudo_logits = self._forward_mlp(pseudo_feat, self.clf_pseudo)
        pseudo_evidence = F.softplus(pseudo_logits)
        pseudo_alpha = pseudo_evidence + 1

        depth_rgb_alpha = self.DS_Combin_two(self.DS_Combin_two(depth_alpha, rgb_alpha), pseudo_alpha)
        return depth_alpha, rgb_alpha, pseudo_alpha, depth_rgb_alpha


class ETMC_channel_snr(nn.Module):
    """
    ETMC + channel 结构，并显式引入 SNR 条件。

    通过 `args.snr_input_method` 指定融合方式：
    - "concat": 将 SNR 嵌入向量拼接到每个模态的 channel 特征后，再送入各自分类头；
    - "add": 将 SNR 嵌入投影到与 channel 特征同维度后做逐元素相加；
    - "mlp": 将 [feat, snr_embed] 连接后经一层 MLP 融合为与 channel_size 同维度。

    forward 接受 snr（形状 [B] 或 [B,1]），用于条件化特征；为与已有 Channel 接口保持兼容，通道扰动仍使用 batch 平均 SNR 的标量。
    """
    def __init__(self, args):
        super(ETMC_channel_snr, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder_no_pretrain(args)
        self.depthenc = ImageEncoder_no_pretrain(args)
        self.channel = Channel('awgn')

        depth_feat_dim = args.img_hidden_sz * args.num_image_embeds
        rgb_feat_dim = args.img_hidden_sz * args.num_image_embeds

        # 将视觉特征编码到 channel 空间
        last_dim = rgb_feat_dim
        self.rgbchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.rgbchannel_enc.append(nn.Linear(last_dim, hidden))
            self.rgbchannel_enc.append(nn.ReLU())
            self.rgbchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.rgbchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        last_dim = depth_feat_dim
        self.depthchannel_enc = nn.ModuleList()
        for hidden in args.channel_hidden:
            self.depthchannel_enc.append(nn.Linear(last_dim, hidden))
            self.depthchannel_enc.append(nn.ReLU())
            self.depthchannel_enc.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.depthchannel_enc.append(nn.Linear(last_dim, args.channel_size))

        # SNR 嵌入与融合
        self.snr_embed_dim: int = int(getattr(args, 'snr_embed_dim', 64))
        self.snr_input_method: str = str(getattr(args, 'snr_input_method', 'concat'))

        self.snr_embed = nn.Sequential(
            nn.Linear(1, self.snr_embed_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

        # 按融合策略配置分类头的输入维度与必要的投影/融合层
        if self.snr_input_method == 'concat':
            depth_clf_in = args.channel_size + self.snr_embed_dim
            rgb_clf_in = args.channel_size + self.snr_embed_dim
            pseudo_in = (args.channel_size + self.snr_embed_dim) * 2
            self.fuse_depth = None
            self.fuse_rgb = None
            self.snr_to_channel = None
        elif self.snr_input_method == 'add':
            depth_clf_in = args.channel_size
            rgb_clf_in = args.channel_size
            pseudo_in = args.channel_size * 2
            self.snr_to_channel = nn.Linear(self.snr_embed_dim, args.channel_size)
            self.fuse_depth = None
            self.fuse_rgb = None
        else:  # 'mlp'
            depth_clf_in = args.channel_size
            rgb_clf_in = args.channel_size
            pseudo_in = args.channel_size * 2
            self.fuse_depth = nn.Sequential(
                nn.Linear(args.channel_size + self.snr_embed_dim, args.channel_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.fuse_rgb = nn.Sequential(
                nn.Linear(args.channel_size + self.snr_embed_dim, args.channel_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.snr_to_channel = None

        # 分类头（evidential）
        self.clf_depth = nn.ModuleList()
        last_dim = depth_clf_in
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(last_dim, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_depth.append(nn.Linear(last_dim, args.n_classes))

        self.clf_rgb = nn.ModuleList()
        last_dim = rgb_clf_in
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(last_dim, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_rgb.append(nn.Linear(last_dim, args.n_classes))

        self.clf_pseudo = nn.ModuleList()
        last_dim = pseudo_in
        for hidden in args.hidden:
            self.clf_pseudo.append(nn.Linear(last_dim, hidden))
            self.clf_pseudo.append(nn.ReLU())
            self.clf_pseudo.append(nn.Dropout(args.dropout))
            last_dim = hidden
        self.clf_pseudo.append(nn.Linear(last_dim, args.n_classes))

    def _forward_mlp(self, x: torch.Tensor, mlp: nn.ModuleList) -> torch.Tensor:
        for layer in mlp:
            x = layer(x)
        return x

    def DS_Combin_two(self, alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        alpha = {0: alpha1, 1: alpha2}
        b, S, E, u = {}, {}, {}, {}
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        bb_sum = torch.sum(bb, dim=(1, 2))
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        S_a = self.args.n_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def _fuse_with_snr(self, feat: torch.Tensor, snr_embed: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'concat':
            return torch.cat([feat, snr_embed], dim=-1)
        if mode == 'add':
            assert self.snr_to_channel is not None
            snr_proj = self.snr_to_channel(snr_embed)
            return feat + snr_proj
        # mlp
        return feat

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, snr: torch.Tensor):
        # 规范化 snr 张量形状为 [B, 1]
        if snr is None:
            # 回退到固定 SNR
            # 构造与 batch 匹配的常数 snr 向量用于嵌入
            batch_size = rgb.shape[0]
            snr = torch.full((batch_size,), float(getattr(self.args, 'channel_snr', 20.0)), dtype=torch.float32, device=rgb.device)
        if snr.dim() == 1:
            snr_input = snr.view(-1, 1)
        else:
            snr_input = snr

        # 计算 batch 平均 SNR 作为 channel 噪声层的标量（保证与已有接口兼容）
        snr_scalar = float(torch.mean(snr_input).item())

        # 视觉特征 -> channel 编码 -> 通道扰动
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        depth_feat = self.channel(depth_feat, snr_scalar)

        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        rgb_feat = self.channel(rgb_feat, snr_scalar)

        # SNR 嵌入
        snr_embed = self.snr_embed(snr_input)

        # 按策略融合
        if self.snr_input_method == 'mlp':
            assert self.fuse_depth is not None and self.fuse_rgb is not None
            depth_feat_fused = self.fuse_depth(torch.cat([depth_feat, snr_embed], dim=-1))
            rgb_feat_fused = self.fuse_rgb(torch.cat([rgb_feat, snr_embed], dim=-1))
        else:
            depth_feat_fused = self._fuse_with_snr(depth_feat, snr_embed, self.snr_input_method)
            rgb_feat_fused = self._fuse_with_snr(rgb_feat, snr_embed, self.snr_input_method)

        # per-modality logits -> evidences -> alphas
        depth_logits = self._forward_mlp(depth_feat_fused, self.clf_depth)
        rgb_logits = self._forward_mlp(rgb_feat_fused, self.clf_rgb)

        depth_evidence = F.softplus(depth_logits)
        rgb_evidence = F.softplus(rgb_logits)
        depth_alpha = depth_evidence + 1
        rgb_alpha = rgb_evidence + 1

        # pseudo 分支基于融合后的特征
        pseudo_feat = torch.cat([rgb_feat_fused, depth_feat_fused], dim=-1)
        pseudo_logits = self._forward_mlp(pseudo_feat, self.clf_pseudo)
        pseudo_evidence = F.softplus(pseudo_logits)
        pseudo_alpha = pseudo_evidence + 1

        depth_rgb_alpha = self.DS_Combin_two(self.DS_Combin_two(depth_alpha, rgb_alpha), pseudo_alpha)
        return depth_alpha, rgb_alpha, pseudo_alpha, depth_rgb_alpha