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

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # extract and channel-encode depth features
        depth_feat = self.depthenc(depth)
        depth_feat = torch.flatten(depth_feat, start_dim=1)
        depth_feat = self._forward_mlp(depth_feat, self.depthchannel_enc)
        depth_feat = self.channel(depth_feat, self.args.channel_snr)

        # extract and channel-encode rgb features
        rgb_feat = self.rgbenc(rgb)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)
        rgb_feat = self._forward_mlp(rgb_feat, self.rgbchannel_enc)
        rgb_feat = self.channel(rgb_feat, self.args.channel_snr)

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