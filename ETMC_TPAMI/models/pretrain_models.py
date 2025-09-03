
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
# from .dat.dat_blocks import DAttentionBaseline
# from .dat.utils import off_diagonal
# from .dat.utils import accuracy

''' DeCUR '''
class DeCUR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # backbone
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True,pretrained=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True,pretrained=True)
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True,pretrained=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True,pretrained=True)
        elif args.backbone == 'vits16':
            import timm
            self.backbone_1 = timm.create_model('vit_small_patch16_224',pretrained=True)
            self.backbone_2 = timm.create_model('vit_small_patch16_224',pretrained=True)
        elif args.backbone == 'mit_b2':
            from .segformer.encoders.segformer import mit_b2
            self.backbone_1 = mit_b2(num_classes=2048)
            self.backbone_2 = mit_b2(num_classes=2048)
            self.backbone_1.init_weights(pretrained=args.pretrained)
            self.backbone_2.init_weights(pretrained=args.pretrained)
        elif args.backbone == 'mit_b5':
            from .segformer.encoders.segformer import mit_b5
            self.backbone_1 = mit_b5(num_classes=2048)
            self.backbone_2 = mit_b5(num_classes=2048)
            self.backbone_1.init_weights(pretrained=args.pretrained)
            self.backbone_2.init_weights(pretrained=args.pretrained)

        # if 'resnet' in args.backbone:
        #     if args.mode==['s1','s2c']:
        #         self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
        #         self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)        
        #     self.backbone_1.fc = nn.Identity()
        #     self.backbone_2.fc = nn.Identity()
        # elif 'vit' in args.backbone:
        #     if args.mode==['s1','s2c']:
        #         self.backbone_1.patch_embed.proj = nn.Conv2d(2, 384, kernel_size=(16, 16), stride=(16, 16))
        #         self.backbone_2.patch_embed.proj = nn.Conv2d(13, 384, kernel_size=(16, 16), stride=(16, 16))
        #     self.backbone_1.head = nn.Identity()
        #     self.backbone_2.head = nn.Identity()

        # deformable attention
        if args.rda:
            from .dat.dat_blocks import DAttentionBaseline

            self.da1_l3 = DAttentionBaseline(
                q_size=(14,14), kv_size=(14,14), n_heads=8, n_head_channels=128, n_groups=4,
                attn_drop=0, proj_drop=0, stride=2, 
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=5, log_cpb=False
            )

            self.da1_l4 = DAttentionBaseline(
                q_size=(7,7), kv_size=(7,7), n_heads=16, n_head_channels=128, n_groups=8,
                attn_drop=0, proj_drop=0, stride=1, 
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=3, log_cpb=False
            )

            self.da2_l3 = DAttentionBaseline(
                q_size=(14,14), kv_size=(14,14), n_heads=8, n_head_channels=128, n_groups=4,
                attn_drop=0, proj_drop=0, stride=2, 
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=5, log_cpb=False
            )

            self.da2_l4 = DAttentionBaseline(
                q_size=(7,7), kv_size=(7,7), n_heads=16, n_head_channels=128, n_groups=8,
                attn_drop=0, proj_drop=0, stride=1, 
                offset_range_factor=-1, use_pe=True, dwc_pe=False,
                no_off=False, fixed_pe=False, ksize=3, log_cpb=False
            )

        # projector
        if args.backbone == 'resnet50':
            sizes = [2048] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'resnet18':
            sizes = [1000] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'vits16':
            sizes = [384] + list(map(int, args.projector.split('-')))
        elif 'mit' in args.backbone:
            sizes = [2048] + list(map(int, args.projector.split('-')))

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector1 = nn.Sequential(*layers)
        self.projector2 = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def bt_loss_cross(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        # torch.distributed.all_reduce(c)

        dim_c = self.args.dim_common
        c_c = c[:dim_c,:dim_c]
        c_u = c[dim_c:,dim_c:]

        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()
        
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        
        loss_c = on_diag_c + self.args.lambd * off_diag_c
        loss_u = on_diag_u + self.args.lambd * off_diag_u
        
        return loss_c,on_diag_c,off_diag_c,loss_u,on_diag_u,off_diag_u   


    def bt_loss_single(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        # torch.distributed.all_reduce(c) 不适用于单个Gpu

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss,on_diag,off_diag

    def forward_resnet_da(self, x, backbone, da_l3, da_l4):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x1,pos1,ref1 = da_l3(x)
        x = x + x1
        x = backbone.layer4(x)
        x2,pos2,ref2 = da_l4(x)
        x = x + x2

        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = backbone.fc(x)
        
        return x

    def forward(self, y1_1,y1_2,y2_1,y2_2):

        if self.args.rda:
            f1_1 = self.forward_resnet_da(y1_1,self.backbone_1,self.da1_l3,self.da1_l4)
            f1_2 = self.forward_resnet_da(y1_2,self.backbone_1,self.da1_l3,self.da1_l4)
            f2_1 = self.forward_resnet_da(y2_1,self.backbone_2,self.da1_l3,self.da1_l4)
            f2_2 = self.forward_resnet_da(y2_2,self.backbone_2,self.da1_l3,self.da1_l4)
        else:
            f1_1 = self.backbone_1(y1_1)
            f1_2 = self.backbone_1(y1_2)
            f2_1 = self.backbone_2(y2_1)
            f2_2 = self.backbone_2(y2_2)  

        z1_1 = self.projector1(f1_1)
        z1_2 = self.projector1(f1_2)
        z2_1 = self.projector2(f2_1)
        z2_2 = self.projector2(f2_2)         

        loss1, on_diag1, off_diag1 = self.bt_loss_single(z1_1,z1_2)
        loss2, on_diag2, off_diag2 = self.bt_loss_single(z2_1,z2_2)        
        loss12_c, on_diag12_c, off_diag12_c, loss12_u, on_diag12_u, off_diag12_u = self.bt_loss_cross(z1_1,z2_1)
        loss12 = (loss12_c + loss12_u) / 2.0

        return loss1,loss2,loss12,on_diag12_c


class SimCLR(nn.Module):

    LARGE_NUMBER = 1e9

    def __init__(self,args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
            ndim_proj = 2048
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=True)
            ndim_proj = 1000
        elif args.backbone == 'vits16':
            import timm
            self.backbone_1 = timm.create_model('vit_small_patch16_224',pretrained=True)
            self.backbone_2 = timm.create_model('vit_small_patch16_224',pretrained=True)
            ndim_proj = 384
            
        # if 'resnet' in args.backbone:
        #     if args.mode==['s1','s2c']:
        #         self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
        #         self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)        
        #     self.backbone_1.fc = nn.Identity()
        #     self.backbone_2.fc = nn.Identity()
        # elif 'vit' in args.backbone:
        #     if args.mode==['s1','s2c']:
        #         self.backbone_1.patch_embed.proj = nn.Conv2d(2, 384, kernel_size=(16, 16), stride=(16, 16))
        #         self.backbone_2.patch_embed.proj = nn.Conv2d(13, 384, kernel_size=(16, 16), stride=(16, 16))
        #     self.backbone_1.head = nn.Identity()
        #     self.backbone_2.head = nn.Identity()
                    
        self.projector1 = nn.Sequential(nn.Linear(ndim_proj, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
        self.projector2 = nn.Sequential(nn.Linear(ndim_proj, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
            
        self.tau = 1.0
        self.multiplier = 2
        self.distributed = False
        self.norm = 1.0            
        
    def forward(self,x1,x2):
        z1 = self.projector1(self.backbone_1(x1))
        z2 = self.projector2(self.backbone_2(x2))        
        z = torch.cat((z1,z2),dim=0)

        n = z.shape[0]
        assert n % self.multiplier == 0
        
        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        
        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]
        
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -1e4

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -1e4
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)                
    
        return loss#, acc.sum()/(acc.shape[0])
    
    
class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True)
        elif args.backbone == 'resnet18':
            self.backbone_1 = torchvision.models.resnet18(zero_init_residual=False)
            self.backbone_2 = torchvision.models.resnet18(zero_init_residual=False)
        elif args.backbone == 'vits16':
            import timm
            self.backbone_1 = timm.create_model('vit_small_patch16_224',pretrained=True)
            self.backbone_2 = timm.create_model('vit_small_patch16_224',pretrained=True)
            
        # if 'resnet' in args.backbone:
        #     if args.mode==['s1','s2c']:
        #         self.backbone_1.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)            
        #         self.backbone_2.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)        
        #     self.backbone_1.fc = nn.Identity()
        #     self.backbone_2.fc = nn.Identity()
        # elif 'vit' in args.backbone:
        #     if args.mode==['s1','s2c']:
        #         self.backbone_1.patch_embed.proj = nn.Conv2d(2, 384, kernel_size=(16, 16), stride=(16, 16))
        #         self.backbone_2.patch_embed.proj = nn.Conv2d(13, 384, kernel_size=(16, 16), stride=(16, 16))
        #     self.backbone_1.head = nn.Identity()
        #     self.backbone_2.head = nn.Identity()

        # projector
        if args.backbone == 'resnet50':
            sizes = [2048] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'resnet18':
            sizes = [1000] + list(map(int, args.projector.split('-')))
        elif args.backbone == 'vits16':
            sizes = [384] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector1 = nn.Sequential(*layers)
        self.projector2 = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector1(self.backbone_1(y1))
        z2 = self.projector2(self.backbone_2(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss#,on_diag,off_diag