import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTModel

from aggregators.netvlad import NetVLAD
from aggregators.salad import SALAD
from aggregators.boq import BoQ

class VPRmodel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arch_name = args.backbone
        self.aggregator_name = args.aggregator
        self.backbone = get_backbone(args)
        self.aggregator = get_aggregator(args)
        if not self.aggregator:  #  ImAge with no extra aggregator
            assert args.num_learnable_aggregation_tokens >= 0
            self.num_learnable_aggregation_tokens = args.num_learnable_aggregation_tokens
            self.learnable_aggregation_tokens = nn.Parameter(torch.zeros(1, args.num_learnable_aggregation_tokens, 768)) if args.num_learnable_aggregation_tokens else None
            self.insertion_pos = args.freeze_te
            if self.learnable_aggregation_tokens is not None:
                torch.nn.init.normal_(self.learnable_aggregation_tokens, std=1e-6)

    def forward(self, x):
        if self.arch_name.startswith("dinov2"):
            # ImAge with no extra aggregator
            if not self.aggregator:
                x = self.backbone.prepare_tokens_with_masks(x)
                B = x.shape[0]

                # Frozen transformer blocks at the front process the original tokens as usual
                for i in range(self.insertion_pos):
                    x = self.backbone.blocks[i](x)

                # Add aggregation tokens before the first trainable transformer block
                x = torch.cat([self.learnable_aggregation_tokens.expand(B,-1,-1), x], dim=1)

                # Subsequent trainable transformer blocks jointly process our aggregation tokens and other original tokens, 
                # thus achieving global interactions and aggregating useful global information within other tokens into our aggregation tokens  
                for i in range(self.insertion_pos, len(self.backbone.blocks)):
                    x = self.backbone.blocks[i](x)
                
                x_norm = self.backbone.norm(x)

                # Directly take aggregation tokens as the final global representaion
                x_agg = x_norm[:, :self.num_learnable_aggregation_tokens, :]
                x_g = x_agg.flatten(1)

            # use explicit aggregators
            else:
                x = self.backbone(x)
                B,P,D = x["x_norm"].shape
                H = W = int(math.sqrt(P-1))
                x_c = x["x_norm_clstoken"].squeeze(1)
                x_p = x["x_norm_patchtokens"].view(B,H,W,D).permute(0,3,1,2)
                if self.aggregator_name == "salad":
                    x_g = self.aggregator((x_p, x_c))
                else:
                    x_g = self.aggregator(x_p)

        x = torch.nn.functional.normalize(x_g, p=2, dim=-1)
        return x
    
def get_aggregator(args):
    if not args.aggregator:
        aggregator = None
    elif args.aggregator == "netvlad":
        aggregator = NetVLAD(clusters_num=8,dim=768)
    elif args.aggregator == "salad":
        aggregator = SALAD(num_channels=768)
        args.features_dim = 8448
    elif args.aggregator == "boq":
        aggregator = BoQ(in_channels=768,proj_channels=384,num_layers=2,num_queries=64,row_dim=32)
        args.features_dim = 12288
    return aggregator

def get_backbone(args):
    if args.backbone.startswith("dinov2"):
        from backbone.vision_transformer import vit_base
        backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0, num_register_tokens=4)
        if not args.resume:
            model_dict = backbone.state_dict()
            state_dict = torch.load(args.foundation_model_path)
            model_dict.update(state_dict.items())
            backbone.load_state_dict(model_dict)
        if args.freeze_te:
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.blocks.named_children():
                if int(name) >= args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
    
    return backbone