
import torch
import math
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor

from lib.models.cycletrack.encoder import build_encoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from lib.models.layers.head import build_box_head

class CycleTrack(nn.Module):
    def __init__(self, encoder, decoder, hidden_dim, head_type,
                  num_frames=1, num_template=1):
        super().__init__()
        self.head_type = head_type
        self.encoder = encoder
        self.num_patch_x = self.encoder.body.num_patches_search   
        self.num_patch_z = self.encoder.body.num_patches_template  
        self.side_fx = int(math.sqrt(self.num_patch_x))           
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim) 
        self.decoder = decoder

        self.num_frames = num_frames         
        self.num_template = num_template     
        
        if head_type in ["CORNER", "CENTER", "FCOS"]:
            self.feat_sz_s = int(decoder.feat_sz)
            self.feat_len_s = int(decoder.feat_sz ** 2)

        num_patches = self.num_patch_x * self.num_frames #256*1


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=False) 
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


    def forward(self, images_list=None, xz=None, seq=None, mode="whole"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(images_list, seq)
        elif mode == "decoder":
            return self.forward_decoder(xz)
        elif mode == "whole":
            xz = self.forward_encoder(images_list, seq)
            return self.forward_decoder(xz)
        else:
            raise ValueError

    def forward_encoder(self, images_list, seq):
        xz = self.encoder(images_list, seq)
        return xz

    def forward_decoder(self, xz, return_att = False):

        xz_mem = xz[-1]
        dec_mem = xz_mem[:,0:self.num_patch_x * self.num_frames]
        hs = xz_mem[:, -1:] 
        
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  #[B,NL,D]
            hs = self.bottleneck(hs)

        #dec_mem [bs, feature_len, C]
        dec_opt = hs.permute(0, 2, 1)
        att = torch.matmul(dec_mem, dec_opt)   
        opt = (dec_mem.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, _ = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)


        L, pred_box = self.decoder(opt_feat, return_att)
        outputs_coord = box_xyxy_to_cxcywh(pred_box)
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        if not return_att:
            return outputs_coord_new
        else:
            return outputs_coord_new, L


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_cycletrack(cfg):
    encoder = build_encoder(cfg)
    box_head = build_box_head(cfg, cfg.MODEL.HIDDEN_DIM)
    
    model = CycleTrack(
        encoder,
        box_head,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,                  
        num_frames = cfg.MODEL.SEARCH_NUMBER,   
        num_template = cfg.MODEL.TEMPLATE_NUMBER,
        head_type = cfg.MODEL.HEAD.TYPE
    )

    return model


