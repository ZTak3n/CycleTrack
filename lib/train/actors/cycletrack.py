from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou, box_xywh_to_cxcywh
import torch 
import torch.nn as nn
import random

class CycleTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  
        self.search_num = cfg.DATA.SEARCH.NUMBER
        self.pz_loss = nn.MSELoss(reduction  = 'sum')
        self.is_EM = cfg.TRAIN.EM
        self.search_rate = cfg.TRAIN.SEARCH_RATE

    def __call__(self, data):
       

        pred_boxes, outputs= self.forward_pass(data)
        loss, status = self.compute_losses(pred_boxes, outputs, data['search_anno'], data['template_back_anno'])

        return loss, status


    def forward_pass(self, data):
        
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (n*b, c, h, w)
        search_list = search_img.split(self.bs,dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(self.bs,dim=0)
        template_back_img =  data['template_back_images'].view(-1, *data['template_back_images'].shape[2:])    
        template_back_list =  template_back_img.split(self.bs,dim=0)  
      
     
        input_seq = data['template_anno'][:,:,0, :].permute(1,0,2)
        input_seq = box_xywh_to_cxcywh(input_seq)
        work_list = template_list + search_list +  template_back_list
        pred_boxes = []
        
        
        for i in range(self.search_num + 1):
                outputs = self.net(images_list=work_list[i:i+2], seq =input_seq)
                pred_boxes.append(outputs)
                input_seq = outputs.detach() if self.is_EM else outputs
        
        
        pred_boxes = torch.cat(pred_boxes, dim = 1)
        return pred_boxes, outputs

    def get_giou(self, pred_box, gt_bbox):
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_box).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)       
        
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].view(-1, 4).clamp(min=0.0,max=1.0)
        
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        
        loss = self.loss_weight['giou'] * giou_loss  + self.loss_weight['l1'] * l1_loss
        mean_iou = iou.detach().mean()
        return loss, mean_iou

    def compute_losses(self, pred_boxes, outputs, search_boxes, template_back_box, return_status=True):
        
        gt_bbox = template_back_box.squeeze(2)
        pred_box = outputs
        if not self.is_EM:
            gt_search_bbox = search_boxes
            pred_search_box = pred_boxes[:,:-1, :].permute(1,0,2).unsqueeze(2)
        
            search_iou,_ = box_iou( box_xywh_to_xyxy(gt_search_bbox.view(-1,4)), 
                               box_cxcywh_to_xyxy(pred_search_box.tile((1,1,gt_search_bbox.shape[2],1)).view(-1,4)))
            search_iou = search_iou.view(1,gt_search_bbox.shape[1], gt_search_bbox.shape[2])
            s_idx = torch.argmax(search_iou, dim = -1)
            #iou_1 = search_iou[:,torch.arange(gt_search_bbox.shape[1]),s_idx[0]]
            iou_1 = search_iou[:,:,0]
            cmp_search_box = gt_search_bbox[:,torch.arange(gt_search_bbox.shape[1]),s_idx[0],:]
       
            pz, s_iou = self.get_giou(pred_search_box, cmp_search_box)
            pxz, iou = self.get_giou(pred_box, gt_bbox)

            lamda = 0 if random.random() > self.search_rate else 1
            loss =   pxz + lamda * pz
        else:
            loss, iou = self.get_giou(pred_box, gt_bbox)
        
        mean_iou = iou.detach().mean()

        if return_status:

            if not self.is_EM:
                status = {"Loss/total": loss.item(),
                      "pz":pz.item(),
                      "pxz":pxz.item(),
                      "IoU": mean_iou.item(),
                      "X_IoU": iou_1.detach().mean().item(),
                      "S_IOU": s_iou.detach().mean().item(),
                      }
                # You have to keep in mind that none of these IoU really reflect the tracker's quality 
                # — it's unsupervised, after all. 
            else:
                status = {"Loss/total": loss.item(),
                      "IoU": mean_iou.item(),
                      }
            return loss, status
        else:
            return loss

    def to(self, device):
        self.net.to(device)