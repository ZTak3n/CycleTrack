
import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import lib.train.data.transforms as mytfm

def stack_tensors(x):
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x)
    return x


class BaseProcessing:
    __doc__ = " Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it\n     through the network. For example, it can be used to crop a search region around the object, apply various data\n     augmentations, etc."

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
       
        self.transform = {'template':transform if (template_transform is None) else template_transform, 
         'search':transform if (search_transform is None) else search_transform, 
         'joint':joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class CycleTrackProcessing(BaseProcessing):

    def __init__(self, search_area_factor, output_sz, temp_transform, T_S_T, mode='pair', settings=None, *args, **kwargs):
        (super().__init__)(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.output_sz["template_back"] = self.output_sz["search"]
        self.mode = mode
        self.settings = settings
        self.temp_transform = temp_transform
        self.transform["template_back"] = self.transform["search"]
        self.box_num = 10
        self.T_S_T = T_S_T

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        if self.transform["joint"] is not None:
            data["template_images"], data["template_anno"], data["template_masks"] = self.transform["joint"](image=(data["template_images"]),
              bbox=(data["template_anno"]),
              mask=(data["template_masks"]))
            data["search_images"], data["search_anno"], data["search_masks"] = self.transform["joint"](image=(data["search_images"]),
              bbox=(data["search_anno"]),
              mask=(data["search_masks"]),
              new_roll=False)
        
        
        if self.T_S_T: # template-search-template : copy the second template from the first template 
                flip_img, flip_anno, flip_mask = self.temp_transform(image=(data["template_images"]), bbox=(data["template_anno"]),
                  mask=(data["template_masks"]))
                data["template_back_images"] = flip_img
                data["template_back_anno"] = flip_anno
                data["template_back_masks"] = flip_mask
                #data["search_images"], data["search_anno"], data["search_masks"] = self.temp_transform(image=(data["search_images"]), bbox=(data["search_anno"]),
                #  mask=(data["search_masks"]))
        else:
                data["search_images"], data["search_anno"], data["search_masks"] = self.temp_transform(image=(data["search_images"]), bbox=(data["search_anno"]),
                  mask=(data["search_masks"]))
        s_list = [
         "template", "search", "template_back"]
        for s in s_list:
            if not self.mode == "sequence":
                assert len(data[s + "_images"]) == 1, "In pair mode, num train/test frames must be 1"
            crops, boxes, _, mask_crops = prutils.resize((data[s + "_images"]), (data[s + "_anno"]),
              (self.output_sz[s]),
              masks=(data[s + "_masks"]))
            if "template" in s:
                x1, y1, x2, y2 = (
                 boxes[-1][0][0], boxes[-1][0][1],
                 boxes[-1][0][0] + boxes[-1][0][2], boxes[-1][0][1] + boxes[-1][0][3])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1, x2), min(1, y2)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0 or w * h < 0.0001:
                    data["valid"] = False
                    return data
            data[s + "_images"], data[s + "_anno"], data[s + "_masks"] = self.transform[s](image=crops,
              bbox=boxes,
              mask=mask_crops,
              joint=False)
            if "search" in s:
                align_lst = []
                for anno in data[s + "_anno"]:
                    anno = torch.tile(anno, (self.box_num, 1))
                    align_lst.append(anno[:self.box_num, :])
                data[s + "_anno"] = align_lst

        data["valid"] = True
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        elif self.mode == "sequence":
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        return data
