import os
import os.path
import numpy as np
import torch
import csv
import pandas
import json
import glob
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from bisect import bisect_left, bisect_right


class imagenetvid_flow(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, seq_ids=None, data_fraction=None, min_score = 0.5):

        root = env_settings().imagenet_dir if root is None else root
        super().__init__('imagenetvid_flow', root, image_loader)

        self.flow_root = env_settings().flow_anno_dir
        self.ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        
        self.sequence_list = self._get_sequence_list()


        L1 = len(self.sequence_list)
        self.threshold = min_score
        self.sequence_list = self._get_valid_seq()
        L2 = len(self.sequence_list)
        print('ImagenetVID: Optical Flow score: %s, keep ratio: %f'%(self.threshold, L2/L1))


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self.sequence_meta_info = self._load_meta_info()

    def get_name(self):
        return 'imagenetvid_flow'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _get_valid_seq(self):
        max_file =  os.path.join(self.flow_root,'vid_max.txt')
        max_vec = pandas.read_csv(max_file, delimiter=' ', header=None).values
        max_dic = {v[0]:v[2] for v in max_vec}
        ret = []
        for seq in self.sequence_list:
            if seq in max_dic and max_dic[seq] > self.threshold:
                ret.append(seq)
        return ret

    def _read_meta(self, seq_path):
        object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta


    def _get_sequence_list(self):
        file_path = os.path.join(self.flow_root,'vid_max.txt')
        max_vec = pandas.read_csv(file_path, delimiter=' ', header=None).values
        dir_list = max_vec[:,0]
        return dir_list


    def _read_flow_anno(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        det_file = os.path.join(self.flow_root,'vid', seq_name+'.txt')
        try:
            det_gt = pandas.read_csv(det_file, delimiter=' ', header=None, dtype=np.float32, \
                                    na_filter=False, low_memory=False).values
        except:
            det_gt = np.zeros((0,4))
        return np.array(det_gt)


    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_sequence_length(self, seq_id): 
        # TODO: generate a list in dataset
        seq_name = self.sequence_list[seq_id]
        img_lst  = glob.glob(os.path.join(self.root, seq_name,'*.JPEG'))
        return len(img_lst)
    
    def get_sequence_info(self, seq_id):
        # for DET dataset, visible is HAS A BOX, valid is ANY VALID BOX.
        
        det_gt = self._read_flow_anno(seq_id)
        tot = self._get_sequence_length(seq_id)
        
        visible = torch.zeros(tot).bool()
        valid = torch.zeros(tot).bool()
        for i in range(det_gt.shape[0]):
            box = det_gt[i,:]
            idx =int(box[0]) 
            visible[idx] = True
            if (box[3] > 0) & (box[4] > 0) & (box[6]> self.threshold):
                valid[idx] = True
        return {'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06}.JPEG'.format(frame_id ))    

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None, multi_box = False):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        anno_frames = {}
        det_gt = self._read_flow_anno(seq_id) # start from start_id
        valid_idx = det_gt[:,6] > self.threshold
        det_gt = det_gt[valid_idx,:]
        
        
        box_lst = []
        for f_id in frame_ids:
            L_idx = bisect_left(det_gt[:,0], f_id)
            R_idx = bisect_right(det_gt[:,0], f_id )
            if R_idx > L_idx:
                if not multi_box:
                    idx = det_gt[L_idx:R_idx, 5].argmax() + L_idx
                    box_lst.append(torch.tensor(det_gt[idx:idx+1,1:5]))
                else:
                    box_lst.append(torch.tensor(det_gt[L_idx:R_idx, 1:5]))
            else:
                box_lst.append(torch.zeros(1,4)) 
        anno_frames['bbox'] = box_lst
        return frame_list, anno_frames, obj_meta