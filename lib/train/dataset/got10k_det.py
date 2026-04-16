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


class Got10k_det(BaseVideoDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None, min_score = 0.5):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().got10k_dir if root is None else root
        super().__init__('GOT10k_det', root, image_loader)

        self.det_root = env_settings().det_anno_dir
        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

            if split == 'vottrain':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_train_split.txt')
            else:
                raise ValueError('Unknown split name.')
            
            seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        L1 = len(self.sequence_list)
        self.threshold = min_score -0.2
        self.sequence_list = self._get_valid_seq()
        L2 = len(self.sequence_list)
        print('Got10k : Detection score: %s, keep ratio: %f'%(self.threshold, L2/L1))


        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()
        
        
        #_, anno,_ = self.get_frames_det(2075, [0, 40, 80])
        #_, anno, _ = self.get_frames(2075, [0, 40, 80])
        #det_id = self._get_det_id(self.sequence_list[2075], 39)
        #self._read_det_anno(self.sequence_list[2075],det_id)

    def get_name(self):
        return 'got10k_det'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _get_valid_seq(self):
        max_file =  os.path.join(self.det_root,'got10k_max.txt')
        max_vec = pandas.read_csv(max_file, delimiter=' ', header=None).values
        max_dic = {v[0]:v[1] for v in max_vec}
        ret = []
        for seq in self.sequence_list:
            if max_dic[seq] > self.threshold:
                ret.append(seq)
        return ret

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]
        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list


    def _read_det_anno(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        det_file = os.path.join(self.det_root,'got10k', seq_name+'.txt')
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
        img_lst  = glob.glob(os.path.join(self.root, seq_name,'*.jpg'))
        return len(img_lst)

    def get_sequence_info(self, seq_id):
        # for DET dataset, visible is HAS A BOX, valid is ANY VALID BOX.
        
        det_gt = self._read_det_anno(seq_id)
        tot = self._get_sequence_length(seq_id)
        
        visible = torch.zeros(tot).bool()
        valid = torch.zeros(tot).bool()
        for i in range(det_gt.shape[0]):
            box = det_gt[i,:]
            idx =int(box[0]) -1
            visible[idx] = True
            if (box[3] > 0) & (box[4] > 0) & (box[5]> self.threshold):
                valid[idx] = True
        return {'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1))    

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None, multi_box=False):
        # Notice: a
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        
        anno_frames = {}
        det_gt = self._read_det_anno(seq_id) # start from 1.
        valid_idx = det_gt[:,5] > self.threshold
        det_gt = det_gt[valid_idx,:]
        
        box_lst = []
        for f_id in frame_ids:
            L_idx = bisect_left(det_gt[:,0], f_id + 1)
            R_idx = bisect_right(det_gt[:,0], f_id + 1)
            if R_idx > L_idx:
                
                #idx = random.randint(L_idx, R_idx-1)
                if not multi_box:
                    idx = det_gt[L_idx:R_idx, 5].argmax() + L_idx
                    box_lst.append(torch.tensor(det_gt[idx:idx+1,1:5]))
                else:
                    box_lst.append(torch.tensor(det_gt[L_idx:R_idx, 1:5]))
            else:
                box_lst.append(torch.zeros(1,4)) 
        anno_frames['bbox'] = box_lst
        return frame_list, anno_frames, obj_meta
            
        