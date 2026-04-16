import torch
import os
import os.path
import numpy as np
import pandas
import random
import glob
from collections import OrderedDict

from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from bisect import bisect_left, bisect_right



def list_sequences(root, set_ids):
    """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)

    args:
        root: Root directory to TrackingNet
        set_ids: Sets (0-11) which are to be used

    returns:
        list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
    """
    sequence_list = []

    for s in set_ids:
        anno_dir = os.path.join(root, "TRAIN_" + str(s), "anno")

        sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]
        sequence_list += sequences_cur_set

    return sequence_list


class trackingnet_det(BaseVideoDataset):
    """ TrackingNet dataset.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """
    def __init__(self, root=None, image_loader=jpeg4py_loader, set_ids=None, data_fraction=None, min_score = 0.5):

        root = env_settings().trackingnet_dir if root is None else root
        super().__init__('trackingnet_det', root, image_loader)
        self.det_root = env_settings().det_anno_dir
        if set_ids is None:
            set_ids = [i for i in range(12)]

        self.set_ids = set_ids

        # Keep a list of all videos. Sequence list is a list of tuples (set_id, video_name) containing the set_id and
        # video_name for each sequence
        self.sequence_list = list_sequences(self.root, self.set_ids)


        L1 = len(self.sequence_list)
        self.threshold = min_score
        self.sequence_list = self._get_valid_seq()
        L2 = len(self.sequence_list)
        print('TrackingNet : Detection score: %s, keep ratio: %f'%(self.threshold, L2/L1))

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_to_class_map, self.seq_per_class = self._load_class_info()

        # we do not have the class_lists for the tracking net
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def _load_class_info(self):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        class_map_path = os.path.join(ltr_path, 'data_specs', 'trackingnet_classmap.txt')

        with open(class_map_path, 'r') as f:
            seq_to_class_map = {seq_class.split('\t')[0]: seq_class.rstrip().split('\t')[1] for seq_class in f}

        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = seq_to_class_map.get(seq[1], 'Unknown')
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_to_class_map, seq_per_class

    def get_name(self):
        return 'trackingnet_det'

    def has_class_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_det_anno(self, seq_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        det_file = os.path.join(self.det_root,'trackingnet',"TRAIN_" + str(set_id), vid_name+'.txt')
        try:
            det_gt = pandas.read_csv(det_file, delimiter=' ', header=None, dtype=np.float32, \
                                    na_filter=False, low_memory=False).values
        except:
            det_gt = np.zeros((0,4))
        return np.array(det_gt)

    def _get_sequence_length(self, seq_id): 
        # TODO: generate a list in dataset
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        frame_path = os.path.join(self.root, "TRAIN_" + str(set_id), "frames", vid_name, "*.jpg")
        img_lst  = glob.glob(frame_path)
        return len(img_lst)

    def _get_valid_seq(self):
        max_dic = {}
        for set_id in self.set_ids:
            max_file =  os.path.join(self.det_root,'trackingnet','TRAIN_%d_max.txt'%set_id)
            max_vec = pandas.read_csv(max_file, delimiter=' ', header=None).values
            #print(len(max_vec))
            for v in max_vec:
                max_dic[v[0]] = v[1]
        ret = []
        for seq in self.sequence_list:
            if max_dic[seq[1]] > self.threshold:
                ret.append(seq)
        return ret

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

    def _get_frame(self, seq_id, frame_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        frame_path = os.path.join(self.root, "TRAIN_" + str(set_id), "frames", vid_name, str(frame_id) + ".jpg")
        return self.image_loader(frame_path)

    def _get_class(self, seq_id):
        seq_name = self.sequence_list[seq_id][1]
        return self.seq_to_class_map[seq_name]

    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None, multi_box=False):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

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
        obj_class = self._get_class(seq_id)

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
