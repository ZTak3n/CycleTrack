from easydict import EasyDict as edict
import yaml

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HIDDEN_DIM = 256 # hidden dimension for the decoder and vocabulary
cfg.MODEL.SEARCH_NUMBER = 1
cfg.MODEL.TEMPLATE_NUMBER = 1


cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16

cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = 'FCOS'
# MODEL.ENCODER

cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "vit_base_patch16" # encoder model
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "mae" # mae, default, or scratch
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.USE_CHECKPOINT = False # to save the memory.
# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.NHEADS = 8
cfg.MODEL.DECODER.DROPOUT = 0.1
cfg.MODEL.DECODER.DIM_FEEDFORWARD = 1024
cfg.MODEL.DECODER.DEC_LAYERS = 6
cfg.MODEL.DECODER.PRE_NORM = False

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.T_S_T = True # template-search-template cycle, otherwise should be template-search as usual tracker
cfg.TRAIN.EM = True # Expectation-Maxmization training
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1  
cfg.TRAIN.FREEZE_ENCODER = False 
cfg.TRAIN.ENCODER_OPEN = [] 
cfg.TRAIN.CE_WEIGHT = 1.0 
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.SEARCH_RATE = 1 # the keeping rate of search bbox in loss function
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.MIN_DET_SCORE = 0.8
cfg.DATA.MIN_FLOW_SCORE = 0.4
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
cfg.DATA.UNSUPERVISED_SCALE_JITTER = 2.0 
# UNSUPERVISED_SCALE_JITTER = 2.0 is highly recommond, as suggest in 
# “Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation”.

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1  
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 256
cfg.DATA.TEMPLATE.FACTOR = 4.0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 500
cfg.TEST.NUM_TEMPLATES = 1




def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


