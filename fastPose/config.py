import numpy as np
from easydict import EasyDict as edict

config = edict()

config.USE_POSE_POOLING=True
# image processing config
config.EPS = 1e-14
config.PIXEL_MEANS = np.array([[[123.68, 116.779, 103.939]]])
config.SCALES = [(600, 1000)]  # first is scale (the shorter side); second is max size
config.IMAGE_STRIDE = 0

config.TRAIN = edict()


config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 1.2
config.TRAIN.BG_THRESH_LOW = 0.7

#pose
config.TRAIN.POSE_NORMALIZATION_PRECOMPUTED= False


config.TRAIN.POSEWEIGHT=np.ones((28,),dtype=np.float32)


# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 1.0
config.TRAIN.RPN_NEGATIVE_OVERLAP = 1.1
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = np.ones((28,),dtype=np.float32)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.RPN_NMS_THRESH = 0.3
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 6000
config.TRAIN.RPN_MIN_SIZE = 16
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False



config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.RPN_NMS_THRESH = 0.3
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = 16

# RCNN nms
config.TEST.NMS = 0.3