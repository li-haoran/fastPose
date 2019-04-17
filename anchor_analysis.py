import argparse
import logging
import os
import pprint
import mxnet as mx
import numpy as np

from fastPose.config import config
from fastPose.symbol import *
from fastPose.dataset.MPII import MPII
from fastPose.core import callback,metric
from fastPose.core.loader import AnchorLoader
from fastPose.core.module import MutableModule
from fastPose.utils.load_model import load_param
from fastPose.core.tester import vis_all_detection
import matplotlib.pyplot as plt
from fastPose.processing import image_processing

def anchor_ananlysis(args,ctx):
    config.TRAIN.BATCH_IMAGES = 1
    config.TRAIN.BATCH_ROIS = 128
    config.TRAIN.END2END = True
    config.TRAIN.BG_THRESH_LOW = 1.2
    sym = eval('get_'+args.network+'_train')()

    feat_sym= sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    batch_size = 1
    input_batch_size = 1



    #imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    imdb = MPII(args.image_set, args.anno_file, args.dataset_path)
    roidb = imdb.gt_roidb()


    #load traing data
    train_data = AnchorLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=True,
                              ctx=ctx, work_load_list=args.work_load_list)

    batch = train_data.next()

    batch.label[0]
    for i in range(100):
        batch = train_data.next()
        image=batch.data[0].asnumpy()
        image=image_processing.transform_inverse(image,config.PIXEL_MEANS)
        plt.figure(0)
        plt.imshow(image)

        scale=batch.data[1].asnumpy()[0,2]
        data=batch.label[0].asnumpy()[0]
        
        non_zeros= np.sum(data>0)
        
        data=data.reshape(27,train_data.provide_label[2][1][2],train_data.provide_label[2][1][3])
        data=np.max(data,axis=0)
        #non_zeros= np.argwhere(data>0)
        print '--------',non_zeros#,non_zeros.shape[0]
        plt.figure(1)
        plt.imshow(data)
        plt.show()

    



def parse_args():
    parser = argparse.ArgumentParser(description='Train a Region Proposal Network')
    # general
    parser.add_argument('--network', help='network name',
                        default='vgg', type=str)
    parser.add_argument('--dataset', help='dataset name',
                        default='MPII', type=str)
    parser.add_argument('--image_set', help='image_set name',
                        default=r'D:\dataset\evalMPII\mpii_human_pose_v1\images', type=str)
    parser.add_argument('--anno_file', help='output data folder',
                        default=r'D:\documents\mx-rcnn\mx-rcnn\fastPose\tools\Final_MPI_annotation.json', type=str)
    parser.add_argument('--dataset_path', help='dataset path',
                        default=os.path.join('D:/dataset', 'evalMPII'), type=str)
    parser.add_argument('--work_load_list', help='work load for different devices',
                        default=None, type=list)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    ctx = [mx.cpu()]
    anchor_ananlysis(args,ctx)

if __name__ == '__main__':
    main()