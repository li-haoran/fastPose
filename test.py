import argparse
import os
import mxnet as mx

from fastPose.config import config
from fastPose.symbol import *
from fastPose.dataset import MPII
from fastPose.core.loader import TestLoader
from fastPose.core.tester import Predictor,generate_proposals
from fastPose.utils.load_model import load_param
import cPickle

def test_rcnn(args, ctx, prefix, epoch,vis=False, shuffle=False, thresh=1e-3):
    # load symbol and testing data

    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = 6000
    config.TEST.RPN_POST_NMS_TOP_N = 300
    sym = eval('get_' + args.network + '_test')()
    imdb = MPII.MPII(args.image_set, args.anno_file, args.dataset_path)

    roidb = imdb.gt_roidb()
  
    # get test data iter
    test_data = TestLoader(roidb, batch_size=1, shuffle=shuffle)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)
    #arg_params=cPickle.load(open(prefix,'r'))
    #aux_params={}

    # check parameters
    param_names = [k for k in sym.list_arguments() + sym.list_auxiliary_states()
                   if k not in dict(test_data.provide_data) and 'label' not in k]
    missing_names = [k for k in param_names if k not in arg_params and k not in aux_params]
    if len(missing_names):
        print 'detected missing params', missing_names

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data]
    label_names = ['cls_prob_label']
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          test_data=test_data, arg_params=arg_params, aux_params=aux_params)

    generate_proposals(predictor,test_data,imdb,vis=True,thresh=0.)
    # start detection
    #pred_eval(predictor, test_data, imdb, vis=vis, thresh=thresh)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
   # general
    parser.add_argument('--network', help='network name',
                        default='vgg', type=str)
    parser.add_argument('--dataset', help='dataset name',
                        default='MPII', type=str)
    parser.add_argument('--image_set', help='image_set name',
                        default=r'D:\dataset\evalMPII\mpii_human_pose_v1\images', type=str)
    parser.add_argument('--anno_file', help='output data folder',
                        default=r'D:\dataset\evalMPII\mpii_human_pose_v1\MPI_annotations.json', type=str)
    parser.add_argument('--dataset_path', help='dataset path',
                        default=os.path.join('D:/dataset', 'evalMPII'), type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=r'D:\documents\mx-rcnn\mx-rcnn\model\new_e2e',type=str)
    parser.add_argument('--epoch', help='model to test with', default=0,type=int)
    parser.add_argument('--gpu', help='GPU device to test with', type=int)
    # rcnn
    parser.add_argument('--vis', dest='vis', help='turn on visualization', default=True,action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', default=True,action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly',default=True,
                        action='store_true')
    parser.add_argument('--proposal', dest='proposal', help='can be ss for selective search or rpn',
                        default='rpn', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ctx = mx.cpu()
    print args
    test_rcnn(args, ctx, args.prefix, args.epoch,
              vis=args.vis, shuffle=args.shuffle,thresh=args.thresh)

if __name__ == '__main__':
    main()
