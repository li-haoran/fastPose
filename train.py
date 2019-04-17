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

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
              lr=0.001, lr_step=50000):

    #set up logger
    head = '%(asctime)-15s Node[0] %(message)s'
    log_file='MPII_fastPose.log'
    log_dir='model/'
    log_file_full_name = os.path.join(log_dir, log_file)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('start with arguments %s', args)

    # setup config
    config.TRAIN.BATCH_IMAGES = 1
    config.TRAIN.BATCH_ROIS = 128
    config.TRAIN.END2END = True
    #config.TRAIN.BG_THRESH_LOw = 0.65

    # load symbol
    sym = eval('get_'+args.network+'_train')()

    feat_sym= sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)


    #imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    imdb = MPII(args.image_set, args.anno_file, args.dataset_path)
    roidb = imdb.gt_roidb()

    #load traing data
    train_data = AnchorLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=True,
                              ctx=ctx, work_load_list=args.work_load_list)

    batch = train_data.next()
    # infer max shape
    max_data_shape = [('data', (input_batch_size, 3, 1000, 1000))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_poses', (input_batch_size, 100, 29)))
    print 'providing maximum shape', max_data_shape, max_label_shape

    # load pretrained
    arg_params, aux_params = load_param(pretrained, epoch, convert=True)
    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    #mx.viz.plot_network(sym, shape=data_shape_dict).view()
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print 'output shape'
    pprint.pprint(out_shape_dict)

    # initialize params
    if not args.resume:
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        #arg_params['rpn_conv_3x3_1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_1_weight'])
        #arg_params['rpn_conv_3x3_1_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_1_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_pose_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_pose_pred_weight'])
        arg_params['rpn_pose_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_pose_pred_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        arg_params['pose_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['pose_pred_weight'])
        arg_params['pose_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['pose_pred_bias'])


    #check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)


    # create solver
    fixed_param_prefix = ['conv1', 'conv2']
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=args.work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)
    # decide training params
    # metric
    rpn_eval_metric = metric.RPNAccMetric()
    rpn_pose_metric = metric.RPNL1LossMetric()
    eval_metric = metric.RCNNAccMetric()
    pose_metric = metric.RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_pose_metric, eval_metric,  pose_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    epoch_end_callback = callback.do_checkpoint(prefix)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': mx.lr_scheduler.FactorScheduler(lr_step, 0.1),
                        'rescale_grad': (1.0 / batch_size)}

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=args.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)

    


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
                        default=r'D:\dataset\evalMPII\mpii_human_pose_v1\MPI_annotations.json', type=str)
    parser.add_argument('--dataset_path', help='dataset path',
                        default=os.path.join('D:/dataset', 'evalMPII'), type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kvstore', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--work_load_list', help='work load for different devices',
                        default=None, type=list)
    
    parser.add_argument('--resume', help='continue training', action='store_true')
    # e2e
    parser.add_argument('--gpus', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix',
                        default=os.path.join('model', 'vgg16'), type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', help='new model prefix',
                        default=os.path.join('model', 'e2e'), type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training',
                        default=10, type=int)
    parser.add_argument('--lr', help='base learning rate', default=0.001, type=float)
    parser.add_argument('--lr_step', help='learning rate step', default=50000, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print 'Called with argument:', args
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_net(args, ctx, args.pretrained, args.epoch, args.prefix, args.begin_epoch, args.end_epoch,
              lr=args.lr, lr_step=args.lr_step)

if __name__ == '__main__':
    main()