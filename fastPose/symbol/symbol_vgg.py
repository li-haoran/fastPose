import mxnet as mx
import proposal
import proposal_target
import poses2rois
from fastPose.config import config
NUM_ANCHORS = 3*9

NORM_FACTOR=28*10

def get_vgg_conv(data,default_work_space=2048):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=default_work_space,name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=default_work_space,name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=default_work_space,name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=default_work_space,name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=default_work_space,name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=default_work_space,name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=default_work_space,name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=default_work_space,name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=default_work_space,name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=default_work_space,name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=default_work_space,name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=default_work_space,name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=default_work_space,name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    return relu5_3




def get_vgg_test(num_classes=2, num_anchors=NUM_ANCHORS):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")

    #rpn_conv1 = mx.symbol.Convolution(
    #    data=rpn_conv, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3_1")
    #rpn_relu1 = mx.symbol.Activation(data=rpn_conv1, act_type="relu", name="rpn_relu_1")
    rpn_pose_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=28* num_anchors, name="rpn_pose_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    poses = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, pose_pred=rpn_pose_pred, im_info=im_info, name='poses',
        op_type='proposal', feat_stride=16, scales=(4, 8, 16), rotates=(0,),
        rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
        threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    if config.USE_POSE_POOLING:
        pool5=mx.sym.POSEPooling(data=relu5_3,poses=poses,pooled_size=(7,7),spatial_scale=0.0625,name='roi_pool5')
    else:
        rois=mx.sym.Custom(poses=poses,name='rois',op_type='poses2rois')

        pool5 = mx.symbol.ROIPooling(name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    pose_pred = mx.symbol.FullyConnected(name='pose_pred', data=drop7, num_hidden=num_classes * 28)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    pose_pred = mx.symbol.Reshape(data=pose_pred, shape=(config.TEST.BATCH_IMAGES, -1, 28 * num_classes), name='pose_pred_reshape')

    # group output
    group = mx.symbol.Group([poses, cls_prob, pose_pred,rpn_cls_score])
    return group


def get_vgg_train(num_classes=2, num_anchors=NUM_ANCHORS):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_poses = mx.symbol.Variable(name="gt_poses")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_pose_target = mx.symbol.Variable(name='pose_target')
    rpn_pose_weight = mx.symbol.Variable(name='pose_weight')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")

    #rpn_conv1 = mx.symbol.Convolution(
    #    data=rpn_conv, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3_1")
    #rpn_relu1 = mx.symbol.Activation(data=rpn_conv1, act_type="relu", name="rpn_relu_1")
    rpn_pose_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=28 * num_anchors, name="rpn_pose_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_pose_loss_ = rpn_pose_weight * mx.symbol.smooth_l1(name='rpn_pose_loss_', scalar=3.0, data=(rpn_pose_pred - rpn_pose_target))
    rpn_pose_loss = mx.sym.MakeLoss(name='rpn_pose_loss', data=rpn_pose_loss_, grad_scale=1.0 / (NORM_FACTOR*config.TRAIN.RPN_BATCH_SIZE))

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    poses = mx.symbol.Custom(
        cls_prob=rpn_cls_act_reshape, pose_pred=rpn_pose_pred, im_info=im_info, name='poses',
        op_type='proposal', feat_stride=16, scales=(4, 8, 16), rotates=(0,),
        rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
        threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_poses_reshape = mx.symbol.Reshape(data=gt_poses, shape=(-1, 29), name='gt_poses_reshape')
    group = mx.symbol.Custom(poses=poses, gt_poses=gt_poses_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_poses=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    poses = group[0]
    label = group[1]
    pose_target = group[2]
    pose_weight = group[3]

    # Fast R-CNN
    if config.USE_POSE_POOLING:
        pool5=mx.sym.POSEPooling(data=relu5_3,poses=poses,pooled_size=(7,7),spatial_scale=0.0625,name='roi_pool5')
    else:
        rois=mx.symbol.Custom(poses=poses,name='rois',op_type='poses2rois')
        pool5 = mx.symbol.ROIPooling(name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch',use_ignore=True, ignore_label=-1,)
    # bounding box regression
    pose_pred = mx.symbol.FullyConnected(name='pose_pred', data=drop7, num_hidden=num_classes * 28)
    pose_loss_ = pose_weight * mx.symbol.smooth_l1(name='pose_loss_', scalar=1.0, data=(pose_pred - pose_target))
    pose_loss = mx.sym.MakeLoss(name='pose_loss', data=pose_loss_, grad_scale=1.0 / (NORM_FACTOR*config.TRAIN.BATCH_ROIS))

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    pose_loss = mx.symbol.Reshape(data=pose_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 28 * num_classes), name='pose_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_pose_loss, cls_prob, pose_loss, mx.symbol.BlockGrad(label)])
    return group



