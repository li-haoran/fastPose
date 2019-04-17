
import cv2
import numpy as np
import numpy.random as npr
import random
import os
import cPickle

from fastPose.config import config
from fastPose.processing import image_processing
from fastPose.processing.pose_regression import pose_overlaps
from fastPose.processing.pose_regression import expand_pose_regression_targets

from fastPose.processing.pose_transform import pose_transform
from fastPose.processing.generate_anchor import generate_anchors

DEBUG=True

def get_image(roidb):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = image_processing.resize(im, target_size, max_size, stride=config.IMAGE_STRIDE)
        im_tensor = image_processing.transform(im, config.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['poses'] = roi_rec['poses'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb



def _compute_targets(ex_poses,gt_poses):
    '''
    compute the pose targets for image
    ex_poses: []
    '''
    assert ex_poses.shape[0] == gt_poses.shape[0],'inconsistent poses number '
    assert ex_poses.shape[1] == 28,'wrong pose shape'
    assert gt_poses.shape[1] == 29,'wrong gt shape'
    return pose_transform(ex_poses,gt_poses[:,:-1].astype(np.float32,copy=False))


def sample_pose(poses,fg_poses_per_image,poses_per_image,num_classes,labels=None, overlaps=None, pose_targets=None, gt_poses=None):
    '''
    generate random sample of rois at the fg /bg 

    all_poses [N,29] with batch index
    fg_poses_per_image: foreground roi number
    poses_per_image: total roi number
    num_classes: number of classes
    labels: maybe precomputed
    overlaps: maybe precomputed (max_overlaps)
    pose_targets: maybe precomputed
    gt_poses: optional for e2e [n, 5] ((x,y)^14, cls)
    return: (labels, poses, pose_targets, pose_weights)
    '''
    if labels is None:
        overlaps = pose_overlaps(poses[:,1:].astype(np.float),gt_poses[:,:-1].astype(np.float))
        gt_assignment = overlaps.argmin(axis=1)
        overlaps = overlaps.min(axis=1)
        labels = gt_poses[gt_assignment,-1]
    if DEBUG:
        print 'final proposal gt_pose overlaps:{}'.format(np.sort(overlaps))
    # fg index
    fg_indexes =  np.where(overlaps <= config.TRAIN.FG_THRESH)[0]

    # gurantee the poses per images  is fg_poses_per_image
    fg_poses_per_this_image =  np.minimum(fg_indexes.size,fg_poses_per_image)
    if len(fg_indexes) > fg_poses_per_this_image:
        fg_indexes=npr.choice(fg_indexes,size=fg_poses_per_this_image,replace=False)

    # bg index
    bg_indexes = np.where((overlaps > config.TRAIN.BG_THRESH_LOW) &(overlaps < config.TRAIN.BG_THRESH_HI))[0]

    bg_poses_per_this_image = poses_per_image -  fg_poses_per_this_image
    bg_poses_per_this_image = np.minimum(bg_poses_per_this_image,bg_indexes.size)
    if len(bg_indexes)>bg_poses_per_this_image:
        bg_indexes = npr.choice(bg_indexes,size=bg_poses_per_this_image,replace=False)

    #all index append
    keep_indexes = np.append(fg_indexes,bg_indexes)

    #pad to the fixed batch size
    while keep_indexes.shape[0] < poses_per_image:
        gap = np.minimum(len(poses),poses_per_image-keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(poses)),size=gap,replace=False)
        keep_indexes = np.append(keep_indexes,gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set the bg label=0  
    
    # hint if the gap exist, the labels will ambiguous
    labels[fg_poses_per_this_image:]=0
    if fg_poses_per_this_image*3<config.TRAIN.BATCH_ROIS-fg_poses_per_image:
        labels[fg_poses_per_this_image*4:]=-1 #____________smaple inefficent for 1:3 manully
    poses = poses[keep_indexes]

    # load and compute the pose_targets
    if pose_targets is not None:
        pose_target_data = pose_targets[keep_indexes,:]
    else:
        targets = _compute_targets(poses[:,1:],gt_poses[gt_assignment[keep_indexes],:])
        if config.TRAIN.POSE_NORMALIZATION_PRECOMPUTED:

            ##not implement
            print 'hello'
        pose_target_data=np.hstack((labels[:,np.newaxis],targets))
    pose_targets,pose_weights=\
        expand_pose_regression_targets(pose_target_data,num_classes)

    return poses,labels,pose_targets,pose_weights

def assign_anchor(feat_shape, gt_poses, im_info, feat_stride=16,
                  scales=(8, 16, 32), rotates=(0,), allowed_border=0):
    '''
     assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_poses: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :param feat_stride: anchor position step
    :param scales: used to generate anchors, affects num_anchors (per location)
    :param rotates: rotates of generated anchors
    :param allowed_border: filter out anchors with edge overlap > allowed_border
    :return: dict of label
    'label': of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    'pose_target': of shape (batch_size, num_anchors * 28, feat_height, feat_width)
    'pose_inside_weight': *todo* mark the assigned anchors
    'pose_outside_weight': used to normalize the pose_loss, all weights sums to RPN_POSITIVE_WEIGHT
    '''
    def _unmap(data,count,inds,fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret
    im_info=im_info[0]
    scales=np.array(scales,dtype=np.float32)
    base_anchors = generate_anchors(rotates=rotates, scales=scales)
    num_anchors = base_anchors.shape[0]
    feat_height,feat_width=feat_shape[-2:]

    if DEBUG:
        print 'anchors:'
        #print base_anchors

        print 'im_info=',im_info
        print 'feat_height, feat_width=',(feat_height,feat_width)
        print 'gt_poses=',gt_poses
    #1. generate proposals from boxes delta and shifted
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    all_shift=[]
    for i in range(14):
        all_shift+=[shift_x.ravel(), shift_y.ravel()]
    shifts = np.vstack(all_shift).transpose()  ##[HW,28]
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 28) to
    # cell K shifts (K, 1, 28) to get
    # shift anchors (K, A, 28)
    # reshape to (K*A, 28) shifted anchors
    A = num_anchors
    K = shifts.shape[0]
    all_anchors = base_anchors.reshape((1, A, 28)) + shifts.reshape((1, K, 28)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 28))

    total_anchors= int(K*A)

    # keep inside pose
    inds_inside=np.where(np.all(all_anchors>=-allowed_border,axis=1) &
                            np.all(all_anchors[:,0::2]<im_info[1]+allowed_border,axis=1) &
                            np.all(all_anchors[:,1::2]<im_info[0]+allowed_border,axis=1))[0]

    if DEBUG:
        print 'total anchors:',total_anchors
        print 'inside image:',len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors shape', anchors.shape
    # labels:1 positive 0 negtive -1 dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    if gt_poses.size >0:
        #calculat the overlap between the proposal and gt poses
        overlaps=pose_overlaps(anchors.astype(np.float),gt_poses[:,:-1].astype(np.float))
        argmin_overlaps= overlaps.argmin(axis=1)
        min_overlaps=overlaps[np.arange(len(inds_inside)),argmin_overlaps]
        gt_argmin_overlaps =overlaps.argmin(axis=0)
        gt_min_overlaps = overlaps[gt_argmin_overlaps,np.arange(overlaps.shape[1])]
        gt_argmin_overlaps = np.where(overlaps == gt_min_overlaps)[0]

        #if DEBUG:
        #     mo=np.sort(min_overlaps)[:128]
        #     print mo
        #if DEBUG:
        #    for ww in range(overlaps.shape[1]):
        #        imap=overlaps[:,ww]
        #        imap=_unmap(imap,total_anchors,inds_inside,fill=-1)
        #        imap=imap.reshape((feat_height, feat_width, A))
        #        import matplotlib.pyplot as plt
        #        #fig,ax=plt.subplots(6,5)
        #        #axes=ax.ravel()
        #        #for i in range(27):
        #        #    axes[i].imshow(imap[:,:,i])
        #        imap[imap<0]=1
        #        imap[imap>1]=1
        #        imk=np.min(imap,axis=2)
        #        plt.imshow(imk)
        #    plt.show()
            
        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            #assign bg first
            labels[min_overlaps>config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        #assign fg
        labels[gt_argmin_overlaps]=1
        print min_overlaps.shape
        over_index=np.argwhere(min_overlaps<=config.TRAIN.RPN_POSITIVE_OVERLAP)
        sort_over_index=min_overlaps[over_index].ravel().argsort()
        real_index=over_index[sort_over_index]
        #disable_inds=index[num_fg:len(fg_inds)]
        labels[over_index]=1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[min_overlaps > config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 
    else:
        labels[:]=0

    #sample positives if positive too many

    num_fg =  int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds=real_index[num_fg:len(fg_inds)]
        #disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        #if DEBUG:
        #    disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
        labels[disable_inds] = -1

    #sample negtive if neg too many
    num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
    if num_bg>len(fg_inds):
        num_bg=len(fg_inds)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        #if DEBUG:
        #    disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
        labels[disable_inds] = -1

    if DEBUG:
        from fastPose.tools.visual import visual_pose
        choose_anchor=anchors[labels==1,:]
        visual_pose(choose_anchor,(feat_height*16,feat_width*16))

    pose_targets=np.zeros((len(inds_inside),28),dtype=np.float32)
    if gt_poses.size>0:
        pose_targets[:] = _compute_targets(anchors,gt_poses[argmin_overlaps,:])
    if DEBUG:
        print 'targets:',pose_targets[labels == 1,:]

    pose_weights = np.zeros((len(inds_inside),28),dtype =np.float32)
    pose_weights[labels==1,:] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)


    if DEBUG:
        _sums = pose_targets[labels == 1, :].sum(axis=0)
        _squared_sums = (pose_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts = config.EPS + np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means', means
        print 'stdevs', stds 

    labels = _unmap(labels,total_anchors,inds_inside,fill=-1)
    pose_targets=_unmap(pose_targets,total_anchors,inds_inside,fill=0)
    pose_weights = _unmap(pose_weights,total_anchors,inds_inside,fill=0)


    if DEBUG:
        #print 'rpn: min min_overlaps', np.min(min_overlaps)
        print 'rpn: num_positives', np.sum(labels == 1)
        print 'rpn: num_negatives', np.sum(labels == 0)
        _fg_sum = np.sum(labels == 1)
        _bg_sum = np.sum(labels == 0)
        _count = 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, A * feat_height * feat_width))
    pose_targets = pose_targets.reshape((1, feat_height, feat_width, A * 28)).transpose(0, 3, 1, 2)
    pose_weights = pose_weights.reshape((1, feat_height, feat_width, A * 28)).transpose((0, 3, 1, 2))

    label = {'label': labels,
             'pose_target': pose_targets,
             'pose_weight': pose_weights}
    return label


def get_rpn_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_poses
    :param roidb: ['image', 'flipped'] + ['gt_poses', 'poses', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    # gt poses: (xy^14, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_poses = np.empty((roidb[0]['poses'].shape[0], 29), dtype=np.float32)
        gt_poses[:, 0:-1] = roidb[0]['poses'][gt_inds, :]
        gt_poses[:, -1] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_poses': gt_poses}

    return data, label


def get_rpn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {}

    return data, label, im_info