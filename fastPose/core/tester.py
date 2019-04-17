import cPickle
import os
import time
import mxnet as mx
import numpy as np

from fastPose.config import config
from fastPose.processing import image_processing
from fastPose.processing.pose_transform import pose_pred,clip_poses
from fastPose.processing.nms import nms
from fastPose.core.module import MutableModule

class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 test_data=None, arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(test_data.provide_data, test_data.provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))

def im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)

    # drop the batch index
    poses=output['poses_output'].asnumpy()[:, 1:]
    poses_delta = output['pose_pred_reshape_output'].asnumpy()[0][:,28:]
    scores = np.squeeze(output['cls_prob_reshape_output'].asnumpy())[:,1:2]
    #scores_map=output['rpn_cls_score_output'].asnumpy()[0][27:,:,:]
    #heat_map=np.max(scores_map,axis=0)
    #import matplotlib.pyplot as plt
    #plt.imshow(heat_map)
    #plt.show()
    # transform to original scale
    posesr=pose_pred(poses,poses_delta)
    #poses = poses

    return scores, poses, data_dict,posesr


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    #assert not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    t = time.time()
    imdb_poses = list()
    original_poses = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, poses, data_dict,posesr = im_proposal(predictor, data_batch, data_names, scale)
        t2 = time.time() - t
        t = time.time()

        # assemble proposals
        dets = np.hstack((poses, scores))
        #print dets[:5,:]
        #original_boxes.append(dets)
        dets2=np.hstack((posesr, scores))
        #print dets2[:5,:]
        # filter proposals
        keep = np.argsort(dets[:, -1])[::-1]
        #imdb_poses.append(dets[keep, :])
        print dets[keep, -1]
        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), dets[keep[:50],:], ['obj'], scale)

        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), dets2[keep[:50],:], ['obj'], scale)

        print 'generating %d/%d' % (i + 1, imdb.num_images), 'proposal %d' % (poses.shape[0]), \
            'data %.4fs net %.4fs' % (t1, t2)
        i += 1

    assert len(imdb_poses) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_poses, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_poses




def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    configs=('head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip', 'Rkne', 'Rank','Lhip', 'Lkne', 'Lank')
    links=((0, 1), (1,2),(1,5),(2 ,3), (3, 4), (5, 6), (6, 7), (1,8),(8, 9), (9, 10),(1,11),(11 ,12), (12, 13))
    eps=1e-14
    import matplotlib.pyplot as plt
    import random
    from matplotlib.patches import Ellipse
    from matplotlib.patches import Circle
    im = image_processing.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        colors=[]
        for link in links:
            colors.append((random.random(),random.random(),random.random()))

        dets = detections
        for i in range(10):
            new_pose = dets[i,:-1]
            new_pose =new_pose.reshape((14,2))
            score= dets[i,-1]
            for i,link in enumerate(links):
                color=(0,0,0)
                start = new_pose[link[0],:]
                circle=Circle(start,2,fill=True,facecolor=color)
                plt.gca().add_artist(circle)
                end = new_pose[link[1],:]
                circle=Circle(end,2,fill=True,facecolor=color)
                plt.gca().add_artist(circle)

                center= (start+end)/2
                width = np.sqrt(np.sum((end-start)**2))
                height = 0.15 *width

                tan= (end[1]-start[1])/(end[0]-start[0]+eps)
                angle=np.arctan(tan)/np.pi *180

                stem=Ellipse(center,width=width,height=height,angle=angle,color=colors[i],alpha=0.5)
                plt.gca().add_artist(stem)

    plt.show()
