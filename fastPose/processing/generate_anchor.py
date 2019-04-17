"""
Generate base anchors on index 0
"""

import numpy as np
import cPickle

POSE_ANCHORS_FILE=r'D:\documents\mx-rcnn\mx-rcnn\fastPose/tools/Final_pose_anchors_prior.pkl'
POSE_ANCHORS_PRIOR=cPickle.load(open(POSE_ANCHORS_FILE,'r'))    ## anchor_num=27  
def generate_anchors(base_anchor=POSE_ANCHORS_PRIOR, rotates=np.array([-1, 0, 1]),
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) structure ,here base_cluster is prior of the structure are list
    each is structure here we use [14,2(xy)], and rotate is which angles to rotate the structure
    scale is which scale to scaling the structure

    notice: the structure is centerized
    """

    rotate_anchors = rotate_anchor(base_anchor, rotates)
    anchors = scale_anchor(rotate_anchors,scales)
    N=anchors.shape[0]
    anchors= np.reshape(anchors,(N,-1)) ##[N,28]
    return anchors



def _caculate_affine(nanchor,angle):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    affine_matirx=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]).astype(np.float32)

    new_anchor = np.stack([ np.dot(nanchor[i], affine_matirx) for i in range(len(nanchor))],axis=0)
    return new_anchor


def rotate_anchor(anchor, angles):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    
    anchors=np.vstack([_caculate_affine(anchor,ag) for ag in angles])
    return anchors


def scale_anchor(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    anchors=np.vstack([anchor*sc for sc in scales])
    return anchors
