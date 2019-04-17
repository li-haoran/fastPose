'''
this file for pose regression, include how to measure the similarity and, how to decide the target pose and label
'''

import numpy as np
from pose_transform import pose_transform
from fastPose.config import config

EPS=1e-14

def pose_overlaps(poses,query_poses):
    '''
    determine the distance between the poses and query poses the smaller the better
    not the overlaps sorry!
    here 
    poses  [N,28]
    query_poses [k,28]
    return  [N,k] overlaps
    '''

    n_ =poses.shape[0]
    k_ =query_poses.shape[0]
    overlaps = np.zeros((n_,k_),dtype=np.float)
    ################change the scale factor from the pose to query pose
    x=query_poses[:,0::2] 
    y=query_poses[:,1::2]
    centerx=np.mean(x,axis=1).reshape((-1,1))
    centery=np.mean(y,axis=1).reshape((-1,1))
    scales=np.mean(np.sqrt((x-centerx)**2+(y-centery)**2),axis=1)+EPS

    for k in range(k_):
        query_pose=query_poses[k]
        #euclidean distance version 1
        distance=np.mean(np.sqrt((poses[:,0::2]-query_pose[0::2])**2+(poses[:,1::2]-query_pose[1::2])**2),axis=1)
        overlaps[:,k] = (distance/scales[k])

    return overlaps

def compute_pose_regression_targets(poses,labels):
    '''
    poses: posedb[i]['pose']  k*28
    '''
    pass

def expand_pose_regression_targets(poses_targets_data,num_classes):
    '''
    expand from 29 to 28*num_classes, only right class have none-zeros targets
    poses_targets_data [N,29]
    num_classes :
    return target [N*28 num_classes],weights !only forground is none_zero
    '''
    classes=poses_targets_data[:,0]
    pose_targets = np.zeros((classes.size,28*num_classes),dtype=np.float32)
    pose_weights = np.zeros(pose_targets.shape,dtype=np.float32)
    indexes = np.where(classes>0)[0]
    for index in indexes:
        cls = classes[index]
        start = int(28*cls)
        end=start+28
        pose_targets[index,start:end] = poses_targets_data[index,1:]
        pose_weights[index,start:end] = config.TRAIN.POSEWEIGHT

    return pose_targets,pose_weights


