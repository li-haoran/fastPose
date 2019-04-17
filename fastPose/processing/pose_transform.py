'''
this file for transform the box and the re-r=transform the box
'''
import numpy as np
EPS=1e-14

DEBUG=False

def pose_transform(ex_poses,gt_poses):
    '''
    compute target pose from ex_poses to gt_poses
    :param ex_rois:[N,28]
    :param gt_rois: [N,28]
    return [N,28]
    '''
    ## verson 1 without magnitude

    #target=np.zeros_like(ex_poses,dtype=np.float32)
    #target[:]=gt_poses-ex_poses

    ## verson 2 normalize with with magnitude
    posex = ex_poses[:,0::2]
    posey = ex_poses[:,1::2]
    centerx=np.mean(posex,axis=1).reshape((-1,1))
    centery=np.mean(posey,axis=1).reshape((-1,1))
    scales=(np.mean(np.sqrt((posex-centerx)**2+(posey-centery)**2),axis=1)+EPS).reshape(-1,1)


    target=np.zeros_like(ex_poses,dtype=np.float32)
    target[:]=(gt_poses-ex_poses)/scales


    return target

def pose_pred(poses,poses_deltas):
    '''
    transform the posed_deltas into real pose locations
    here poses [N,28]
    poses-deltas [N,28]
    return  [N,28]
    '''
    ## version 2 without magnitude
    if poses.shape[0]==0:
        return np.zeros(0,poses_deltas.shape[1])
    poses=poses.astype(np.float,copy=False)

    #pred_pose=np.zeros_like(poses,dtype=np.float32)
    #pred_pose[:]=poses+poses_deltas

    ## version 2 without magnitude
    
    posex = poses[:,0::2]
    posey = poses[:,1::2]
    centerx=np.mean(posex,axis=1).reshape((-1,1))
    centery=np.mean(posey,axis=1).reshape((-1,1))
    scales=(np.mean(np.sqrt((posex-centerx)**2+(posey-centery)**2),axis=1)+EPS).reshape(-1,1)
   

    pred_pose=np.zeros_like(poses,dtype=np.float32)
    pred_pose[:]=poses+poses_deltas*scales

    #if DEBUG:
    #    from fastPose.tools.visual import visual_pose
    #    visual_pose(pred_pose[100:110,],(600,1000))

    return pred_pose

def clip_poses(poses,im_shape):
    '''
    clip poses to image boundaries
    poses [N, 28]
    im_shape tuple 2 [h,w]
    return [n,28]
    '''

    poses[:,0::2]=np.maximum(np.minimum(poses[:,0::2],im_shape[1]-1),0)
    poses[:,1::2]=np.maximum(np.minimum(poses[:,1::2],im_shape[0]-1),0)

    #if DEBUG:
    #    from fastPose.tools.visual import visual_pose
    #    print poses[0,:]
    #    visual_pose(poses[:10,],(600,1000))

    return poses

