import numpy as np

def unique_pose(poses,scale=1.0):
    '''
    find the all the different poses,(unique pose)
    '''
    return 

def filter_small_pose(poses,min_size):
    '''
    poses is [n,28], return the pose size 
    (average encludean distance to the center of body > min_size)
    hint: the extreme shape, we forget 
    '''
    x=poses[:,::2]
    y=poses[:,1::2]
    newx=x-np.reshape(np.mean(x,axis=1),(x.shape[0],1))
    newy=y-np.reshape(np.mean(y,axis=1),(y.shape[0],1))
    size=np.mean(np.sqrt(newx**2+newy**2),axis=1)
    keep = np.where(size>min_size)[0]
    return keep