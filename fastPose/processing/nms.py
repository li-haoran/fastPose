import numpy as np

EPS=1e-14
DEBUG=False

def nms(dets,thresh):
    '''
    greedy select high score pose without too much overlaps
    dets:[N,29] where 29 is 28 xy and scores
    thresh is control the distance
    '''
    posex = dets[:,0:-1:2]
    posey = dets[:,1:-1:2]
    scores = dets[:,-1]

    centerx=np.mean(posex,axis=1).reshape((-1,1))
    centery=np.mean(posey,axis=1).reshape((-1,1))
    scales=np.mean(np.sqrt((posex-centerx)**2+(posey-centery)**2),axis=1)+EPS

    order = scores.argsort()[::-1]

    keep=[]
    while order.size>0:
        i=order[0]
        keep.append(i)
        distance=np.mean(np.sqrt((posex[order[1:],:]-posex[i,:])**2+(posey[order[1:],:]-posey[i,:])**2),axis=1)
        distance /=scales[i]
        if DEBUG:
            print 'distant:{}'.format(distance)
        inds = np.where(distance >thresh)[0]
        order = order[inds+1]

    return keep
