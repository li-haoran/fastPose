import mxnet as mx
import numpy as np

##version 1

DEBUG = False

BORDER_VALUE = 20
MAX=1e10
MIN=-1e3
class Poses2RoisOperator(mx.operator.CustomOp):
    def __init__(self):
        super(Poses2RoisOperator, self).__init__()
        
    def forward(self, is_train, req, in_data, out_data, aux):
        poses=in_data[0].asnumpy()
        batch_indexes=poses[:,0]

        # the naive way is eleminate  points the around the (0,0) the naive version
        ind_indexs=(poses[:,1::2] > BORDER_VALUE) &(poses[:,2::2] > BORDER_VALUE)
        inverse_ind_indexs= 1-ind_indexs
        left_x=np.min(poses[:,1::2]+ inverse_ind_indexs*MAX,axis=1)
        left_y=np.min(poses[:,2::2]+ inverse_ind_indexs*MAX,axis=1)
        right_x=np.max(poses[:,1::2]+ inverse_ind_indexs*MIN,axis=1)
        right_y=np.max(poses[:,2::2]+ inverse_ind_indexs*MIN,axis=1)
        rois=np.stack([batch_indexes,left_x,left_y,right_x,right_y],axis=1)
        if DEBUG:
            print 'rois=',rois
            print 'num_rois=',rois.shape[0]

        self.assign(out_data[0],req[0],rois)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        

@mx.operator.register('poses2rois')
class Poses2RoisProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(Poses2RoisProp, self).__init__(need_top_grad = False)
        

    def list_arguments(self):
        return['poses']

    def list_outputs(self):
        return ['rois_output']

    def infer_shape(self, in_shape):
        poses_shape=in_shape[0]
        rois_shape=(in_shape[0][0],5)

        return[poses_shape],[rois_shape]
    def create_operator(self, ctx, shapes, dtypes):
        return Poses2RoisOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

