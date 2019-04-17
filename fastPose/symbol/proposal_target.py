import mxnet as mx
import numpy as np

from fastPose.core.minibatch import sample_pose

DEBUG = False

class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_poses, fg_fraction):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_poses = batch_poses
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_poses %self._batch_images==0,'BATCHIMAGES {} must devide BATCH_POSES {}'.format(self._batch_images, self._batch_poses)

        poses_per_image= self._batch_poses/self._batch_images
        fg_poses_per_image = np.round(self._fg_fraction*poses_per_image).astype(int)

        all_poses = in_data[0].asnumpy()
        gt_poses =  in_data[1].asnumpy()

        # add the ground truth pose into condidate poses
        zeros =np.zeros((gt_poses.shape[0],1),dtype=gt_poses.dtype)
        all_poses = np.vstack((all_poses,np.hstack((zeros,gt_poses[:,:-1]))))

        # single batch only
        assert np.all(all_poses[:,0]==0),'only single batch support'

        poses,labels,pose_targets,pose_weights = \
            sample_pose(all_poses,fg_poses_per_image,poses_per_image,self._num_classes,gt_poses=gt_poses)

        if DEBUG:
            print 'labels=',labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        for ind,val in enumerate([poses,labels,pose_targets,pose_weights]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self,num_classes,batch_images,batch_poses,fg_fraction):
        super(ProposalTargetProp, self).__init__(need_top_grad = False)
        self._num_classes=int(num_classes)
        self._batch_images=int(batch_images)
        self._batch_poses=int(batch_poses)
        self._fg_fraction=float(fg_fraction)

    def list_arguments(self):
        return['poses','gt_poses']

    def list_outputs(self):
        return ['poses_output','label','pose_target','pose_weight']

    def infer_shape(self, in_shape):
        rpn_poses_shape=in_shape[0]
        gt_poses_shape=in_shape[1]

        output_poses_shape=(self._batch_poses,29)##28+1
        label_shape=(self._batch_poses,)
        pose_target_shape=(self._batch_poses,self._num_classes*28)
        pose_weight_shape=(self._batch_poses,self._num_classes*28)

        return[rpn_poses_shape,gt_poses_shape],[output_poses_shape,label_shape,pose_target_shape,pose_weight_shape]
    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_poses, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

