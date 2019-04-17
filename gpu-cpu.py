from fastPose.utils.load_model import load_param
from fastPose.utils.save_model import save_checkpoint
import mxnet as mx
import cPickle

def converter(prefix,epoch,ctx):
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)
    for k,v in arg_params.iteritems():
        arg_params[k]=arg_params[k].asnumpy()
    with open('params.pkl','w+') as f:
        cPickle.dump(arg_params,f)

def recon(file,prefix,epoch):
    with open(file,'r') as f:
        arg_params=cPickle.load(f)
    for k,v in arg_params.iteritems():
        arg_params[k]=mx.nd.array(arg_params[k])
    save_checkpoint(prefix,0,arg_params,{})

if __name__ =='__main__':
    recon(r'D:\documents\mx-rcnn\mx-rcnn\open_pose\model\openpose.pkl',r'D:\documents\mx-rcnn\mx-rcnn\open_pose\model\OpenPose_vgg19',1)
    #converter('model/e2e',10,mx.cpu())