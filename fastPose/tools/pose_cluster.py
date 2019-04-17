import json
import numpy as np
import sklearn.manifold as mnf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cPickle
import visual

index2index=[9,8,12,11,10,13,14,15,2,1,0,3,4,5] 

def visual_pose(annotation_file):
    annotation = json.loads(open(annotation_file,'r').read())['root']
    xx=[]
    size=[] 
    for i in range(len(annotation)):
        mainPersonJoints=annotation[i]['joint_self']       
        joints=np.vstack(mainPersonJoints)
        joint=joints[index2index,:-1]
        vis=joints[index2index,-1]

        
        #print joints.shape
        joint-=np.reshape(np.mean(joint,axis=0),(1,2))
        s=np.mean(np.sqrt(np.sum( joint**2,axis=1)))   
        joint/=(s+1e-14)
        xx.append(joint.ravel())
        size.append(s)
    X=np.vstack(xx)
    print 'total visuanl pose:',len(size)
    #tsne = mnf.TSNE(n_components=2, init='random')
    #np.save('xx.npy',X)
    
    #Y = tsne.fit_transform(X)
    #plt.figure(1)
    #plt.scatter(Y[:, 0], Y[:, 1])
    
    #plt.figure(2)
    #plt.hist(size,100)

    #kmeans3=KMeans(n_clusters=3,n_init=3)
    #kmeans3.fit(X)
    
    #labels=kmeans3.labels_
    #np.save('kmeans3.npy',labels)
    #plt.figure(3)
    #plt.scatter(Y[:, 0], Y[:, 1],
    #           c=labels.astype(np.float), edgecolor='k')
    
    #kmeans3=KMeans(n_clusters=6,n_init=6)
    #kmeans3.fit(X)
    #labels=kmeans3.labels_
    #np.save('kmeans6.npy',labels)
    #plt.figure(4)
    #plt.scatter(Y[:, 0], Y[:, 1],
    #           c=labels.astype(np.float), edgecolor='k')

    kmeans3=KMeans(n_clusters=9,n_init=9)
    kmeans3.fit(X)
    labels=kmeans3.labels_
    np.save('new_kmeans9.npy',labels)
    #plt.figure(5)
    #plt.scatter(Y[:, 0], Y[:, 1],
    #           c=labels.astype(np.float), edgecolor='k')

    #plt.show()
    

def anchor_show(annotation_file):
    annotation = json.loads(open(annotation_file,'r').read())['root']
    xx=[]
    size=[] 
    for i in range(len(annotation)):
        mainPersonJoints=annotation[i]['joint_self']
        joints=np.vstack(mainPersonJoints)
        joint=joints[index2index,:-1]
        vis=joints[index2index,-1]

        
        #print joints.shape
        joint-=np.reshape(np.mean(joint,axis=0),(1,2))
        s=np.mean(np.sqrt(np.sum( joint**2,axis=1)))   
        joint/=(s+1e-14)
        xx.append(joint.ravel())
        size.append(s)
    X=np.vstack(xx)
    plt.hist(size,100)
    plt.show()
    index =np.load('new_kmeans9.npy')
    num = np.max(index)

    anchors=[]
    for n in range(num+1):
        anchor=X[index==n,:]
        anchor=np.mean(anchor,axis=0).reshape(-1,2)
        s=np.mean(np.sqrt(np.sum( joint**2,axis=1))) 
        anchor*=8
        #anchor=anchor[index2index,:]
        
        anchors.append(anchor)
    with open('new_pose_anchors_prior.pkl','w') as f:
        cPickle.dump(anchors,f)
    visual.visual_anchor_pose(anchors)



def refine_json(annotation_file):
    annotation = json.loads(open(annotation_file,'r').read())['root']
    with open(r'D:\documents\mx-rcnn\mx-rcnn\fastPose\tools\full_visual_pose_anchors_prior.pkl','r') as f:
        anchors=cPickle.load(f)
    visual.visual_anchor_pose(anchors)

    for index in range(len(annotation)):
        mainPersonJoints=annotation[index]['joint_self']
        w=annotation[index]['img_width']
        h=annotation[index]['img_height']       
        joints=np.vstack(mainPersonJoints)
        joint=joints[index2index,:-1]
        zeros=np.all(joint<1.0,axis=1)
        non_zeros =True-zeros
        #print 'mainPersonJoints 'non_zeros                    
        #print joints.shape
        shift=np.mean(joint[non_zeros,:],axis=0)
        joint-=np.reshape(shift,(1,2))
        
        s=np.mean(np.sqrt(np.sum( joint[non_zeros,:]**2,axis=1))) 
        joint/=(s+1e-14)
        
        a=1e10
        flag=0
        for i in range(len(anchors)):
            temp=np.sqrt(((joint[non_zeros,:]-anchors[i][non_zeros,:]/8)**2).mean())
            if temp<a:
                a=temp
                flag=i
        for ik,item in enumerate(index2index):
            if annotation[index]['joint_self'][item][0] <1.0 and annotation[index]['joint_self'][item][1] <1.0:
                 annotation[index]['joint_self'][item][0]=max(min(anchors[flag][ik][0]*s/8.0+shift[0],w),0)
                 annotation[index]['joint_self'][item][1]=max(min(anchors[flag][ik][1]*s/8.0+shift[1],h),0)
        numOtherPerson=annotation[index]['numOtherPeople']
        if numOtherPerson>0:
            if numOtherPerson==1:
                if len(annotation[index]['joint_others'])<16:
                    for i in range(16-len(annotation[index]['joint_others'])):
                        annotation[index]['joint_others'].append([0.0,0.0,0.0])
                joints=np.vstack(annotation[index]['joint_others'])
                joint=joints[index2index,:-1]
                #vis=joints[index2index,-1]
                zeros=np.all(joint<1.0,axis=1)
                non_zeros =True-zeros
                            
                #print joints.shape
                shift=np.mean(joint[non_zeros,:],axis=0)
                joint-=np.reshape(shift,(1,2))
                
                s=np.mean(np.sqrt(np.sum( joint[non_zeros,:]**2,axis=1))) 
                joint/=(s+1e-14)
                a=1e10
                flag=0
                for i in range(len(anchors)):
                    temp=np.sqrt(((joint[non_zeros,:]-anchors[i][non_zeros,:]/8)**2).mean())
                    if temp<a:
                        a=temp
                        flag=i
                for ik,item in enumerate(index2index):
                    if annotation[index]['joint_others'][item][0] <1.0 and annotation[index]['joint_others'][item][1] <1.0:
                         annotation[index]['joint_others'][item][0]=max(min(anchors[flag][ik][0]*s/8.0+shift[0],w),0)
                         annotation[index]['joint_others'][item][1]=max(min(anchors[flag][ik][1]*s/8.0+shift[1],h),0)

            if numOtherPerson>1:
                for k,iother in enumerate(annotation[index]['joint_others']):
                    if len(iother)<16:
                        for i in range(16-len(iother)):
                            annotation[index]['joint_others'][k].append([0.0,0.0,0.0])  
                    joints=np.vstack(iother)
                    joint=joints[index2index,:-1]
                    #vis=joints[index2index,-1]
                    zeros=np.all(joint<1.0,axis=1)
                    
                    non_zeros =True-zeros
                    #if not np.all(non_zeros):
                    #    print joint
                    #print joints.shape
                    shift=np.mean(joint[non_zeros,:],axis=0)
                    joint-=np.reshape(shift,(1,2))
                    
                    s=np.mean(np.sqrt(np.sum( joint[non_zeros,:]**2,axis=1))) 
                    joint/=(s+1e-14)
                    a=1e10
                    flag=0
                    for i in range(len(anchors)):
                        temp=np.sqrt(((joint[non_zeros,:]-anchors[i][non_zeros,:]/8)**2).mean())
                        if temp<a:
                            a=temp
                            flag=i
                    for ik,item in enumerate(index2index):

                        if annotation[index]['joint_others'][k][item][0] <1.0 and annotation[index]['joint_others'][k][item][1] <1.0:
                             annotation[index]['joint_others'][k][item][0]=max(min(anchors[flag][ik][0]*s/8+shift[0],w),0)
                             annotation[index]['joint_others'][k][item][1]=max(min(anchors[flag][ik][1]*s/8+shift[1],h),0)
                    if annotation[index]['joint_others'][k][15][0]<1.0 and annotation[index]['joint_others'][k][15][1]<1.0:
                        print annotation[index]['joint_others'][k]
                        print annotation[index]['img_paths']


    dicts={}
    dicts['root']=annotation  
    with open('new_MPI_annotation.json','w') as f:
        json.dump(dicts,f)

 
def check_dataset(dataset):
    with open(r'D:\documents\mx-rcnn\mx-rcnn\fastPose\tools\Final_pose_anchors_prior.pkl','r') as f:
        anchors=cPickle.load(f)
    visual.visual_anchor_pose(anchors)
    anchors4=[anchor*4 for anchor in anchors]
    anchors8=[anchor*8 for anchor in anchors]
    anchors16=[anchor*16 for anchor in anchors]
    anchors32=[anchor*32 for anchor in anchors]
    ach=anchors4+anchors8+anchors16+anchors32
    annotation = json.loads(open(dataset,'r').read())['root']
    belowing=[]
    dist=[]
    for index in range(len(annotation)):
        numOtherPerson=annotation[index]['numOtherPeople']
        h=annotation[index]['img_height']
        w=annotation[index]['img_width']
        s=w
        if h>w:
            s=h
        scale=1000.0/s   
        otherPersonJoints=[]
        if numOtherPerson >0:
            if numOtherPerson>1:
                otherPersonJoints=otherPersonJoints+annotation[index]['joint_others']
            else:
                otherPersonJoints.append(annotation[index]['joint_others'])
        mainPersonJoints=annotation[index]['joint_self']
        allPerson=[mainPersonJoints]#otherPersonJoints+[mainPersonJoints]
        num_objs = len(allPerson)  
        for pose in allPerson:
            joints=np.vstack(pose)
            joint=joints[index2index,:-1]*scale
            non_zeros=np.all(joint<1.0,axis=1)
            #print index,non_zeros
            if np.max(non_zeros):
                print annotation[index]['img_paths'],non_zeros
                print pose
                break
            fini=1e10
            flag=-1
            v=0
            shift=np.mean(joint,axis=0)
            joint-=np.reshape(shift,(1,2))
            s=np.mean(np.sqrt(np.sum(joint**2,axis=1)))+1e-14
            for ak,ac in enumerate(ach):
                distance= np.sqrt(np.sum((joint-ac)**2,axis=1)).mean()/s
                if distance<fini:
                    flag=ak
                    fini=distance
            if flag<0:
                print allPerson
                print s,numOtherPerson
                print annotation[index]['img_paths']

            belowing.append(flag)
            dist.append(fini)
    dist=np.array(dist)
    plt.figure()
    plt.hist(dist[dist<1000],100)
    plt.figure()
    plt.hist(belowing,48)
    plt.show()
                          
           

if __name__ =='__main__':
    dataset=r'D:\documents\mx-rcnn\mx-rcnn\fastPose\tools\Final_MPI_annotation.json'
    dataset1=r'D:\dataset\evalMPII\mpii_human_pose_v1\MPI_annotations.json'
    #visual_pose(dataset)
    

    #anchor_show(dataset)
    #refine_json(dataset1)
    check_dataset(dataset)