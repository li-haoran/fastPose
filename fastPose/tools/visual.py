import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
import random
import numpy as np
config=('head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip', 'Rkne', 'Rank','Lhip', 'Lkne', 'Lank')
links=((0, 1), (1,2),(1,5),(2 ,3), (3, 4), (5, 6), (6, 7), (1,8),(8, 9), (9, 10),(1,11),(11 ,12), (12, 13))

eps= 1e-14
def visual_anchor_pose(anchor_pose_list):
    '''
    visual anchors
    '''
    colors=[]
    for link in links:
        colors.append((random.random(),random.random(),random.random()))
    num=len(anchor_pose_list)
    fig, axs = plt.subplots(3,4,subplot_kw={'aspect': 'equal'})
    axs=axs.ravel()
    for ai,pose in enumerate(anchor_pose_list):
        ax=axs[ai]
        new_pose = pose*8
        xmin=np.min(new_pose[:,0])-20
        xmax=np.max(new_pose[:,0])+20
        ymin=np.min(new_pose[:,1])-20
        ymax=np.max(new_pose[:,1])+20
        new_pose[:,1]= -new_pose[:,1]
        
        for i,link in enumerate(links):
            color=(0,0,0)
            start = new_pose[link[0],:]
            circle=Circle(start,2,fill=True,facecolor=color)
            ax.add_artist(circle)
            end = new_pose[link[1],:]
            circle=Circle(end,2,fill=True,facecolor=color)
            ax.add_artist(circle)

            center= (start+end)/2
            width = np.sqrt(np.sum((end-start)**2))
            height = 0.15 *width

            tan= (end[1]-start[1])/(end[0]-start[0]+eps)
            angle=np.arctan(tan)/np.pi *180

            stem=Ellipse(center,width=width,height=height,angle=angle,color=colors[i],alpha=0.5)
            ax.add_artist(stem)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        #ax.axis('off')

    #plt.show()
    plt.savefig('test.png', format='png', transparent=True, dpi=600)

def visual_pose(poses,size):
    colors=[]
    for link in links:
        colors.append((random.random(),random.random(),random.random()))

    fig, axs= plt.subplots(1,1,subplot_kw={'aspect': 'equal'})
    for ai,pose in enumerate(poses[:10,]):
        ax=axs  
        pose=pose.reshape(14,2)
        for i,link in enumerate(links):
            color=(0,0,0)
            start = pose[link[0],:]
            circle=Circle(start,2,fill=True,facecolor=color)
            ax.add_artist(circle)
            end = pose[link[1],:]
            circle=Circle(end,2,fill=True,facecolor=color)
            ax.add_artist(circle)

            center= (start+end)/2
            width = np.sqrt(np.sum((end-start)**2))
            height = 0.15 *width

            tan= (end[1]-start[1])/(end[0]-start[0]+eps)
            angle=np.arctan(tan)/np.pi *180

            stem=Ellipse(center,width=width,height=height,angle=angle,color=colors[i],alpha=0.5)
            ax.add_artist(stem)


            ##########for test
            circle=Circle((500,300),6,fill=True,facecolor=(0.5,0.5,0.5))
            ax.add_artist(circle)
        ax.set_xlim(0,size[1])
        ax.set_ylim(bottom=size[0],top =0)
        #ax.axis('off')

    plt.show()
