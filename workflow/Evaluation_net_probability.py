# -*- coding: utf-8 -*-

'''
Target: evaluate your trained caffe model with the medical images. I use simpleITK to read medical images (hdr, nii, nii.gz, mha and so on)  
Created on March 6th, 2017
Author: Dong Nie 
Note, this is specified for classifying, so I implement the majority voting so that the performance would be stable if highly overlap happens
Also, the input patch can larger than output patch
Moreover, this can be used to generate single-scale or multi-scale
'''


import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from scipy import ndimage as nd

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/caffe3/'  # this is the path in GPU server
#caffe_root = '/home/dongnie/caffe3D/'  # this is the path in GPU server
#caffe_root = '/usr/bin/caffe'  # this is the path in GPU server
import sys
sys.path.insert(0, caffe_root + 'python')
print (caffe_root + 'python')
import caffe

caffe.set_device(7) #very important
caffe.set_mode_gpu()
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
#solver = caffe.SGDSolver('infant_fcn_solver.prototxt') #for training
#protopath='/home/dongnie/caffe3D/examples/prostate/'
#protopath='/home/dongnie/caffe3D/examples/pelvicSeg/'
protopath='/shenlab/lab_stor/liwang/more_patch/'
#mynet = caffe.Net(protopath+'prostate_deploy_v12_1.prototxt',protopath+'prostate_fcn_v12_1_iter_2170000.caffemodel',caffe.TEST)
mynet = caffe.Net(protopath+'infant_deploy.prototxt',protopath+'yoursavename_iter_511000.caffemodel',caffe.TEST)
print("blobs {}\nparams {}".format(mynet.blobs.keys(), mynet.params.keys()))

d1=32
d2=32
d3=32
dFA=[d1,d2,d3]
dSeg=[32,32,32]

step1=8
step2=8
step3=8
step=[step1,step2,step3]
NumOfClass=4 #the number of classes in this segmentation project
print(step1)
def cropCubic(matFA,matMR,fileID,d,step,rate):
    eps=1e-5
    #transpose
    matFA=np.transpose(matFA,(0,2,1))
    matMR=np.transpose(matMR,(0,2,1))
   # matSeg=np.transpose(matSeg,(0,2,1))
    [row,col,leng]=matFA.shape
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    
    print('matFA shape is ',matFA.shape)
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print('matFAOut shape is ',matFAOut.shape)
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA


    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    matMROut=matMR #note, if the input size and out size is different, you should do the same thing (above 10 lines) for matMR
  
    matFAOutScale = nd.interpolation.zoom(matFAOut, zoom=rate)
    matMROutScale = nd.interpolation.zoom(matMROut, zoom=rate)

    matOut=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2],NumOfClass))
    BG=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2]))
    CSF=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2]))
    GM=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2]))
    WM=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2]))
    Visit=np.zeros((matFA.shape[0],matFA.shape[1],matFA.shape[2]))+eps	
    [row,col,leng]=matFA.shape
        
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-d[0]+1,step[0]):
        for j in range(0,col-d[1]+1,step[1]):
            for k in range(0,leng-d[2]+1,step[2]):
               # volSeg=matSeg[i:i+d[0],j:j+d[1],k:k+d[2]]
                #print 'volSeg shape is ',volSeg.shape
                volFA=matFAOutScale[i:i+d[0]+2*marginD[0],j:j+d[1]+2*marginD[1],k:k+d[2]+2*marginD[2]]
                volMR=matMROutScale[i:i+d[0]+2*marginD[0],j:j+d[1]+2*marginD[1],k:k+d[2]+2*marginD[2]]

        if np.sum(volFA) > 1 :
                
            volFA=np.float64(volFA)
            volMR=np.float64(volMR)

            #print 'volFA shape is ',volFA.shape
            mynet.blobs['dataT1'].data[0,0,...]=volFA
            mynet.blobs['dataT2'].data[0,0,...]=volMR

            mynet.forward()
            temppremat = mynet.blobs['softmax'].data #Note you have add softmax layer in deploy prototxt
            Visit[i:i+d[0],j:j+d[1],k:k+d[2]]=Visit[i:i+d[0],j:j+d[1],k:k+d[2]]+1
            BG[i:i+d[0],j:j+d[1],k:k+d[2]]=BG[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat[0,0,...]
            CSF[i:i+d[0],j:j+d[1],k:k+d[2]]=CSF[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat[0,1,...]
            GM[i:i+d[0],j:j+d[1],k:k+d[2]]=GM[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat[0,2,...]
            WM[i:i+d[0],j:j+d[1],k:k+d[2]]=WM[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat[0,3,...]
                #print 'temppremat shape is ',temppremat.shape
                #temppremat = mynet.blobs['conv3e'].data[0] #Note you have add softmax layer in deploy prototxt
                #temppremat=np.zeros([volSeg.shape[0],volSeg.shape[1],volSeg.shape[2]])
                
#                 matOut[i:i+d[0],j:j+d[1],k:k+d[2]]=matOut[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat
#                 used[i:i+d[0],j:j+d[1],k:k+d[2]]=used[i:i+d[0],j:j+d[1],k:k+d[2]]+1
                #for labelInd in range(NumOfClass): #note, start from 0
                 #   currLabelMat = np.where(temppremat==labelInd, 1, 0) # true, vote for 1, otherwise 0
                 #   matOut[i:i+d[0],j:j+d[1],k:k+d[2],labelInd]=matOut[i:i+d[0],j:j+d[1],k:k+d[2],labelInd]+currLabelMat;
    
    #matOut=matOut.argmax(axis=3) #always 3
#     matOut=matOut/used
    #matOut=np.rint(matOut)
    BG = BG/Visit
    WM = WM/Visit
    GM = GM/Visit
    CSF = CSF/Visit
    matOut[...,0]=BG
    matOut[...,1]=CSF
    matOut[...,2]=GM
    matOut[...,3]=WM
    matOut=matOut.argmax(axis=3) #always 3
    matOut=np.rint(matOut)

    matOut=np.transpose(matOut,(0,2,1))
    WM=np.transpose(WM,(0,2,1))
    GM=np.transpose(GM,(0,2,1))
    CSF=np.transpose(CSF,(0,2,1))
   # matSegScale=np.transpose(matSegScale,(0,2,1))
    return WM, GM, CSF, matOut


#this function is used to compute the dice ratio
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


def main():
    #datapath='/home/dongnie/warehouse/xxx/'
    datapath='/shenlab/lab_stor/liwang/DenseNet_Archive/DensetNet_grid_hist/'
    #datapath='/home/dongnie/warehouse/NDAR_ACE/'
 
    #ids=[51]
    ids=[121,125,127,131,125,140,39,21,22,23,24,25,501,502,503,504,505,506] 
    #ids=range(1,21)
    for i in range(0, len(ids)):
        myid=ids[i]   
        print(myid)
        #datafilename='prostate_%dto1_MRI.nii'%myid
        #datafilename='img%d.mhd'%myid
        dataT1filename='NORMAL0%d_cbq-h.hdr'%myid
        dataT1fn=os.path.join(datapath,dataT1filename)
        
        dataT2filename='NORMAL0%d_cbq-T2-h.hdr'%myid
        dataT2fn=os.path.join(datapath,dataT2filename)
        
     
        #labelfilename='prostate_%dto1_CT.nii'%myid  # provide a sample name of your filename of ground truth here
        #labelfilename='NORMAL0%d-ls-corrected.hdr'%myid  # provide a sample name of your filename of ground truth here

        #labelfn=os.path.join(datapath,labelfilename)
        imgOrg=sitk.ReadImage(dataT1fn)
        mrimgT1=sitk.GetArrayFromImage(imgOrg)
        
        imgOrg=sitk.ReadImage(dataT2fn)
        mrimgT2=sitk.GetArrayFromImage(imgOrg)
        
       #  mrimgT3=np.float64(mrimgT3)
#         mu=np.mean(mrimg)
#         maxV=np.max(mrimg)
#          minV=np.min(mrimg)
#         print mrimg.dtype
#          #mrimg=float(mrimg)
#          mrimg=(mrimg-mu)/(maxV-minV)
        #labelOrg=sitk.ReadImage(labelfn)
       # labelimg=sitk.GetArrayFromImage(labelOrg) 
        #you can do what you want here for for your label img
        
        fileID='%d'%myid
        rate=1
        WM, GM, CSF, LABEL = cropCubic(mrimgT1,mrimgT2,fileID,dSeg,step,rate)
        volOut=sitk.GetImageFromArray(WM)
        sitk.WriteImage(volOut,'preSub%d-WM.nii.gz'%myid)
        volOut=sitk.GetImageFromArray(GM)
        sitk.WriteImage(volOut,'preSub%d-GM.nii.gz'%myid)
        volOut=sitk.GetImageFromArray(CSF)
        sitk.WriteImage(volOut,'preSub%d-CSF.nii.gz'%myid)
        volOut=sitk.GetImageFromArray(LABEL)
        sitk.WriteImage(volOut,'preSub%d-label-511000.nii.gz'%myid)
        #volSeg=sitk.GetImageFromArray(matSeg)
        #sitk.WriteImage(volOut,'gt%d.nii.gz'%myid)
        #np.save('preSub'+fileID+'.npy',matOut)
        # here you can make it round to nearest integer 
        #now we can compute dice ratio

if __name__ == '__main__':     
    main()
