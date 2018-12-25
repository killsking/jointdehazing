import os
import numpy as np

import sys
caffe_root='/home/killsking/Desktop/caffe/'   
sys.path.insert(0,caffe_root+'python')
import caffe
import cv2
import math
def gettestfiles():
    filepath='/home/killsking/Desktop/dehazing/FastImageProcessing/data/dehazing/input/'
    pathDir =  os.listdir(filepath)
    output_names=[]
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        #print child()
        #cv2.imread(child)
        output_names.append(child )
    return output_names

def DarkChannel(im,sz):
	b,g,r = cv2.split(im)
	dc = cv2.min(cv2.min(r,g),b)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
	dark = cv2.erode(dc,kernel)
	return dark
def Guidedfilter(im,p,r,eps):
	mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
	mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
	mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
	cov_Ip = mean_Ip - mean_I*mean_p
	mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
	var_I   = mean_II - mean_I*mean_I
	a = cov_Ip/(var_I + eps)
	b = mean_p - a*mean_I
	mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
	mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
	q = mean_a*im + mean_b
	return q

def AtmLight(im,dark):
	[h,w] = im.shape[:2]
	imsz = h*w
	numpx = int(max(math.floor(imsz/1000),1))
	darkvec = dark.reshape(imsz,1)
	imvec = im.reshape(imsz,3)
	indices = darkvec.argsort()
	indices = indices[imsz-numpx::]
	atmsum = np.zeros([1,3])
	for ind in range(1,numpx):
		atmsum = atmsum + imvec[indices[ind]]
	A = atmsum / numpx
        return A

def TransmissionRefine(im,et):
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	gray = np.float64(gray)/255
	r = 60
	eps = 0.0001
	t = Guidedfilter(gray,et,r,eps)
        return t

def test():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    deploy='DeployT.prototxt'
    caffemodel='joint.caffemodel'
    net = caffe.Net(deploy, caffemodel, caffe.TEST);
    val_names=gettestfiles()
    for indimg in range(len(val_names)):
    #for indimg in range(1):
      img = cv2.imread(val_names[indimg])
      #print val_names[ind]
      I=img
      img=img/255.0
      
      
      allhazingpixel={}
      count=0
      h = img.shape[0]
      w = img.shape[1]
      hazebin=5
      rgb=np.ones((52,52,52),np.int32)*(-1)
      for i in range(h):
          for j in range(w):
              r=int(I[i,j,0]/hazebin)
              g=int(I[i,j,1]/hazebin)
              b=int(I[i,j,2]/hazebin)
              
              if rgb[r,g,b]>0:
                 continue
              else:
                 allhazingpixel[count]=I[i,j,:]
                 rgb[r,g,b]=count
                 count=count+1
                 #print count
      tcount=np.zeros((count,100),np.int32)
      finalt=  np.zeros((count),np.float)        
      dark = DarkChannel(img,15)
      A = AtmLight(img,dark)
      #for ind in range(0,3):
          #A[0,ind]=0.53

      low_bound=cv2.min(cv2.min(1-img[:,:,0]/A[0,0],1-img[:,:,1]/A[0,1]),1-img[:,:,2]/A[0,2])
      for ind in range(0,3):
          A[0,ind]=0.8
      A[0,0]=0.5
      A[0,1]=0.6
      A[0,2]=1
      for ind in range(0,3):
          img[:,:,ind] = img[:,:,ind]-A[0,ind]
      
      Radias_hazy=np.sqrt(img[:,:,1]*img[:,:,1]+img[:,:,2]*img[:,:,2]+img[:,:,0]*img[:,:,0])


      #print h,w
      img_data=img.transpose((2,0,1))
      net.blobs['data'].reshape(1, *img_data.shape)
      net.blobs['data'].data[...] = img_data

      out = net.forward()
      detail=net.blobs['convc'].data
      print detail.shape
      detail=detail.transpose((0,2,3,1))
      detail=np.reshape(detail,I.shape)
      Radias_clear=np.sqrt(detail[:,:,1]*detail[:,:,1]+detail[:,:,2]*detail[:,:,2]+detail[:,:,0]*detail[:,:,0])
      t=Radias_hazy/Radias_clear
      
      t=cv2.max(low_bound,t)
      t=cv2.min(t,1)
      
      for i in range(h):
          for j in range(w):
              r=int(I[i,j,0]/hazebin)
              g=int(I[i,j,1]/hazebin)
              b=int(I[i,j,2]/hazebin)
              index=rgb[r,g,b]
              tmpt=int(t[i,j]*99)
              #print tmpt
              tcount[index,tmpt]=tcount[index,tmpt]+1

      for i in range(count):
          tmpcount=0
          for j in range(100):
              if tmpcount<tcount[i,j]:
                 tmpcount=tcount[i,j]
                 finalt[i]=j/99.0 
                 #break
                 #print  j,finalt[i]


      for i in range(h):
          for j in range(w):
              r=int(I[i,j,0]/hazebin)
              g=int(I[i,j,1]/hazebin)
              b=int(I[i,j,2]/hazebin)
              index=rgb[r,g,b]
              t[i,j]=finalt[index]

      #cv2.imshow('img', t)
      #cv2.waitKey(0)
      #t=TransmissionRefine(I,t)
      t=cv2.max(t,0.1)
      res=img
      for ind in range(0,3):
          res[:,:,ind] = img[:,:,ind]/t+A[0,ind]
          res[:,:,ind] = detail[:,:,ind]+A[0,ind]
          #print res[:,:,1]
      cv2.imshow('img', t)
      cv2.waitKey(0)
      cv2.imwrite("result/%s"%val_names[indimg].split("/")[-1],res*255)

def main():
    test()


if __name__ == '__main__':
    main();
