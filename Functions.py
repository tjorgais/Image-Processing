# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 23:19:58 2021

@author: TSERING JORGAIS
"""
from pathlib import Path
import skimage.io
import skimage.filters
import numpy as np

import math
import time
import seaborn as sns                 # Require Installation of seaborn
from matplotlib import pyplot as plt
 

#Solution1

def compute_hist(image_path: Path, bins: int) -> list:
    image=skimage.io.imread(image_path.as_posix())
    skimage.io.imshow(image)
    my_freq, my_bins=my_hist(image)   #my_bins is bins centre here
    hist, edge=np.histogram(image, bins, range=(0,255))

    for i in range(10):
        edge[i]=(edge[i]+(edge[i+1]-edge[i])/2) #here we are storing bins centre
    edge=np.delete(edge,-1)
    plt.figure()
    sns.barplot(x=my_bins,y=my_freq)
    plt.xlabel('Bins centre')
    plt.ylabel('Frequency')
    plt.title('My Plot') 
    plt.figure()
    sns.barplot(x=edge,y=hist)
    plt.title('Inbuilt PLot',)
    plt.xlabel('Bins centre')
    plt.ylabel("Frequency")
    plt.show()


    print('displaying the bins centre and freq for inbulit function hist ')
    print('bins-centre: ', edge)
    print('frequency: ', hist)

    
    return[my_freq,my_bins, hist, edge]

def my_hist(image):
    row, col=image.shape
    initial_freq=np.zeros((256))
    for i in range(0,row):
        for j in range(0,col):
            initial_freq[image[i,j]]+=1
    spaces=255/10
    bins=np.arange(0,255.1,spaces)

    freq=np.zeros(10)

    y=0
    for i in range(10):
        x=y
        if(i!=9):   
            y=math.ceil(bins[i+1])
        else:
            y=math.ceil(bins[i+1]+1)
        for j in range(x,y):
            freq[i]+=initial_freq[j]             
    for i in range(10):
        bins[i]=(bins[i]+(bins[i+1]-bins[i])/2)
    print("displaying the bins-centre for my histogram: ")
    bins=np.delete(bins,-1)
    print('bins-centre: ', bins)
    freq=freq.astype(np.int)
    print('frequency: ', freq)

    return freq, bins


#Solution2

def otsu_binarization(image_path=Path):
    image=skimage.io.imread(image_path.as_posix())
    skimage.io.imshow(image)
    hist, edges=np.histogram(image, bins=256, range=(0,255))
    #print(hist)
    edges=(np.ceil(edges)).astype(int)
    
    row, col=image.shape
    start_w=time.time()
    t_w= sigma_w(hist, row*col,edges)
    end_w=time.time()

    start_b=time.time()
    t_b= sigma_b(hist, row*col,edges)
    end_b=time.time()

    t = skimage.filters.threshold_otsu(image)

    image[image<t]=0
    image[image>=t]=255

    return [end_w-start_w, end_b-start_b,t_w,t_b,image,t]

def sigma_w(hist,total_pix,edges):
    #normalised=hist/total_pix  #vectorization
    omega_0=0
    omega_1=0
    min_w=100000
    t=0
    for i in range(1,256):
        omega_0=np.sum(hist[:i])/total_pix
        omega_1=1-omega_0
        mean_0=np.sum((edges[0:i]*hist[0:i])/total_pix)/omega_0
        mean_1=np.sum((edges[i:-1]*hist[i:])/total_pix)/omega_1
        sigma_0=np.sum((np.square(edges[:i]-mean_0)*hist[:i])/total_pix)/omega_0
        sigma_1=np.sum((np.square(edges[i:-1]-mean_1)*hist[i:])/total_pix)/omega_1
        sig_w=omega_0*sigma_0+omega_1*sigma_1

        if(sig_w<min_w):
            min_w=sig_w
            t=i-1
    return t

def sigma_b(hist, total_pix, edges):
    #normalised=hist/total_pix  #vectorization
    omega_0=0
    omega_1=0
    max_b=-1
    t=0
    for i in range(1,256):
        omega_0=np.sum(hist[:i])/total_pix
        omega_1=1-omega_0
        mean_0=np.sum((edges[0:i]*hist[0:i])/total_pix)/omega_0
        mean_1=np.sum((edges[i:-1]*hist[i:])/total_pix)/omega_1
        sig_b=omega_0*omega_1*np.square(mean_1-mean_0)
        if(sig_b>max_b):
            max_b=sig_b
            t=i-1
    return t


#solution3

def foreground_extraction(text_path=Path, background_path=Path):
    text_image=skimage.io.imread(text_path.as_posix())
    back_img=skimage.io.imread(background_path.as_posix())
    '''
    plt.figure()
    plt.subplot(121)
    skimage.io.imshow(text_image)
    plt.subplot(122)
    skimage.io.imshow(back_img)
    '''
    hist, edges=np.histogram(text_image, bins=256, range=(0,255))
    #print(hist)
    edges=(np.ceil(edges)).astype(int)
    row, col=text_image.shape
    t=sigma_b(hist, row*col, edges)
    #print(t)
    otsu_image=text_image.copy()
    otsu_image[text_image<t]=0
    otsu_image[text_image>=t]=255
    
    plt.figure()
    skimage.io.imshow(otsu_image)
    

    otsu_image=np.stack((otsu_image,)*3,axis=-1)

    otsu_image[:,:,1]=0
    otsu_image[:,:,2]=0
    #print(back_img)
    back_img[otsu_image[:,:,0]==255]=0
    '''
    plt.figure()
    skimage.io.imshow(otsu_image)
    '''
    final_image=otsu_image+back_img
    '''
    plt.figure()
    skimage.io.imshow(final_image)
    '''

    
    
    return final_image
        
  #solution4

def connected_component(image_path=Path):
    image=skimage.io.imread(image_path.as_posix())
    plt.figure()
    skimage.io.imshow(image)
    hist, edges=np.histogram(image, bins=256, range=(0,255))
    #print(hist)
    edges=(np.ceil(edges)).astype(int)
    row, col=image.shape
    t=sigma_b(hist, row*col, edges)
    bin_image=image.copy()
    bin_image[image>t]=255
    bin_image[image<=t]=0
    plt.figure()
    skimage.io.imshow(bin_image)
    digits, count_1=digits_count(bin_image)

    return digits, count_1
          
def digits_count(image):
    row, col= image.shape
    region=np.zeros((row, col))
    region=region.astype(int)
    k=0
    for i in range(row):
        for j in range(col):
            if (image[i,j]==255):
                if image[i,j-1]==0 and image[i-1,j]==0:
                    region[i,j]=k
                    k=k+1
                
                elif image[i,j-1]==0 and image[i-1,j]==255:
                    region[i,j]=region[i-1,j]
                elif image[i,j-1]==255 and image[i-1,j]==0:
                    region[i,j]=region[i,j-1]
                else:
                    if(region[i,j-1] == region[i-1,j]):
                        region[i,j]=region[i,j-1]
                    else:
                        x=min(region[i,j-1],region[i-1,j])
                        y=max(region[i,j-1],region[i-1,j])
                        region[i,j]=x
                        region[region==y]=x      #using vectorization                        
                            
    set_k, counts=np.unique(region,return_counts=True)        #to find unique value in the region which is number of 
    #freq=np.asarray((unique, counts)).T
    my_dict = {set_k[i]: counts[i] for i in range(len(set_k))}
    dist_count=set(counts)
    dist_count=sorted(dist_count)
    total_1 = np.sum(x == dist_count[1] for x in my_dict.values())
    '''
    for i in range(len(set_k)):
        if(counts[i]>dist_count[0] and count[i]<dist_count[2]):
    '''        
    final_k=len(set_k)
                
    return final_k, total_1


#solution5

def binary_morphology(image_path=Path):
    image=skimage.io.imread(image_path.as_posix())
    plt.figure()
    skimage.io.imshow(image)
    hist, edges=np.histogram(image, bins=256, range=(0,255))
    #print(hist)
    edges=(np.ceil(edges)).astype(int)
    row, col=image.shape
    t=sigma_b(hist, row*col, edges)
    bin_image=image.copy()
    bin_image[image>=t]=255
    bin_image[image<t]=0
    plt.figure()
    skimage.io.imshow(bin_image)
    cleared_image=erosion(bin_image)
    '''
    plt.figure()
    plt.imshow(ero_image, cmap='gray')
    '''
    cleared_image=dilation(cleared_image)
    #cleared_image=dilation(cleared_image)   #The more no of times you apply dilation the cleaner you get but the cost is time taken.
    
    '''
    plt.figure()
    plt.imshow(ero_image, cmap='gray')
    '''
    return cleared_image
 
def erosion(image):
    win=np.ones((3,3), np.int8)
    #win=win.astype(int)   #window
    row, col=image.shape
    new_image=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if i==0 and j==0:
                x=min(image[i,j]*win[1,1],image[i,j+1]*win[1,2],image[i+1,j]*win[2,1],image[i+1,j+1]*win[2,2])
            elif i==0 and (j in range(1,col-1)):
                x=min(image[i,j]*win[1,1],image[i,j-1]*win[1,0],image[i,j+1]*win[1,2],image[i+1,j-1]*win[2,0],image[i+1,j]*win[2,1],image[i+1,j+1]*win[2,2])
            elif (i in range(1,row-1) and j==0):
                x=min(image[i,j]*win[1,1],image[i-1,j]*win[0,1],image[i+1,j]*win[2,1],image[i-1,j+1]*win[0,2],image[i,j+1]*win[1,2],image[i+1,j+1]*win[2,2])
            elif i==0 and j==col-1: 
                x=min(image[i+1,j-1]*win[2,0], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i+1,j]*win[2,1])
            elif i==row-1 and j==0:
                x=min(image[i,j]*win[1,1],image[i-1,j]*win[0,1],image[i-1,j+1]*win[0,2])
            elif i==row-1 and (j in range(1, col-1)):

                x=min(image[i-1,j]*win[0,1], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i,j+1]*win[1,2], image[i-1,j-1]*win[0,0], image[i-1,j+1]*win[0,2])
            elif i in range(1,row-1) and j== col-1:
                x=min(image[i-1,j]*win[0,1], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i+1,j]*win[2,1],image[i-1,j-1]*win[0,0], image[i+1,j-1]*win[2,0])
            elif i==row-1 and j==col-1:
                x=min(image[i-1,j]*win[0,1], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i-1,j-1]*win[0,0])
            else:
                x=min(image[i-1,j-1]*win[0,0], image[i-1,j]*win[0,1], image[i-1,j-1]*win[0,2], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i,j+1]*win[1,2], image[i+1,j-1]*win[2,0], image[i+1,j]*win[2,1], image[i+1,j+1]*win[2,2])
            
            new_image[i,j]=x
    
    
    return new_image

def dilation(image):
    win=np.ones((3,3), np.int8)
    #win=win.astype(int)   #window
    row, col=image.shape
    new_image=np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if i==0 and j==0:
                x=max(image[i,j]*win[1,1],image[i,j+1]*win[1,2],image[i+1,j]*win[2,1],image[i+1,j+1]*win[2,2])
            elif i==0 and (j in range(1,col-1)):
                x=max(image[i,j]*win[1,1],image[i,j-1]*win[1,0],image[i,j+1]*win[1,2],image[i+1,j-1]*win[2,0],image[i+1,j]*win[2,1],image[i+1,j+1]*win[2,2])
            elif (i in range(1,row-1) and j==0):
                x=max(image[i,j]*win[1,1],image[i-1,j]*win[0,1],image[i+1,j]*win[2,1],image[i-1,j+1]*win[0,2],image[i,j+1]*win[1,2],image[i+1,j+1]*win[2,2])
            elif i==0 and j==col-1: 
                x=max(image[i+1,j-1]*win[2,0], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i+1,j]*win[2,1])
            elif i==row-1 and j==0:
                x=max(image[i,j]*win[1,1],image[i-1,j]*win[0,1],image[i-1,j+1]*win[0,2])
            elif i==row-1 and (j in range(1, col-1)):

                x=max(image[i-1,j]*win[0,1], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i,j+1]*win[1,2], image[i-1,j-1]*win[0,0], image[i-1,j+1]*win[0,2])
            elif i in range(1,row-1) and j== col-1:
                x=max(image[i-1,j]*win[0,1], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i+1,j]*win[2,1],image[i-1,j-1]*win[0,0], image[i+1,j-1]*win[2,0])
            elif i==row-1 and j==col-1:
                x=max(image[i-1,j]*win[0,1], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i-1,j-1]*win[0,0])
            else:
                x=max(image[i-1,j-1]*win[0,0], image[i-1,j]*win[0,1], image[i-1,j-1]*win[0,2], image[i,j-1]*win[1,0], image[i,j]*win[1,1], image[i,j+1]*win[1,2], image[i+1,j-1]*win[2,0], image[i+1,j]*win[2,1], image[i+1,j+1]*win[2,2])
            
            new_image[i,j]=x
    
    
    return new_image
     
            
        
        
        
    
    
    
    



