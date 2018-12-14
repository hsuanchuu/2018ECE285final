import numpy as np
import cv2
import glob
import itertools
from os.path import join, exists, basename
import os
import scipy.misc
import random

def getSegmentationArr( path , nClasses=7,  width=512, height=512):

    seg_labels = np.zeros((  height , width  , nClasses ))
    mask = scipy.misc.imread(path)
    mask = cv2.resize(mask, (width , height))    
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    
    seg_labels[mask==3, 0] = 1 # (Cyan: 011) Urban land 
    seg_labels[mask==6, 1] = 1 # (Yellow: 110) Agriculture land
    seg_labels[mask==5, 2] = 1 # (Purple: 101) Rangeland  
    seg_labels[mask==2, 3] = 1 # (Green: 010) Forest land 
    seg_labels[mask==1, 4] = 1 # (Blue: 001) Water
    seg_labels[mask==7, 5] = 1 # (White: 111) Barren land 
    seg_labels[mask==0, 6] = 1 # (Black: 000) Unknown 
    
    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels

def getImageArr( path , width , height , imgNorm="None" , odering='channels_last'):


    img = scipy.misc.imread(path)
    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
    elif imgNorm == "divide":
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)
        img = img/255.0

    if odering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img




def imageSegmentationGenerator( images_path , segs_path ,  batch_size=16,  n_classes=7 , input_height=512 , input_width=512 , output_height=512 , output_width=512):

    images = glob.glob(join(images_path,"*.jpg"))
    images.sort()
    segmentations = glob.glob(join(segs_path,"*.png"))
    segmentations.sort()

    assert len( images ) == len(segmentations)
    for im , seg in zip(images,segmentations):
        assert(  im.split('/')[-1].split("_")[0] ==  seg.split('/')[-1].split("_")[0] )

    zipped = itertools.cycle(zip(images,segmentations))

    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            im , seg = next(zipped)
            # if random.random() > 0.5:
            #     X.append(np.fliplr(getImageArr(im , input_width , input_height )  ))
            #     Y.append(np.fliplr(getSegmentationArr( seg , n_classes , output_width , output_height )  ))
            # else:
            X.append(getImageArr(im , input_width , input_height )  )
            Y.append(getSegmentationArr( seg , n_classes , output_width , output_height )  )

        yield np.array(X) , np.array(Y)


def getSegmentationCropArr( path , nClasses=7,  width=512, height=512):

    seg_labels = np.zeros((  height , width  , nClasses ))
    mask = scipy.misc.imread(path)
    mask = cv2.resize(mask, (width , height))    
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    
    seg_labels[mask==3, 0] = 1. # (Cyan: 011) Urban land 
    seg_labels[mask==6, 1] = 1. # (Yellow: 110) Agriculture land
    seg_labels[mask==5, 2] = 1. # (Purple: 101) Rangeland  
    seg_labels[mask==2, 3] = 1. # (Green: 010) Forest land 
    seg_labels[mask==1, 4] = 1. # (Blue: 001) Water
    seg_labels[mask==7, 5] = 1. # (White: 111) Barren land 
    seg_labels[mask==0, 6] = 1. # (Black: 000) Unknown 
    
    # seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels

def getImageCropArr( path , width , height , imgNorm="sub_mean" , odering='channels_last'):


    img = cv2.imread(path)
    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    elif imgNorm == "v":
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
    elif imgNorm == "divide":
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)
        img = img/255.0

    if odering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def imageSegmentationCropGenerator( images_path , segs_path ,  batch_size=16,  n_classes=7 , input_height=2448 , input_width=2448 , output_height=2448 , output_width=2448):

    images = glob.glob(join(images_path,"*.jpg"))
    images.sort()
    segmentations = glob.glob(join(segs_path,"*.png"))
    segmentations.sort()

    assert len( images ) == len(segmentations)
    for im , seg in zip(images,segmentations):
        assert(  im.split('/')[-1].split("_")[0] ==  seg.split('/')[-1].split("_")[0] )

    zipped = itertools.cycle(zip(images,segmentations))

    while True:
        X = []
        Y = []
        im , seg = next(zipped)
        im_tmp = getImageCropArr(im , input_width , input_height )
        seg_tmp = getSegmentationCropArr( seg , n_classes , output_width , output_height )
        for _ in range( batch_size) :
            
            x_index = np.random.choice(range(0,input_width-512))
            y_index = np.random.choice(range(0,input_height-512))
            # if random.random() > 0.5:
            #     X.append(np.fliplr(im_tmp[x_index:x_index+512,y_index:y_index+512,:]))
            #     Y.append(np.fliplr(seg_tmp[x_index:x_index+512,y_index:y_index+512,:]).reshape((512*512,7)))
            # else:
            X.append(im_tmp[x_index:x_index+512,y_index:y_index+512,:] )
            Y.append(seg_tmp[x_index:x_index+512,y_index:y_index+512,:].reshape((512*512,7)) )

        yield np.array(X) , np.array(Y)

