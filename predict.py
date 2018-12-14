import argparse
import FCN32, FCN8, FCN16, VGGUnet, dataloader
from keras.models import load_model

import cv2
import numpy as np
import random
from skimage import io
from skimage.transform import resize
from os import mkdir
from os.path import join, exists, basename
from glob import glob
import os
import scipy.misc
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.optimizers import SGD, Adam
import scipy.misc



parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 512 )
parser.add_argument("--input_width", type=int , default = 512 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int, default=7 )

args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)



n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number


if not exists(args.output_path):
    mkdir(args.output_path)



# modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFns = {'fcn32': FCN32.FCN32, 'fcn8':FCN8.FCN8, 'vgg_unet':VGGUnet.VGGUnet, 'fcn16': FCN16.FCN16}

modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width, load_pretrain=False)
m.load_weights( args.save_weights_path)
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob( join(images_path, "*.jpg")) #+ glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

#Color
# colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]
colors = [(0,255,255), (255,255,0), (255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]

for imgName in images:
	outName = imgName.replace( images_path,  args.output_path).replace("sat","mask").replace("jpg","png")
	X = dataloader.getImageArr(imgName, args.input_width, args.input_height)
	pr = m.predict( np.array([X]))[0]
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax(axis=-1)
	seg_img = np.zeros(( output_height , output_width , 3 ))
	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ( (pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ( (pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (input_width , input_height ))
	scipy.misc.imsave(outName, seg_img )
