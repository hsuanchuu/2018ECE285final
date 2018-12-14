import argparse
import FCN32, FCN8, FCN16, VGGUnet, dataloader
import numpy as np
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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import History


history = History()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
set_session(sess)


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int, default = 7 )
parser.add_argument("--input_height", type=int , default = 512  )
parser.add_argument("--input_width", type=int , default = 512 )

parser.add_argument("--validate", action='store_true')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 30 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name


TRAIN_PATH = 'train'
VAL_PATH = 'validation'
OUTPUT_PATH = 'output'

TRAIN_SIZE = 2313
VALID_SIZE = 257


if not exists(save_weights_path):
    mkdir(save_weights_path)



if validate:
    val_images_path = args.val_images
    val_segs_path = args.val_annotations
    val_batch_size = args.val_batch_size

modelFns = {'fcn32': FCN32.FCN32, 'fcn8':FCN8.FCN8, 'vgg_unet':VGGUnet.VGGUnet, 'fcn16':FCN16.FCN16}

# modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

# m = modelFN(n_classes, input_height=input_height, input_width=input_width, load_pretrain=True)



if len( load_weights ) > 0:
    m = modelFN(n_classes, input_height=input_height, input_width=input_width, load_pretrain=False)
    m.load_weights(load_weights)
else:
    m = modelFN(n_classes, input_height=input_height, input_width=input_width, load_pretrain=True)

print(m.summary())
m.compile(loss='categorical_crossentropy',
      optimizer= Adam(lr=0.0001),
      metrics=['accuracy'])

print("Model output shape",m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G  = dataloader.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

print("validate",validate)
print("batch_size",train_batch_size)
if validate:
    print("Validate applied")
    G2  = dataloader.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
    for ep in range( epochs ):
        m.fit_generator( G , steps_per_epoch=512, epochs=1, workers=4)
        if ep%5==0:
            m.save_weights(join(save_weights_path, model_name + "_" + str( ep )))
        # m.save( save_weights_path + ".model." + str( ep ) )
else:
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for ep in range( epochs ):
        m.fit_generator( G , steps_per_epoch=512, validation_data=G2 , validation_steps=100 , epochs=1, workers=4, callbacks=[history])
        if ep%5==0:
            m.save_weights(join(save_weights_path, model_name + "_" + str( ep )))
        print(history.history)
        #print(history.history['acc'])
        #print(len(history.history['acc']))
        #print(history.history['acc'][0])
        acc.append(history.history['acc'][0])
        val_acc.append(history.history['val_acc'][0])
        loss.append(history.history['loss'][0])
        val_loss.append(history.history['val_loss'][0])
    x = [i for i in range(len(acc))]
    plt.figure()
    plt.plot(x,acc)
    plt.plot(x,val_acc)
    plt.title('model_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','val'], loc = 'upper left')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(x,loss)
    plt.plot(x,val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig('loss.png') 
        # m.save( save_weights_path + ".model." + str( ep ) )


