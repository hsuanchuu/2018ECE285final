from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *
#from renet_helper import *
from keras.models import *
from keras.layers import *

weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
IMAGE_ORDERING = 'channels_last'

# crop o1 wrt o2
def crop( o1 , o2 , i  ):
    o_shape2 = Model( i  , o2 ).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model( i  , o1 ).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)
    
    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

    return o1 , o2 


def FCN_Resnet50( classes ,  input_height=512, input_width=512 , vgg_level=3, load_pretrain=True):
    img_input = Input(shape=(input_height, input_width, 3))
    
    batch_momentum=0.9
    weight_decay=0.
    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    f1 = x

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)
    f2 = x

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)
    f3 = x

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)
    f4 = x


    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)
    f5 = x

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1000, activation='softmax', name='fc1000')(x)
    #classifying layer
    #x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    #x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)
    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path)
    #weights_path = keras_utils.get_file(
    #            'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
    #             WEIGHTS_PATH,                
    #            md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    #model.load_weights(weights_path)
    o = f5

    o = ( Conv2D( 2048 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 2048 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = ( Conv2D( classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose( classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o2 = f4
    o2 = ( Conv2D( classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)
    
    o , o2 = crop( o , o2 , img_input )
    
    o = Add()([ o , o2 ])

    o = Conv2DTranspose( classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
    o2 = f3 
    o2 = ( Conv2D( classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)
    o2 , o = crop( o2 , o , img_input )
    o  = Add()([ o2 , o ])

#     convsize = int(o.shape[2])
#     deconv_output_size = (convsize - 1) * 8 + 16  # INFO: =34 when images are 512x512
#     extra_margin = deconv_output_size - convsize * 8  # INFO: =2 when images are 512x512
#     assert (extra_margin > 0)
#     assert (extra_margin % 2 == 0)
#     print("extra_margin",extra_margin)
   
#     c = ((int(extra_margin/2), int(extra_margin/2)),(int(extra_margin/2), int(extra_margin/2)))

    o = Conv2DTranspose( classes, kernel_size=(16,16), strides=(8,8) , padding='same', use_bias=False, data_format=IMAGE_ORDERING )(o)
#     o = Cropping2D(cropping=c)(o)

    o_shape = Model(img_input , o ).output_shape
    
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape(( outputHeight*outputWidth,-1 )))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight


    return model

