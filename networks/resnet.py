from keras.layers import Input
from keras import layers
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.models import Model

import keras.backend as K

from networks.classifiers import build_pyramid_pooling_module


def BN(axis, name=""):
    return BatchNormalization(axis=axis, momentum=0.1, name=name, epsilon=1e-5)


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation=1):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        dilation: dilation of the intermediate convolution

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BN(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b', use_bias=False, dilation_rate=dilation)(x)
    x = BN(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BN(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1), dilation=1):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BN(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b', use_bias=False, dilation_rate=dilation)(x)
    x = BN(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BN(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BN(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet101(input_tensor=None):

    img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BN(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v', dilation=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='w', dilation=2)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dilation=4)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation=4)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation=4)

    return x


def build_network(nb_classes, input_shape, resnet_layers=101, classifier='psp', sigmoid=False, output_size=None,
                  num_input_channels=4):
    """Build Network"""
    inp = Input((input_shape[0], input_shape[1], num_input_channels))
    if resnet_layers == 101:
        res = ResNet101(inp)
    else:
        ValueError('Resnet {} does not exist'.format(resnet_layers))
    if classifier == 'psp':
        print("Building network based on ResNet %i and PSP module expecting inputs of shape %s predicting %i classes" % (
            resnet_layers, input_shape, nb_classes))
        x = build_pyramid_pooling_module(res, input_shape, nb_classes, sigmoid=sigmoid, output_size=output_size)
    else:
        raise ValueError('Classifier not implemented.')
    model = Model(inputs=inp, outputs=x)

    return model
