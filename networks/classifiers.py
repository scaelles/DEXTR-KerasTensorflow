from math import ceil

from keras.layers.merge import Concatenate, Add
from keras.layers import AveragePooling2D, ZeroPadding2D

from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Layer

from networks import resnet
import keras.backend as K
from keras.backend import tf as ktf


class Upsampling(Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Upsampling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Upsampling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Upsampling, self).get_config()
        config['new_size'] = self.new_size
        return config


def psp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (512, 512):
        kernel_strides_map = {1: [64, 64],
                              2: [32, 32],
                              3: [22, 21],
                              6: [11, 9]}  # TODO: Level 6: Kernel correct, but stride not exactly the same as Pytorch
    else:
        raise ValueError("Pooling parameters for input shape " + input_shape + " are not defined.")

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    names = [
        "class_psp_" + str(level) + "_conv",
        "class_psp_" + str(level) + "_bn"
    ]
    kernel = (kernel_strides_map[level][0], kernel_strides_map[level][0])
    strides = (kernel_strides_map[level][1], kernel_strides_map[level][1])
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer)
    prev_layer = resnet.BN(bn_axis, name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Upsampling(feature_map_shape)(prev_layer)
    return prev_layer


def build_pyramid_pooling_module(res, input_shape, nb_classes, sigmoid=False, output_size=None):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = psp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = psp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = psp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = psp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    res = Concatenate()([interp_block1,
                         interp_block2,
                         interp_block3,
                         interp_block6,
                         res])
    x = Conv2D(512, (1, 1), strides=(1, 1), padding="same", name="class_psp_reduce_conv", use_bias=False)(res)
    x = resnet.BN(bn_axis, name="class_psp_reduce_bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="class_psp_final_conv")(x)

    if output_size:
        x = Upsampling(output_size)(x)

    if sigmoid:
        x = Activation('sigmoid')(x)
    return x
