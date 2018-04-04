#!/usr/bin/env python
from os.path import splitext, join
import numpy as np
from scipy import misc
from keras import backend as K

import tensorflow as tf
from networks import resnet

from mypath import Path


class DEXTR(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights, num_input_channels=4,
                 classifier='psp', use_numpy=False, sigmoid=False):
        self.input_shape = input_shape
        self.num_input_channels = num_input_channels
        self.sigmoid = sigmoid
        self.model = resnet.build_network(nb_classes=nb_classes, resnet_layers=resnet_layers, num_input_channels=num_input_channels,
                                          input_shape=self.input_shape, classifier=classifier, sigmoid=self.sigmoid, output_size=self.input_shape)
        if use_numpy:
            print("No Keras model & weights found, import from npy weights.")
            self.set_npy_weights(weights)
        else:
            print("Loading weights from H5 file.")
            h5_path = join(Path.models_dir(), weights + '.h5')
            self.model.load_weights(h5_path)

    def predict(self, img):
        # Preprocess
        img = misc.imresize(img, self.input_shape)
        img = img.astype('float32')
        probs = self.feed_forward(img)
        return probs

    def feed_forward(self, data):
        print("Predicting...")
        assert data.shape == (self.input_shape[0], self.input_shape[1], self.num_input_channels)
        prediction = self.model.predict(np.expand_dims(data, 0))[0]
        print("Finished prediction...")
        return prediction

    def set_npy_weights(self, weights_path):
        npy_weights_path = join("weights", "npy", weights_path + ".npy")
        h5_path = join(Path.models_dir(), weights_path + '.h5')
        # h5_path_model = join("models", weights_path + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            # print('{}'.format(layer.name))
            if layer.name[:2] == 'bn' or layer.name[-2:] == 'bn':
                print('{}'.format(layer.name))
                # print('{} {}'.format(layer.name, layer.get_weights()[0].shape))
                gamma = weights[layer.name]['gamma']
                beta = weights[layer.name]['beta']
                moving_mean = weights[layer.name]['moving_mean']
                moving_variance = weights[layer.name]['moving_variance']

                self.model.get_layer(layer.name).set_weights([gamma, beta, moving_mean, moving_variance])

            elif layer.name[:3] == 'res' or layer.name[-4:] == 'conv' or layer.name[:4] == 'conv':
                print('{}'.format(layer.name))
                # print('{} {}'.format(layer.name, layer.get_weights()[0].shape))
                if len(self.model.get_layer(layer.name).get_weights()) == 2:
                    weight = weights[layer.name]['weights']
                    biases = weights[layer.name]['biases']
                    if biases is None:
                        raise ValueError('Bias inconsistency')
                    self.model.get_layer(layer.name).set_weights([weight, biases])
                else:
                    weight = weights[layer.name]['weights']
                    self.model.get_layer(layer.name).set_weights([weight])

        print('Finished importing weights.')

        print("Writing keras weights")
        self.model.save_weights(h5_path)
        # models.save_model(self.model, h5_path_model, include_optimizer=False)

        print("Finished writing Keras model & weights")


if __name__ == "__main__":
    classifier = 'psp'
    input_size = 512
    num_input_channels = 4
    resnet_size = 101
    image = 'ims/dog_512.png'
    extreme_points = 'ims/dog_512_extreme.png'

    input_type = 'bbox' if num_input_channels == 3 else 'extreme'

    model = 'dextr_pascal-sbd'

    # Handle input and output args
    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        dextr = DEXTR(nb_classes=1, resnet_layers=resnet_size, input_shape=(input_size, input_size), weights=model,
                      num_input_channels=num_input_channels, classifier=classifier, use_numpy=False, sigmoid=True)

        img = misc.imread(image, mode='RGB')
        if num_input_channels == 4:
            extreme = misc.imread(extreme_points)

            img_extreme = np.zeros((input_size, input_size, 4))
            img_extreme[:, :, :3] = img
            img_extreme[:, :, 3] = extreme
            img_extreme = np.expand_dims(img_extreme.astype('float32'), 0)
        else:
            img_extreme = np.expand_dims(img.astype('float32'), 0)

        pred = dextr.model.predict(img_extreme)[0, :, :, 0]

        mask = pred > 0.8

        filename, ext = splitext(image)

        misc.imsave(filename + "_seg_"+ classifier + ext, mask.astype(np.uint8)*255)
