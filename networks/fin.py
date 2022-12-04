import tensorflow as tf
from tensorflow import keras

import utilities.math as mutils
import utilities.network as nutils

from networks.fin_modules import SpatialFourierCNN, SpatialFourierCNNGroup


class FourierImagerNetwork(keras.Model):

    def __init__(self, arch, in_shape, out_shape, seed=0, name="fin"):
        super(FourierImagerNetwork, self).__init__()
        self.seed = seed
        keras.utils.set_random_seed(self.seed)

        self.arch = arch
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.layers_ = list()
        self._name = name

        self._init_layers()

        self.build(input_shape=[None] + self.in_shape)

    def call(self, inputs, training=False):
        for layer in self.layers_:
            inputs = layer(inputs, training=training)
        return inputs

    def info(self, width=None):
        input_ = keras.Input(shape=self.in_shape, name="input")
        model = keras.Model(inputs=[input_], outputs=self.call(input_), name=self._name)
        model.summary(line_length=width)
        print()

    def variables(self):
        return self.trainable_weights

    def _init_layers(self):
        pre = self.arch["pre"]
        spaf = self.arch["spaf"]
        post = self.arch["post"]

        # PRE CONVOLUTIONS
        i = -1
        for i in range(len(pre["filters"])):
            self.layers_.append(keras.layers.Conv2D(
                filters=pre["filters"][i], kernel_size=pre["kernels"][i],
                strides=(pre["downsample"][i],) * 2, padding="same",
                kernel_initializer="he_normal", name=f"conv2d_{i + 1}"))
            nutils.add_activation(pre["activation"], self.layers_, i=i)
            if pre["batch_norm"]:
                self.layers_.append(keras.layers.BatchNormalization(name=f"batch_norm_{i + 1}"))

        # RESNET GROUPS
        j = -1
        for j in range(len(spaf["filters"])):
            self.layers_.append(SpatialFourierCNNGroup(
                filters=spaf["filters"][j], kernels=spaf["kernels"][j], repeat=spaf["repeat"][j],
                downsample=spaf["downsample"][j], activation=spaf["activation"],
                batch_norm=spaf["batch_norm"], long_skip=spaf["long_skip"],
                dct=spaf["dct"], type=spaf["type"], name=f"spaf_cnn_{i + j + 2}"))

        # POST CONVOLUTIONS
        k = -1
        for k in range(len(post["filters"])):
            self.layers_.append(keras.layers.Conv2D(
                filters=post["filters"][k], kernel_size=post["kernels"][k],
                strides=(post["downsample"][k],) * 2, padding="same",
                kernel_initializer="he_normal", name=f"conv2d_{i + j + k + 1}"))
            nutils.add_activation(post["activation"], self.layers_, i=i + j + k)
            if post["batch_norm"]:
                self.layers_.append(keras.layers.BatchNormalization(
                    name=f"batch_norm_{i + j + k + 3}"))

        # FLATTEN & FCN OUTPUT
        if self.arch["global_average_pooling"]:
            self.layers_.append(
                keras.layers.GlobalAveragePooling2D(name=f"global_avg_pool_{i + j + k + 4}"))
        else:
            self.layers_.append(keras.layers.Flatten(name=f"flatten_{i + j + k + 4}"))
        self.layers_.append(keras.layers.Dense(
            units=self.out_shape[-1], name=f"dense_{i + j + k + 4}"))
        nutils.add_activation(self.arch["activation_out"], self.layers_, i=i + j + k + 4)