import tensorflow as tf
from tensorflow import keras

import utilities.math as mutils
import utilities.network as nutils


class DCT(keras.layers.Layer):

    def __init__(self, name="dct"):
        super(DCT, self).__init__()
        self._name = name

    def call(self, inputs, training=False):
        inputs = mutils.dct(inputs)
        return inputs


class iDCT(keras.layers.Layer):

    def __init__(self, name="idct"):
        super(iDCT, self).__init__()
        self._name = name

    def call(self, inputs, training=False):
        inputs = mutils.idct(inputs)
        return inputs


class SpatialFourierCNN(keras.layers.Layer):
    """Spatial Fourier CNN base class, default Fourier domain learning"""
    def __init__(self, filters, kernels, downsample=1, activation="relu", batch_norm=True,
                 dct=True, name="spaf_cnn"):
        super(SpatialFourierCNN, self).__init__()
        self.filters = filters
        self.kernels = kernels
        self.downsample = downsample
        self.activation = activation
        self.batch_norm = batch_norm
        self.dct = dct
        self._name = name

        self.layers_ = []
        self.layers_fourier = []
        self.layers_skip = []

        self._init_layers()

    def call(self, inputs, training=False):
        skip = tf.identity(inputs)
        for layer in self.layers_:
            inputs = layer(inputs, training=training)
        for layer in self.layers_skip:
            skip = layer(skip, training=training)
        inputs = inputs + skip  # no activation/norm after sum: ResNet V2
        return inputs

    def _init_layers(self):
        # STANDARD RESNET CONVOLUTIONS IN FOURIER DOMAIN
        if self.dct:  # transform to Fourier domain
            self.layers_.append(DCT(name=f"dct_0"))
        for i in range(len(self.filters)):  # standard ResNet convolutions
            self.layers_.append(keras.layers.Conv2D(
                filters=self.filters[i], kernel_size=self.kernels[i],
                strides=(self.downsample,) * 2 if i == 0 else (1, 1),
                padding="same", kernel_initializer="he_normal", name=f"conv2d_{i + 1}"))
            nutils.add_activation(self.activation, self.layers_, i=i)
            if self.batch_norm and not self.dct:  # no batch normalization in Fourier domain
                self.layers_.append(keras.layers.BatchNormalization(name=f"batch_norm_{i + 1}"))
        if self.dct:  # transform back to Real domain
            self.layers_.append(iDCT(name=f"idct_{i + 2}"))
            if self.batch_norm:  # only apply batch normalization after Fourier -> real/discrete
                self.layers_.append(keras.layers.BatchNormalization(name=f"batch_norm_{i + 2}"))

        # STANDARD SKIP CONNECTION
        if self.downsample > 1:  # do nothing if there no downsample
            self.layers_skip.append(keras.layers.Conv2D(
                filters=self.filters[-1], kernel_size=1, padding="same",
                kernel_initializer="he_normal", name=f"conv2d_skip"))
            self.layers_skip.append(keras.layers.MaxPool2D(  # use maxpool instead of stride ^
                pool_size=(self.downsample,) * 2, name="maxpool2d_skip"))
            nutils.add_activation(self.activation, self.layers_skip, i="skip")
            if self.batch_norm:
                self.layers_skip.append(keras.layers.BatchNormalization(name=f"batch_norm_skip"))


class SpatialFourierCNNSkip(SpatialFourierCNN):
    """Spatial Fourier CNN with an additional Fourier skip connection"""
    def call(self, inputs, training=False):
        skip = tf.identity(inputs)
        fourier = tf.identity(inputs)
        for layer in self.layers_:
            inputs = layer(inputs, training=training)
        for layer in self.layers_skip:
            skip = layer(skip, training=training)
        for layer in self.layers_fourier:
            fourier = layer(fourier, training=training)
        inputs = inputs + fourier + skip  # no activation/norm after sum: ResNet V2
        return inputs

    def _init_layers(self):
        # STANDARD RESNET CONVOLUTIONS IN REAL/DISCRETE DOMAIN
        for i in range(len(self.filters)):  # standard ResNet convolutions
            self.layers_.append(keras.layers.Conv2D(
                filters=self.filters[i], kernel_size=self.kernels[i],
                strides=(self.downsample,) * 2 if i == 0 else (1, 1),
                padding="same", kernel_initializer="he_normal", name=f"conv2d_{i + 1}"))
            nutils.add_activation(self.activation, self.layers_, i=i)
            if self.batch_norm:
                self.layers_.append(keras.layers.BatchNormalization(name=f"batch_norm_{i + 1}"))

        # STANDARD SKIP CONNECTION
        if self.downsample > 1:  # do nothing if there no downsample
            self.layers_skip.append(keras.layers.Conv2D(
                filters=self.filters[-1], kernel_size=1, padding="same",
                kernel_initializer="he_normal", name=f"conv2d_skip"))
            self.layers_skip.append(keras.layers.MaxPool2D(  # use maxpool instead of stride ^
                pool_size=(self.downsample,) * 2, name="maxpool2d_skip"))
            nutils.add_activation(self.activation, self.layers_skip, i="skip")
            if self.batch_norm:
                self.layers_skip.append(keras.layers.BatchNormalization(name=f"batch_norm_skip"))

        # FOURIER SKIP CONNECTION
        if self.dct:  # fourier skip connection
            self.layers_fourier.append(DCT(name="dct_fourier"))
            self.layers_fourier.append(keras.layers.Conv2D(
                filters=self.filters[-1], kernel_size=1,
                padding="same", kernel_initializer="he_normal", name=f"conv2d_fourier"))
            if self.downsample > 1:  # downsample
                self.layers_fourier.append(keras.layers.MaxPool2D(
                    pool_size=(self.downsample,) * 2, name="maxpool2d_fourier"))
            nutils.add_activation(self.activation, self.layers_fourier, i="fourier")
            self.layers_fourier.append(iDCT(name="idct_fourier"))
            if self.batch_norm:  # only apply batch normalization after Fourier -> real/discrete
                self.layers_fourier.append(keras.layers.BatchNormalization(
                    name=f"batch_norm_fourier"))


class SpatialFourierCNNParallel(SpatialFourierCNN):
    """Spatial Fourier CNN with parallel Fourier and discrete/real paths, larger network"""
    def call(self, inputs, training=False):
        skip = tf.identity(inputs)
        fourier = tf.identity(inputs)
        for layer in self.layers_:
            inputs = layer(inputs, training=training)
        for layer in self.layers_skip:
            skip = layer(skip, training=training)
        for layer in self.layers_fourier:
            fourier = layer(fourier, training=training)
        inputs = inputs + fourier + skip  # no activation/norm after sum: ResNet V2
        return inputs

    def _init_layers(self):
        # STANDARD RESNET CONVOLUTIONS IN REAL/DISCRETE DOMAIN
        for i in range(len(self.filters)):  # standard ResNet convolutions
            self.layers_.append(keras.layers.Conv2D(
                filters=self.filters[i], kernel_size=self.kernels[i],
                strides=(self.downsample,) * 2 if i == 0 else (1, 1),
                padding="same", kernel_initializer="he_normal", name=f"conv2d_{i + 1}"))
            nutils.add_activation(self.activation, self.layers_, i=i)
            if self.batch_norm:
                self.layers_.append(keras.layers.BatchNormalization(name=f"batch_norm_{i + 1}"))

        # STANDARD RESNET CONVOLUTIONS IN FOURIER DOMAIN
        self.layers_fourier.append(DCT(name=f"dct_0"))
        for i in range(len(self.filters)):  # standard ResNet convolutions
            self.layers_fourier.append(keras.layers.Conv2D(
                filters=self.filters[i], kernel_size=self.kernels[i],
                strides=(self.downsample,) * 2 if i == 0 else (1, 1),
                padding="same", kernel_initializer="he_normal", name=f"conv2d_{i + 1}"))
            nutils.add_activation(self.activation, self.layers_fourier, i=i)
        self.layers_fourier.append(iDCT(name=f"idct_{i + 2}"))
        if self.batch_norm:  # only apply batch normalization after Fourier -> real/discrete
            self.layers_fourier.append(keras.layers.BatchNormalization(name=f"batch_norm_{i + 2}"))

        # STANDARD SKIP CONNECTION
        if self.downsample > 1:  # skip connection
            self.layers_skip.append(keras.layers.Conv2D(
                filters=self.filters[-1], kernel_size=1,  # strides=(self.downsample,) * 2,
                padding="same", kernel_initializer="he_normal", name=f"conv2d_skip"))
            self.layers_skip.append(keras.layers.MaxPool2D(
                pool_size=(self.downsample,) * 2, name="maxpool2d_skip"))
            nutils.add_activation(self.activation, self.layers_skip, i="skip")
            if self.batch_norm:
                self.layers_skip.append(keras.layers.BatchNormalization(name=f"batch_norm_skip"))


class SpatialFourierCNNGroup(keras.layers.Layer):
    """Spatial Fourier CNN group with repeat for long skip"""
    def __init__(self, filters, kernels, repeat, downsample=1, activation="relu", batch_norm=True,
                 long_skip=True, dct=True, type="fourier_resnet", name="spaf_cnn_group"):
        super(SpatialFourierCNNGroup, self).__init__()
        self.filters = filters
        self.kernels = kernels
        self.repeat = repeat
        self.downsample = downsample
        self.activation = activation
        self.batch_norm = batch_norm
        self.long_skip = long_skip
        self.dct = dct
        self.type = type
        self._name = name

        self.layers_ = []
        if self.long_skip:  # long skip
            self.layers_skip = []

        self._init_layers()

    def call(self, inputs, training=False):
        if self.long_skip:
            skip = tf.identity(inputs)
        for layer in self.layers_:
            inputs = layer(inputs, training=training)
        if self.long_skip:
            for layer in self.layers_skip:
                skip = layer(skip, training=training)
            inputs = inputs + skip
        return inputs

    def _init_layers(self):
        if self.type == "fourier_resnet":
            block = SpatialFourierCNN
        elif self.type == "fourier_skip":
            block = SpatialFourierCNNSkip
        elif self.type == "fourier_parallel":
            block = SpatialFourierCNNParallel

        # RESNET BLOCKS (REPEAT)
        for r in range(self.repeat):
            self.layers_.append(block(
                filters=self.filters, kernels=self.kernels,
                downsample=self.downsample if r == 0 else 1, activation=self.activation,
                batch_norm=self.batch_norm, dct=self.dct[r], name=f"spaf_cnn_{r + 1}"))

        # LONG SKIP CONNECTION
        if self.long_skip and self.downsample > 1:
            self.layers_skip.append(keras.layers.Conv2D(
                filters=self.filters[-1], kernel_size=1,  # strides=(self.downsample,) * 2,
                padding="same", kernel_initializer="he_normal", name=f"conv2d_skip"))
            self.layers_skip.append(keras.layers.MaxPool2D(
                pool_size=(self.downsample,) * 2, name="maxpool2d_skip"))
            nutils.add_activation(self.activation, self.layers_skip, i="skip")
            if self.batch_norm:
                self.layers_skip.append(keras.layers.BatchNormalization(name=f"batch_norm_skip"))
