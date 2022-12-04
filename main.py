"""
Fourier Imager Network for Image Classification

Benchmark program written by Jordan Lin, for Fall 2022 ECE 194 interm report.
"""

import os
import argparse

import tensorflow as tf
from tensorflow import keras, math

from networks.fin import FourierImagerNetwork
import train
import utilities.data as dutils
import utilities.writer as wutils


def set_tensorflow():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress tensorflow warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # also ^


def set_seed(seed=0):
    keras.utils.set_random_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser(
        prog="Fourier Imager Network for Image Classification",
        description="Benchmark program written by Jordan Lin, Fall 2022.")
    parser.add_argument("-e", "--seed", default=0, type=int)
    parser.add_argument("-d", "--data", default="cifar10", type=str)
    parser.add_argument("-n", "--network", default="resnet", type=str)
    parser.add_argument("-o", "--optimizer", default="nsgd", type=str)
    parser.add_argument("-t", "--type", default="fourier_resnet", type=str)
    parser.add_argument("-f", "--fourier", nargs="*", default=[1, 1, 1], type=int)
    parser.add_argument("-l", "--learning_rate", default=0.1, type=float)
    parser.add_argument("-b", "--batch", default=128, type=int)
    parser.add_argument("-bt", "--batch_test", default=4096, type=int)
    parser.add_argument("-s", "--long_skip", default=True, type=int)
    parser.add_argument("-i", "--iter_max", default=64000, type=int)
    parser.add_argument("-w", "--weight_decay", default=0.0001, type=float)
    parser.add_argument("-p", "--path", default="logs", type=str)
    parser.add_argument("-r", "--resnet_repeat", default=3, type=int)
    arguments = parser.parse_args()

    return arguments


def get_architecture(network="resnet", resnet_repeat=3, spaf_type="fourier_resnet",
                     dct=[1, 1, 1], long_skip=True):
    architecture_resnet = dict(
        pre=dict(filters=[16], kernels=[3], downsample=[1], activation="relu", batch_norm=True),
        spaf=dict(
            filters=[[16, 16], [32, 32], [64, 64]], kernels=[[3, 3], [3, 3], [3, 3]],
            repeat=[resnet_repeat] * 3, downsample=[1, 2, 2],
            activation="relu", batch_norm=True, long_skip=long_skip, dct=dct, type=spaf_type),
        post=dict(filters=[], kernels=[], downsample=[], activation="relu", batch_norm=True),
        global_average_pooling=True, activation_out=None)
    architecture_cnn = dict(
        pre=dict(filters=[], kernels=[], downsample=[], activation="relu", batch_norm=True),
        spaf=dict(
            filters=[[16], [32], [32], [32], [64], [64]], kernels=[[5], [3], [3], [3], [3], [3]],
            repeat=[1, 1, 1, 1, 1, 1], downsample=[1, 2, 1, 2, 1, 2],
            activation="relu", batch_norm=True, long_skip=long_skip, dct=dct, type=spaf_type),
        post=dict(filters=[], kernels=[], downsample=[], activation="relu", batch_norm=True),
        global_average_pooling=False, activation_out=None)
    if network == "resnet":
        return architecture_resnet
    elif network == "cnn":
        return architecture_cnn


def get_data(name="cifar10", train_valid_split=[9, 1], center=True):
    if name == "cifar10":
        (train_inputs, train_labels), (test_inputs, test_labels) =\
            keras.datasets.cifar10.load_data()
        (train_inputs, train_labels), (valid_inputs, valid_labels) = dutils.get_train_valid_split(
            (train_inputs, train_labels), train_valid_split=train_valid_split, shuffle=True)
        train_inputs = tf.cast(train_inputs, dtype=tf.float32) / 255
        valid_inputs = tf.cast(valid_inputs, dtype=tf.float32) / 255
        test_inputs = tf.cast(test_inputs, dtype=tf.float32) / 255
        if center:
            train_mean = math.reduce_mean(train_inputs, axis=0)
            train_inputs -= train_mean
            valid_inputs -= train_mean
            test_inputs -= train_mean
        train_data = (train_inputs, train_labels)
        valid_data = (valid_inputs, valid_labels)
        test_data = (test_inputs, test_labels)
    else:
        raise ValueError(f"Data \"{name}\" currently unsupported.")
    return train_data, valid_data, test_data


def get_network(architecture, info=True):
    network = FourierImagerNetwork(architecture, in_shape=[32, 32, 3], out_shape=[10])
    if info:
        network.info(width=100)
    return network


def get_criterion(name="cross_entropy"):
    reduction = "auto"
    if name == "mse":
        criterion = keras.losses.MeanSquaredError(reduction=reduction)
    elif name == "cross_entropy":
        criterion = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=reduction)
    return criterion


def get_optimizer(learning_rate_start=0.1, iter_max=64000, name="nsgd"):
    learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[iter_max // 2, iter_max // 4 * 3],
        values=[learning_rate_start, learning_rate_start / 10, learning_rate_start / 100])
    if name == "nsgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif name == "nadam":
        optimizer = keras.optimizers.experimental.Nadam(learning_rate=learning_rate_start)
    return optimizer


def get_writer(categories, arguments):
    writer = wutils.Writer(categories=categories, hyperparameters=vars(arguments))
    return writer


def main():
    print("")

    arguments = get_arguments()
    set_tensorflow()
    set_seed(arguments.seed)
    writer = get_writer(
        categories=[
            "train_accuracy", "train_loss", "valid_accuracy", "valid_loss",
            "test_accuracy", "test_loss", "iteration", "time_cumulative"],
        arguments=arguments)
    print(f"Hyperparameters:", writer.hyperparameters)

    data = get_data(name=arguments.data)
    architecture = get_architecture(
        network=arguments.network, spaf_type=arguments.type,
        long_skip=arguments.long_skip, dct=arguments.fourier)
    network = get_network(architecture, info=True)

    criterion = get_criterion()
    optimizer = get_optimizer(
        learning_rate_start=arguments.learning_rate, iter_max=arguments.iter_max,
        name=arguments.optimizer)

    train.train(
        network, data, criterion, optimizer, weight_decay=arguments.weight_decay,
        iter_max=arguments.iter_max, batch=arguments.batch, batch_test=arguments.batch_test,
        writer=writer)
    writer.save(folder=arguments.path, save_pickle=True, save_json=True)


if __name__ == "__main__":
    main()
