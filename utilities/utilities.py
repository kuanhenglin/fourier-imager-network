import numpy as np
import tensorflow as tf
from tensorflow import math


import utilities.data as dutils


def one_hot(labels, depth):
    labels = tf.one_hot(tf.squeeze(labels, axis=1), depth=depth)
    return labels


def get_accuracy(labels, outputs):
    predictions = tf.cast(math.argmax(outputs, axis=1), dtype=tf.uint8)
    labels = tf.squeeze(labels, axis=1)
    accuracy = math.reduce_mean(tf.cast(predictions == labels, dtype=tf.float32))
    return accuracy


def evaluate(network, data, criterion):
    inputs, labels = data
    outputs = network(inputs)
    loss = criterion(labels, outputs)
    accuracy = get_accuracy(labels, outputs)
    return loss, accuracy


def test(network, data, criterion, batch=None, shuffle=True):
    if shuffle:
        data = dutils.sample_data(data, sample=None)  # shuffle data
    inputs, labels = data
    labels_one_hot = one_hot(labels, depth=network.out_shape[-1])
    if batch is None:
        batch = inputs.shape[0]
    outputs = []
    loss = 0

    i = 0
    while i < inputs.shape[0]:
        inputs_ = inputs[i:i + batch]
        labels_ = labels_one_hot[i:i + batch]
        outputs_ = network(inputs_)
        loss_ = criterion(labels_, outputs_)

        outputs.append(outputs_)
        loss += (inputs_.shape[0] / inputs.shape[0]) * loss_
        i += batch

    outputs = tf.concat(outputs, axis=0)
    accuracy = get_accuracy(labels, outputs)

    return outputs, (loss, accuracy)