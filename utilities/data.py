import numpy as np
from numpy import random
import tensorflow as tf
from tensorflow import keras


def sample_data(data, sample=None):
    inputs, labels = data
    indices = np.random.permutation(inputs.shape[0])

    if type(sample) == int:  # sample represents number of samples
        indices = indices[:sample]
    elif type(sample) == float:  # sample represents fraction of total data
        indices = indices[:round(inputs.size(0) * sample)]

    if tf.is_tensor(inputs):
        return (tf.gather(inputs, indices), tf.gather(labels, indices))
    else:  # assume numpy array or numpy-indexing-compatible array type
        return (inputs[indices], labels[indices])


def get_train_valid_split(data, train_valid_split=[9, 1], shuffle=True):
    split_index = round(data[0].shape[0] / sum(train_valid_split) * train_valid_split[0])
    if shuffle:
        data = sample_data(data, sample=None)
    train_data = (data[0][:split_index], data[1][:split_index])
    valid_data = (data[0][split_index:], data[1][split_index:])
    return train_data, valid_data


def augment_data(inputs, pad=0, crop=None, flip=(True, False)):  # flip=(horizontal, vertical)
    crop = inputs.shape[1] if crop is None else crop  # None defaults to same-size crop
    if inputs.shape[1] + pad > crop:  # random crop
        inputs_pad = np.pad(
            inputs, mode="constant", constant_values=(0., 0.),  # pad zeros
            pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)))
        crop_rand = random.randint(low=0, high=2 * pad + 1 + (inputs.shape[1] - crop), size=2)
        inputs = inputs_pad[
            :, crop_rand[0]:crop_rand[0] + crop, crop_rand[1]:crop_rand[1] + crop, :]

    if flip[0] or flip[1]:  # random flip
        flip_rand = random.randint(low=0, high=1, size=2)
        if flip[0] and flip_rand[0]:
            inputs = inputs[:, :, ::-1, :]
        if flip[1] and flip_rand[1]:
            inputs = inputs[:, ::-1, :, :]
        return inputs

    return inputs