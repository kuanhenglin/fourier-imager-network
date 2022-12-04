import time

import tensorflow as tf
from tqdm import tqdm


import utilities.utilities as utils
import utilities.data as dutils
import utilities.optimizer as outils


def train(network, data, criterion, optimizer, weight_decay,
          iter_max, batch, batch_test, writer=None):
    variables = network.variables()
    gradients = [tf.Variable(tf.zeros_like(variable)) for variable in variables]

    @tf.function
    def forward(inputs, labels):  # forward propagation with @tf.function
        with tf.GradientTape() as tape:
            outputs = network(inputs, training=True)
            loss = criterion(labels, outputs)
        gradients_ = tape.gradient(loss, variables)
        for i in range(len(gradients)):
            gradients[i].assign(gradients_[i])
        return outputs, loss

    train_data, valid_data, test_data = data

    t = time.time()
    for i in tqdm(range(iter_max)):
        inputs, labels = dutils.sample_data(train_data, sample=batch)
        inputs = dutils.augment_data(inputs, pad=4)
        labels_one_hot = utils.one_hot(labels, depth=network.out_shape[-1])
        _, _ = forward(inputs, labels_one_hot)
        outils.add_weight_decay(zip(gradients, variables), weight_decay=weight_decay)
        optimizer.apply_gradients(zip(gradients, variables))

        if (i + 1) % (iter_max // 100) == 0:
            delta_t = time.time() - t  # do not count evaluation as part of time
            _, (train_loss, train_accuracy) =\
                utils.test(network, train_data, criterion, batch=batch_test)
            _, (valid_loss, valid_accuracy) =\
                utils.test(network, valid_data, criterion, batch=batch_test)
            # convert to native Python float
            train_loss, train_accuracy = float(train_loss.numpy()), float(train_accuracy.numpy())
            valid_loss, valid_accuracy = float(valid_loss.numpy()), float(valid_accuracy.numpy())

            tqdm.write(
                f"{('[' + str(i + 1) + ']'):8s}   "
                f"Train: {str(train_accuracy * 100):.5}% ({str(train_loss):.5})   "
                f"Validation: {str(valid_accuracy * 100):.5}% ({str(valid_loss):.5})")

            if len(writer.data["time_cumulative"]) == 0:
                t_cumulative = delta_t
            else:
                t_cumulative = writer.data["time_cumulative"][-1] + delta_t
            if writer is not None:
                writer.add(dict(
                    train_accuracy=train_accuracy, train_loss=train_loss,
                    valid_accuracy=valid_accuracy, valid_loss=valid_loss,
                    iteration=i + 1, time_cumulative=t_cumulative))
            t = time.time()

    _, (test_loss, test_accuracy) = utils.test(network, test_data, criterion, batch=batch_test)
    test_loss, test_accuracy = float(test_loss.numpy()), float(test_accuracy.numpy())
    tqdm.write(
        f"{'[Final]':8s}   Test: {str(test_accuracy * 100):.5}% ({str(test_loss):.5})")
    if writer is not None:
        writer.add(dict(test_accuracy=test_accuracy, test_loss=test_loss))
