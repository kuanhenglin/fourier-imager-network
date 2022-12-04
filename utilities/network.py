from tensorflow import keras


def add_activation(name, layers, i=0):
    if type(i) == int:
        index = str(i + 1)
    elif type(i) == str:
        index = i
    if name == "relu":
        layers.append(keras.layers.ReLU(name=f"relu_{index}"))
    elif name == "leaky_relu":
        layers.append(keras.layers.LeakyReLU(name=f"leaky_relu_{index}"))
    elif name == "softmax":
        layers.append(keras.layers.Softmax(name=f"softmax_{index}"))
    elif name == "sigmoid":
        layers.append(keras.activations.sigmoid)