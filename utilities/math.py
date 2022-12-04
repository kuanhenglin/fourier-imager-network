from tensorflow import signal, linalg


def dct(inputs, type=2, norm="ortho", **kwargs):
    """
    2D discrete cosine transform, using tf.signal.dct

    Parameters
    ----------
    inputs : Tensor
        2D input, assume in format (..., H, W), e.g., (B, C, H, W).
        Note that this is different from the TensorFlow default (B, H, W, C).
    norm : str OR None
        Normalization method, None for no normalization.

    Returns
    -------
    Tensor
        2D DCT applied to last two dimension of ``inputs``.
    """
    inputs = signal.dct(inputs, type=type, norm=norm)  # apply DCT to last dimension
    inputs = linalg.matrix_transpose(inputs)  # transpose to expose second-last dimension
    inputs = signal.dct(inputs, type=type, norm=norm)  # apply DCT to second-last dimension
    inputs = linalg.matrix_transpose(inputs)  # transpose back to original image
    return inputs


def idct(inputs, type=2, norm="ortho", **kwargs):
    """
    2D inverse discrete cosine transform, using tf.signal.idct

    Parameters
    ----------
    inputs : Tensor
        2D input, assume in format (..., H, W), e.g., (B, C, H, W).
        Note that this is different from the TensorFlow default (B, H, W, C).
    norm : str OR None
        Normalization method, None for no normalization.

    Returns
    -------
    Tensor
        2D inverse DCT applied to last two dimension of ``inputs``.
    """
    inputs = signal.idct(inputs, type=type, norm=norm)  # apply inverse DCT to last dimension
    inputs = linalg.matrix_transpose(inputs)  # transpose to expose second-last dimension
    inputs = signal.idct(inputs, type=type, norm=norm)  # apply inverse DCT to second-last dimension
    inputs = linalg.matrix_transpose(inputs)  # transpose back to original image
    return inputs