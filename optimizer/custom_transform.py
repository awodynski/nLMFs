import tensorflow as tf

class SignedLogTransform(tf.keras.layers.Layer):
    """
    A custom Layer applying a signed logarithm transform to the inputs.

    For positive values x, the transform is log(x + 1).
    For non-positive values, the transform is -log(-x + 1).

    This is useful when data can take both positive and negative values,
    preserving the sign information while applying a log-scaling.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor to transform.

        Returns
        -------
        tf.Tensor
            Transformed tensor of the same shape as `inputs`.
        """
        return tf.where(
            inputs > 0,
            tf.math.log(inputs + 1.0),
            -tf.math.log(-inputs + 1.0)
        )


class AbsLogTransform(tf.keras.layers.Layer):
    """
    A custom Layer applying a logarithm transform to the absolute value of the inputs.

    The transform is log(|x| + 1e-4), thus avoiding log(0) issues and
    ensuring a smooth gradient even near zero.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor to transform.

        Returns
        -------
        tf.Tensor
            Transformed tensor of the same shape as `inputs`.
        """
        return tf.math.log(tf.math.abs(inputs) + 0.0001)



class NoneTransform(tf.keras.layers.Layer):
    """
    A pass-through Layer that applies no transformation to the inputs.

    Useful if you want to optionally deactivate a transform
    while keeping a consistent interface in your pipeline.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor.

        Returns
        -------
        tf.Tensor
            The same input tensor, unmodified.
        """
        return inputs

