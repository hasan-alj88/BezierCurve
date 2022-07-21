import tensorflow as tf
import numpy as np
from tensorflow import keras
from math import comb
from itertools import product


def bezier_curve_constant_matrix(order: int) -> tf.float64:
    """
    The Bezier curve constant matrix of oxo.
    where o is curve order ( number of Anchor points )
    B_mxd = T_mxo C_oxo A_oxd
    :param order: curve order ( number of Anchor points )
    :return: The Bezier curve constant matrix of oxo
    """
    c = np.zeros((order, order))
    for i, j in product(range(order), repeat=2):
        c[i, j] = comb(i, j) * comb(order - 1, i) * (-1) ** (i + j) if i >= j else 0
    c = tf.constant(c.T)
    c = tf.expand_dims(c, axis=2)
    c = tf.image.flip_left_right(c)
    return tf.squeeze(c)


def power_series(t_values: tf.float64, order: int) -> tf.float64:
    # t_mx1 -> t_mxo
    powers = [tf.math.pow(t_values, n) for n in range(order)]
    return tf.stack(powers, axis=1)


def power_series_prime(t_values: tf.float64, order: int) -> tf.float64:
    # t_mx1 -> t_mxo
    powers = [n * tf.math.pow(t_values, n - 1) for n in range(order)]
    return tf.stack(powers, axis=1)


def magnitudes(vectors: tf.float64) -> tf.float64:
    assert (len(vectors.shape) == 2)
    v_sqr = tf.math.square(vectors)
    v_sqr_sum = tf.math.reduce_sum(v_sqr, axis=1)
    d = tf.math.sqrt(v_sqr_sum)
    return d


class BezierCurveInterpolationLayer(keras.layers.Layer):
    def __init__(self, anchor: tf.float64):
        super(BezierCurveInterpolationLayer, self).__init__()
        self.order = anchor.shape[0]
        # C_oxo
        self.C = bezier_curve_constant_matrix(self.order)
        # t_mx1
        self.t = None
        self.A = anchor

    def build(self, input_shape):
        assert (input_shape[1] == self.A.shape[1])
        self.t = self.add_weight(
            shape=(input_shape[0],),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float64,
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0.0,
                max_value=1.0,
            )
        )

    def __call__(self, p: tf.float64):
        """

        :param p: points to be interpolated
        :return: B_mxd = T_mxo C_oxo A_oxd
        """
        T = power_series(self.t, self.order)  # T_mxo
        TC = tf.matmul(T, self.C)  # TC_mxo = T_mxo C_oxo
        TCA = tf.matmul(TC, self.A)  # B_mxo = TCA_mxo = TC_mxo A_oxd
        p = tf.cast(p, tf.float64)
        error_vectors = TCA - p
        error_distances = magnitudes(error_vectors)
        mae_distances = tf.math.reduce_mean(error_distances)
        self.add_loss(mae_distances)
        return TCA


def bezier_curve_interpolation_model(anchors_, input_points):
    assert (len(input_points.shape) == 2)
    bcl = BezierCurveInterpolationLayer(anchors_)
    bcl.build(input_points.shape)
    x_in = tf.keras.Input(input_points.shape[1], batch_size=input_points.shape[0])
    x_out = bcl(x_in)
    return tf.keras.Model(x_in, x_out)


print(bezier_curve_constant_matrix(2))
print(bezier_curve_constant_matrix(3))

points = np.random.random((10, 2))
anchors = np.array([[-1, 1], [0, 1], [1, 0]])
model = bezier_curve_interpolation_model(anchors, points)
model.build()
model.compile(optimizer='sgd', loss='mse')
print(model.summary())
