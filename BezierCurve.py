import keras.layers
import tensorflow as tf
import numpy as np
from math import comb
from itertools import product
from typing import Union
from matplotlib import pyplot as plt


class BazierCurve(object):
    def __init__(self, anchor_points: tf.float32, data_size: int):
        self.parameter = None
        self.data_size = data_size
        self.anchor_points = tf.constant(anchor_points)
        self.order, self.dimensions = self.anchor_points.shape
        self.parameter = tf.linspace(0.0, 1.0, self.data_size)

    @property
    def data_points(self):
        return BazierCurve.imprecise_data_points(
            self.parameter_power_series, self.constant, self.anchor_points)

    @property
    def constant(self) -> tf.float32:
        return self.constant_matrix(self.order)

    @property
    def constant_inv(self) -> tf.float32:
        return tf.linalg.inv(self.constant)

    @property
    def parameter_power_series(self):
        return BazierCurve.power_series_matrix(tf.expand_dims(self.parameter, axis=1), self.order)

    def plot(self, comment: str = ""):
        if self.dimensions == 2:
            data = self.data_points
            x = data[:, 0]
            y = data[:, 1]
            ax = self.anchor_points[:, 0]
            ay = self.anchor_points[:, 1]
            plt.plot(x, y, 'b')
            plt.plot(ax, ay, 'r*')
            plt.title(f"Bezier Curve - {comment}")
            plt.grid()
            plt.show()
        else:
            raise ValueError(f"plots in dimension {self.dimensions} is not implemented")

    def data_point_at(self, parameter: float) -> tf.float32:
        bc = BazierCurve(self.anchor_points, data_size=1)
        bc.parameter = tf.constant([0.0, parameter, 1.0])
        return bc.data_points[1, :]

    @staticmethod
    @tf.function
    def constant_matrix(order: int, squeeze: bool = True) -> tf.float32:
        """
        The Bezier curve constant matrix of oxo.
        where o is curve order ( number of Anchor points )
        B_mxd = T_mxo C_oxo A_oxd
        :param squeeze:
        :param order: curve order ( number of Anchor points )
        :return: The Bezier curve constant matrix of oxo
        """
        c = np.zeros((order, order))
        for i, j in product(range(order), repeat=2):
            c[i, j] = comb(i, j) * comb(order - 1, i) * (-1) ** (i + j) if i >= j else 0
        c = tf.constant(c)
        c = tf.expand_dims(c, axis=2)
        # c = tf.image.flip_left_right(c)
        c = tf.squeeze(c)
        return tf.expand_dims(c, axis=0) if not squeeze else c

    @staticmethod
    @tf.function
    def power_series_matrix(parameters: tf.float32, power: int) -> tf.float32:
        powers = [tf.pow(parameters, _) for _ in range(power)]
        return tf.concat(powers, axis=1)

    @staticmethod
    @tf.function
    def matmul_triple(tensor_a: tf.float32, tensor_b: tf.float32, tensor_c: tf.float32):
        tensor_a = tf.cast(tensor_a, tf.float32)
        tensor_b = tf.cast(tensor_b, tf.float32)
        tensor_c = tf.cast(tensor_c, tf.float32)
        ab = tf.cast(tensor_a @ tensor_b, tf.float32)
        return tf.cast(ab @ tensor_c, tf.float32)

    @staticmethod
    @tf.function
    def imprecise_parameter_power_series_matrix(
            bc_data_points: tf.float32,
            bc_anchor_points: tf.float32,
            bc_constant_matrix: tf.float32) -> tf.float32:
        bc_constant_matrix_inv = tf.linalg.inv(bc_constant_matrix)
        bc_anchor_points_inv = tf.linalg.pinv(bc_anchor_points)
        return BazierCurve.matmul_triple(bc_data_points, bc_anchor_points_inv, bc_constant_matrix_inv)

    @staticmethod
    @tf.function
    def imprecise_data_points(
            bc_parameter_power_series: tf.float32,
            bc_constant_matrix: tf.float32,
            bc_anchor_points: tf.float32) -> tf.float32:
        return BazierCurve.matmul_triple(bc_parameter_power_series, bc_constant_matrix, bc_anchor_points)

    @staticmethod
    @tf.function
    def imprecise_anchor_points(
            bc_data_points: tf.float32,
            bc_constant_matrix: tf.float32,
            bc_parameter_power_series_matrix: tf.float32) -> tf.float32:
        bc_parameter_power_series_matrix_inv = tf.linalg.pinv(bc_parameter_power_series_matrix)
        bc_constant_matrix_inv = tf.linalg.inv(bc_constant_matrix)
        return BazierCurve.matmul_triple(bc_constant_matrix_inv, bc_parameter_power_series_matrix_inv, bc_data_points)


class ParameterPowerSeriesLayer(keras.layers.Layer):
    def __init__(self):
        super(ParameterPowerSeriesLayer, self).__init__()
        self.order = None
        self.parameters = None

    def build(self, input_shape):
        self.order = input_shape[1]
        self.parameters = tf.Variable(
            shape=(input_shape[0], 1),
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.05, maxval=0.995
            ),
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0
            ),
            trainable=True,
        )

    def call(self, inputs, *args, **kwargs):
        super(ParameterPowerSeriesLayer, self).call(*args, **kwargs)
        output = BazierCurve.power_series_matrix(self.parameters, self.order)
        self.add_loss(tf.keras.losses.MeanAbsoluteError(inputs - output))
        return output


class ImpreciseParameterPowerSeriesLayer(keras.layers.Layer):
    def __init__(self):
        super(ImpreciseParameterPowerSeriesLayer, self).__init__()

    @tf.function
    def call(self, inputs, *args, **kwargs):
        super(ImpreciseParameterPowerSeriesLayer, self).call(*args, **kwargs)
        B, A, C = inputs
        return BazierCurve.imprecise_parameter_power_series_matrix(B, A, C)


class ImpreciseDataPointslayer(keras.layers.Layer):
    def __init__(self):
        super(ImpreciseDataPointslayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        super(ImpreciseDataPointslayer, self).call(*args, **kwargs)
        T, C, A = inputs
        return BazierCurve.imprecise_data_points(T, A, C)


def bazier_curve_fitting(data_points: tf.float32, order: int):
    assert (len(data_points.shape) == 2)
    data_size, dimensions = data_points.shape
    b_in = keras.Input(shape=data_points.shape, batch_size=1)
    b_in = keras.layers.Lambda(tf.squeeze)(b_in)
    ppsl = ParameterPowerSeriesLayer()

    anchors = tf.Variable(
        trainable=True,
        initial_value=tf.random.uniform(shape=(order, dimensions)),
        dtype=tf.float32
    )
    bc_constant = BazierCurve.constant_matrix(order)
    bazier_curve_parameter_power_series_apriori = ImpreciseParameterPowerSeriesLayer()(b_in, anchors, bc_constant)
    bazier_curve_parameter_power_series_posteriori = ppsl(bazier_curve_parameter_power_series_apriori)
    b_hat = ImpreciseDataPointslayer()(
        bazier_curve_parameter_power_series_posteriori,
        bc_constant,
        anchors
    )
    model = keras.Model(b_in, b_hat)
    model.compile(optimizer='adam', loss='mse')
    model.summary()


o = 4
dim = 2
a = tf.random.uniform((o, dim))
bc = BazierCurve(a, 40)

print(f"t=0\t{bc.data_point_at(0.0)}")
print(f"t=1\t{bc.data_point_at(1.0)}")
print(bc.anchor_points)
print(bc.constant)


bazier_curve_fitting(bc.data_points, 4)
bc.plot()
