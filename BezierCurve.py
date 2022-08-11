from abc import ABC

import keras.layers
import tensorflow as tf
import numpy as np
from math import comb
from itertools import product
from typing import Union
from matplotlib import pyplot as plt
import os
import shutil
import glob


def ClearFolder(path: str):
    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f):
            print(f'removing {f}')
            os.remove(f)


class BazierCurve(object):
    def __init__(self, anchor_points: tf.float64, data_size: int):
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
    def constant(self) -> tf.float64:
        return self.constant_matrix(self.order)

    @property
    def constant_inv(self) -> tf.float64:
        return tf.linalg.inv(self.constant)

    @property
    def parameter_power_series(self):
        return BazierCurve.power_series_matrix(tf.expand_dims(self.parameter, axis=1), self.order)

    def plot(self, comment: str = "", save_plot: bool = False):
        if self.dimensions == 2:
            data = self.data_points
            x = data[:, 0]
            y = data[:, 1]
            ax = self.anchor_points[:, 0]
            ay = self.anchor_points[:, 1]
            plt.plot(x, y, 'b.')
            plt.plot(ax, ay, 'r*')
            plt.title(f"Bezier Curve - {comment}")
            plt.grid()
            if save_plot:
                plt.savefig(f'images/{comment}.png')
            else:
                plt.show()
            plt.close()
        else:
            raise ValueError(f"plots in dimension {self.dimensions} is not implemented")

    def data_point_at(self, parameter: float) -> tf.float64:
        bc = BazierCurve(self.anchor_points, data_size=1)
        bc.parameter = tf.constant([0.0, parameter, 1.0])
        return bc.data_points[1, :]

    @staticmethod
    @tf.function
    def constant_matrix(order: int, squeeze: bool = True) -> tf.float64:
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
    def power_series_matrix(parameters: tf.float64, power: int) -> tf.float64:
        powers = [tf.pow(parameters, _) for _ in range(power)]
        return tf.concat(powers, axis=1)

    @staticmethod
    @tf.function
    def matmul_triple(tensor_a: tf.float64, tensor_b: tf.float64, tensor_c: tf.float64):
        tensor_a = tf.cast(tensor_a, tf.float64)
        tensor_b = tf.cast(tensor_b, tf.float64)
        tensor_c = tf.cast(tensor_c, tf.float64)
        ab = tf.cast(tensor_a @ tensor_b, tf.float64)
        return tf.cast(ab @ tensor_c, tf.float64)

    @staticmethod
    @tf.function
    def imprecise_parameter_power_series_matrix(
            bc_data_points: tf.float64,
            bc_anchor_points: tf.float64,
            bc_constant_matrix: tf.float64) -> tf.float64:
        bc_constant_matrix_inv = tf.linalg.inv(bc_constant_matrix)
        bc_anchor_points_inv = tf.linalg.pinv(bc_anchor_points)
        return BazierCurve.matmul_triple(bc_data_points, bc_anchor_points_inv, bc_constant_matrix_inv)

    @staticmethod
    @tf.function
    def imprecise_data_points(
            bc_parameter_power_series: tf.float64,
            bc_constant_matrix: tf.float64,
            bc_anchor_points: tf.float64) -> tf.float64:
        return BazierCurve.matmul_triple(bc_parameter_power_series, bc_constant_matrix, bc_anchor_points)

    @staticmethod
    @tf.function
    def imprecise_anchor_points(
            bc_data_points: tf.float64,
            bc_constant_matrix: tf.float64,
            bc_parameter_power_series_matrix: tf.float64) -> tf.float64:
        bc_parameter_power_series_matrix_inv = tf.linalg.pinv(bc_parameter_power_series_matrix)
        bc_constant_matrix_inv = tf.linalg.inv(bc_constant_matrix)
        return BazierCurve.matmul_triple(bc_constant_matrix_inv, bc_parameter_power_series_matrix_inv, bc_data_points)

    @staticmethod
    @tf.function
    def points_prime(points: tf.float64):
        shiftted = tf.roll(input=points, shift=-1, axis=0)
        return tf.subtract(points, shiftted)

    @staticmethod
    @tf.function
    def cross_product_norm(a: tf.float64, b: tf.float64):
        dot = tf.tensordot(a, b, axes=(1, 1))
        cos = dot / (tf.norm(a) * tf.norm(b))
        cot = cos / tf.math.sqrt(1.0 + tf.math.pow(cos, 2))
        return dot * cot

    @staticmethod
    @tf.function
    def points_curvature(points: tf.float64) -> tf.float64:
        """
        c = | r' X r |'' / |r'^3|
        :param points:
        :return:
        """
        r = points
        rp = BazierCurve.points_prime(points)
        rpp = BazierCurve.points_prime(rp)
        rp_rpp = BazierCurve.cross_product_norm(rp, rpp)
        rp_rpp = tf.norm(rp_rpp)
        rp3 = tf.norm(tf.pow(rp, 3))
        return tf.cast(rp_rpp / rp3, tf.float64)


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


@tf.function
def choice(inputs: tf.float64, sample_num: int):
    idxs = tf.range(tf.shape(inputs)[0])
    ridxs = tf.random.shuffle(idxs)[:sample_num]
    return tf.gather(inputs, ridxs)


class BazierCurveFitting(keras.Model):
    def __init__(self, order: int, datapoints: tf.float64):
        self.anchors = None
        self.parameters = None
        self.variable_anchors = None
        self.dimensions = None
        self.data_size = None
        self.order = order
        start_point = datapoints[0, :]
        end_point = datapoints[-1, :]
        self.start_point = tf.expand_dims(tf.constant(start_point, dtype=tf.float64), axis=0)
        self.end_point = tf.expand_dims(tf.constant(end_point, dtype=tf.float64), axis=0)
        self.constant = BazierCurve.constant_matrix(self.order)
        self.scale = tf.reduce_max(self.end_point)
        print(choice(datapoints, order - 2))
        paddedzeros = tf.zeros_like(self.start_point)
        self.random_inital_anchors = tf.concat([paddedzeros, choice(datapoints, order - 2), paddedzeros], axis=0)
        super(BazierCurveFitting, self).__init__()

    def build(self, input_shape):
        self.data_size, self.dimensions = input_shape
        self.variable_anchors = self.add_weight(
            shape=(self.order - 2, self.dimensions),
            dtype=tf.float64,
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.005 * self.scale, maxval=0.005 * self.scale
            ),
            trainable=True
        )

        self.parameters =
            '''self.add_weight(
            shape=(self.data_size, 1),
            dtype=tf.float64,
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.0, maxval=1.0
            ),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0
            ),'''
        )

    def call(self, inputs: tf.float64, training=True, mask=None):
        self.anchors = tf.concat([self.start_point, self.variable_anchors, self.end_point], axis=0)
        self.anchors = self.anchors + self.random_inital_anchors
        T = BazierCurve.power_series_matrix(self.parameters, self.order)
        TC = tf.matmul(T, self.constant)
        TCA = tf.matmul(TC, self.anchors)
        inputs = tf.cast(inputs, tf.float64)
        # output = tf.abs(tf.subtract(TCA, inputs))
        self.add_loss(tf.keras.losses.MeanAbsoluteError()(inputs, TCA))
        self.add_loss(curvature_cost_function(inputs, TCA))
        return tf.expand_dims(TCA, axis=0)

    @property
    def current_anchor_points(self):
        var_a = self.get_weights()[0]
        return tf.concat([self.start_point, var_a, self.end_point], axis=0) + self.random_inital_anchors


@tf.function
def curvature_cost_function(y_true: tf.float64, y_pred: tf.float64) -> tf.float64:
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)
    curvature_ture = BazierCurve.points_curvature(y_true)
    curvature_pred = BazierCurve.points_curvature(y_pred)
    if tf.math.is_nan(curvature_pred) and tf.math.is_nan(curvature_ture):
        return tf.constant(0, dtype=tf.float64)
    elif tf.math.is_nan(curvature_pred) or tf.math.is_nan(curvature_ture):
        return tf.constant(1000, dtype=tf.float64)
    else:
        return tf.abs(tf.math.tanh(curvature_ture - curvature_pred))


@tf.function
def curvature_accuracy(y_true: tf.float64, y_pred: tf.float64) -> tf.float64:
    return 100.0 * (1.0 - curvature_cost_function(y_true, y_pred))


@tf.function
def fitting_accuracy(y_true: tf.float64, y_pred: tf.float64) -> tf.float64:
    loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return 100.0 * (1.0 - loss)


@tf.function
def fitting_cost_function(y_true: tf.float64, y_pred: tf.float64):
    loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_true)
    loss = tf.abs(tf.math.tanh(loss))
    curvature_loss = curvature_cost_function(y_true, y_true)
    # curvature_loss = curvature_cost_function_v2(y_true, y_pred)
    return tf.math.sqrt(tf.multiply(loss, curvature_loss))


def bazier_curve_fitting(datapoints, order):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=datapoints.shape, batch_size=1),
        keras.layers.Lambda(tf.squeeze),
        BazierCurveFitting(order, datapoints),
    ])
    model.build(datapoints.shape)
    model.summary()

    class PlotBC(keras.callbacks.Callback):
        def __init__(self):
            super(PlotBC, self).__init__()
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            A = model.layers[1].current_anchor_points
            plt.plot(datapoints[:, 0], datapoints[:, 1], 'k.')
            bc = BazierCurve(A, datapoints.shape[0])
            bc.plot(f'epoch #{epoch}', save_plot=True)

    X = tf.expand_dims(datapoints, axis=0)
    # Y = tf.zeros_like(X)

    model.compile(
        optimizer='Adam',
        loss=[
            fitting_cost_function,
        ],
        metrics=[
            curvature_accuracy,
            fitting_accuracy,
            'mean_absolute_percentage_error',
            'kullback_leibler_divergence',
        ])
    model.fit(
        x=X, y=X, batch_size=1,
        epochs=1000,
        verbose=1,
        callbacks=[
            PlotBC(),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=0,
                write_graph=True,
                update_freq='epoch',
            )
        ],
        use_multiprocessing=True
    )
    os.system('tensorboard --logdir=/home/hhj/PycharmProjects/BezierCurve/logs')
    return model.layers[1].current_anchor_points


ClearFolder('/home/hhj/PycharmProjects/BezierCurve/images')
ClearFolder('/home/hhj/PycharmProjects/BezierCurve/logs')
order = 4
dim = 2
data_size = 40
a = tf.random.uniform((order, dim))
a = tf.sort(a, axis=0)
bc = BazierCurve(a, data_size)

print(f"t=0\t{bc.data_point_at(0.0)}")
print(f"t=1\t{bc.data_point_at(1.0)}")
print(bc.anchor_points)

an = bazier_curve_fitting(bc.data_points, order)

print(an)
