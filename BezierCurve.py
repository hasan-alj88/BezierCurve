import keras.layers
import tensorflow as tf
import numpy as np
from math import comb
from itertools import product
from matplotlib import pyplot as plt
import os
import glob
from functools import reduce


def ClearFolder(path: str):
    files = glob.glob(path)
    for f in files:
        if os.path.isfile(f):
            print(f'removing {f}')
            os.remove(f)


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
        c = tf.cast(c, tf.float32)
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

    @staticmethod
    @tf.function
    def points_prime(points: tf.float32):
        shiftted = tf.roll(input=points, shift=1, axis=0)
        shiftted = tf.subtract(points, shiftted)
        return tf.pad(shiftted[1:], tf.constant([[1, 0], [0, 0]], dtype=tf.int32))

    @staticmethod
    @tf.function
    def goodness_of_fit(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        """
        Godness of fit = sum{ (y_true-y_pred)^2 / y_true }
        :param y_true:
        :param y_pred:
        :return:
        """
        gof = tf.subtract(y_pred, y_true)
        gof = tf.square(gof)
        gof = tf.math.divide_no_nan(gof, y_true)
        return tf.reduce_sum(gof)

    @staticmethod
    @tf.function
    def smd(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        x_bar_1 = tf.reduce_mean(y_true)
        x_bar_2 = tf.reduce_mean(y_pred)
        s1 = tf.math.reduce_variance(y_true)
        s2 = tf.math.reduce_variance(y_pred)
        return tf.abs(x_bar_1 - x_bar_2) / tf.sqrt((s1 + s2) / 2.0)

    @staticmethod
    @tf.function
    def cross_product_norm(a: tf.float32, b: tf.float32):
        """
        |a x b| = |a.b| cot(theta)
        :param a:
        :param b:
        :return:
        """
        dot = tf.math.reduce_sum(tf.multiply(a, b), axis=1)
        cos = dot / (tf.norm(a, axis=1) * tf.norm(b, axis=1))
        cot = cos / tf.math.sqrt(1.0 + tf.math.pow(cos, 2))
        return dot * cot

    @staticmethod
    @tf.function
    def points_curvature(points: tf.float32) -> tf.float32:
        """
        k = | r' X r'' | / |r'^3|
        :param points:
        :return:
        """
        r = points
        rp = BazierCurve.points_prime(points)
        rpp = BazierCurve.points_prime(rp)
        rp_rpp = BazierCurve.cross_product_norm(rp, rpp)
        rp3 = tf.norm(tf.pow(rp, 3), axis=1)
        k = tf.math.divide_no_nan(rp_rpp, rp3)
        return tf.cast(k, tf.float32)

    @staticmethod
    @tf.function
    def parameter_distribution(data_points: tf.float32) -> tf.float32:
        prime = BazierCurve.points_prime(data_points)
        dist = tf.math.cumsum(tf.norm(prime, axis=1))
        dist /= tf.math.reduce_max(dist)
        return dist

    @staticmethod
    @tf.function
    def curvature_cost(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        curvature_ture = BazierCurve.points_curvature(y_true)
        curvature_pred = BazierCurve.points_curvature(y_pred)
        return tf.keras.losses.MeanAbsolutePercentageError()(curvature_ture, curvature_pred)

    @staticmethod
    @tf.function
    def curvature_accuracy(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        cost = BazierCurve.curvature_cost(y_true, y_pred)
        cost = tf.abs(tf.tanh(cost))
        acc = 100.0 * (1.0 - cost)
        acc = BazierCurve.replace_nan(acc, 0)
        acc = tf.reduce_mean(acc)
        return acc

    @staticmethod
    @tf.function
    def fitting_accuracy(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        return 100.0 * (1.0 - BazierCurve.fit_cost(y_true, y_pred))

    @staticmethod
    @tf.function
    def replace_nan(tensor: tf.float32, value: float):
        return tf.where(tf.math.is_nan(tensor), tf.ones_like(tensor) * value, tensor)

    @staticmethod
    @tf.function
    def fit_cost(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        cost = tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)
        return tf.cast(cost, tf.float32)

    @staticmethod
    @tf.function
    def prime_curve_fit_cost(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        prime_curve_true = BazierCurve.points_prime(tf.squeeze(y_true))
        prime_curve_pred = BazierCurve.points_prime(tf.squeeze(y_pred))
        return BazierCurve.fit_cost(prime_curve_true, prime_curve_pred)

    @staticmethod
    @tf.function
    def double_prime_curve_fit_cost(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        prime_curve_true = BazierCurve.points_prime(tf.squeeze(y_true))
        prime_curve_pred = BazierCurve.points_prime(tf.squeeze(y_pred))
        double_prime_curve_true = BazierCurve.points_prime(prime_curve_true)
        double_prime_curve_pred = BazierCurve.points_prime(prime_curve_pred)
        return BazierCurve.fit_cost(double_prime_curve_true, double_prime_curve_pred)

    @staticmethod
    @tf.function
    def total_cost(y_true: tf.float32, y_pred: tf.float32) -> tf.float32:
        losses = list()
        losses.append(BazierCurve.fit_cost(y_true, y_pred))
        # losses.append(BazierCurve.curvature_cost(y_true, y_pred))
        losses.append(BazierCurve.prime_curve_fit_cost(y_true, y_pred))
        losses.append(BazierCurve.double_prime_curve_fit_cost(y_true, y_pred))
        losses = [tf.expand_dims(_, axis=0) for _ in losses]
        losses = tf.concat(losses, axis=0)
        return BazierCurve.parallel_sum(losses)

    @staticmethod
    @tf.function
    def choice(inputs: tf.float32, sample_num: int):
        idxs = tf.range(tf.shape(inputs)[0])
        ridxs = tf.random.shuffle(idxs)[:sample_num]
        return tf.gather(inputs, ridxs)

    @staticmethod
    @tf.function
    def parallel_sum(losses):
        losses = tf.math.reciprocal_no_nan(losses)
        losses = tf.math.reduce_sum(losses)
        return tf.math.reciprocal_no_nan(losses)

    @staticmethod
    def points_curve_order(data: tf.float32, delta: float = 1e-4) -> int:
        bp = BazierCurve.points_prime(data)
        curve_order = 1
        while tf.reduce_mean(bp) > delta:
            bp = BazierCurve.points_prime(bp)
            curve_order += 1
            if curve_order >= data.shape[0]:
                break
        return curve_order


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


class BazierCurveFitting(keras.Model):
    def __init__(self, order: int, datapoints: tf.float32):
        self.variable_anchors = None
        self.padded_param1 = None
        self.padded_param2 = None
        self.padded_param3 = None
        self.var_shape = None
        self.nn = None
        self.anchors = None
        self.parameters = None
        self.dimensions = None
        self.data_size = None
        self.order = order
        self.datapoints = datapoints
        start_point = datapoints[0, :]
        end_point = datapoints[-1, :]
        self.start_point = tf.expand_dims(tf.constant(start_point, dtype=tf.float32), axis=0)
        self.end_point = tf.expand_dims(tf.constant(end_point, dtype=tf.float32), axis=0)
        self.constant = BazierCurve.constant_matrix(self.order)
        self.scale = tf.reduce_max(self.end_point)
        print(BazierCurve.choice(datapoints, order - 2))
        paddedzeros = tf.zeros_like(self.start_point)
        self.random_inital_anchors = tf.concat(
            [paddedzeros, BazierCurve.choice(datapoints, order - 2), paddedzeros], axis=0)
        super(BazierCurveFitting, self).__init__()

    def build(self, input_shape):
        self.data_size, self.dimensions = input_shape
        self.var_shape = (self.order - 2, self.dimensions)
        self.variable_anchors = self.add_weight(
            shape=self.var_shape,
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.05 * self.scale, maxval=0.05 * self.scale
            ),
            trainable=True
        )
        self.parameters = tf.linspace(0.0, 1.0, self.data_size)
        self.parameters = tf.expand_dims(self.parameters, axis=1)

    def call(self, inputs: tf.float32, training=True, mask=None):
        self.anchors = tf.concat([self.start_point, self.variable_anchors, self.end_point], axis=0)
        self.anchors = self.anchors + self.random_inital_anchors
        T = BazierCurve.power_series_matrix(self.parameters, self.order)
        TC = tf.matmul(T, self.constant)
        TCA = tf.matmul(TC, self.anchors)
        inputs = tf.cast(inputs, tf.float32)
        self.add_loss(BazierCurve.total_cost(inputs, TCA))
        return tf.expand_dims(TCA, axis=0)

    @property
    def current_anchor_points(self):
        var_a = self.get_weights()[0]
        return tf.concat([self.start_point, var_a, self.end_point], axis=0) + self.random_inital_anchors


class ParallelLayer(keras.layers.Layer):
    def __init__(self, parallel_layers, **kwargs):
        self.parallel_layers = parallel_layers
        super(ParallelLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        for layer in self.parallel_layers:
            layer.build(input_shape)

    def call(self, inputs, *args, **kwargs):
        layer_outputs = [layer(inputs) for layer in self.parallel_layers]
        # layer_outputs = [tf.nn.dropout(out, rate=0.1) for out in layer_outputs]
        return reduce(tf.add, layer_outputs) / len(self.parallel_layers)

    def get_config(self):
        config = super().get_config()
        config.update({
            "parallel_layers": self.parallel_layers,
        })
        return config

    @property
    def current_anchor_points(self):
        anchor_points = [
            layer.current_anchor_points
            for layer in self.parallel_layers]
        return reduce(tf.add, anchor_points) / len(self.parallel_layers)


def bazier_curve_fitting(datapoints, order: int):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=datapoints.shape, batch_size=1),
        keras.layers.Lambda(tf.squeeze),
        ParallelLayer([BazierCurveFitting(order, datapoints) for _ in range(100)]),
    ])
    model.build(datapoints.shape)
    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True,
    )

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
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=[
            BazierCurve.total_cost,
        ],
        metrics=[
            BazierCurve.curvature_cost,
            'mean_absolute_percentage_error',
            BazierCurve.prime_curve_fit_cost,
            BazierCurve.double_prime_curve_fit_cost
        ])
    model.fit(
        x=X, y=X, batch_size=1,
        epochs=1000,
        verbose=1,
        callbacks=[
            PlotBC(),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs',
                update_freq='epoch',
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                verbose=1,
                patience=64,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
            LearningRateLogger(),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='mean_absolute_percentage_error',
                factor=0.995,
                patience=16,
                verbose=1,
            )
        ],
        use_multiprocessing=True
    )
    os.system('tensorboard --logdir=/home/hhj/PycharmProjects/BezierCurve/logs')
    return model.layers[1].current_anchor_points


ClearFolder('/home/hhj/PycharmProjects/BezierCurve/images')
ClearFolder('/home/hhj/PycharmProjects/BezierCurve/logs')
order = 9
dim = 2
data_size = 40
a = tf.random.uniform((order, dim))
a = tf.sort(a, axis=0)
bc = BazierCurve(a, data_size)
this_curve_order = BazierCurve.points_curve_order(bc.data_points)
print(f'this curve order = {this_curve_order}/{type(this_curve_order)}')
an = bazier_curve_fitting(bc.data_points, this_curve_order)
print(an)
