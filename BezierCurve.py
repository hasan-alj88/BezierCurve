import keras.layers
import tensorflow as tf
import numpy as np
from math import comb
from itertools import product


@tf.function
def bezier_curve_constant_matrix(order: int) -> tf.float32:
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
    c = tf.squeeze(c)
    return tf.expand_dims(c, axis=0)


def power_series(tensor: tf.float32, power: int):
    t1 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(tensor)
    t0 = tf.ones_like(t1)
    powers = [t0, t1]
    for _ in range(1, power-1):
        powers.append(tf.keras.layers.multiply([t1, powers[-1]]))
    all_t = tf.keras.layers.Concatenate(axis=2)(powers)
    return all_t


@tf.function
def power_series_inv(tensor: tf.float32):
    power = tensor.shape[1]
    if order < 3:
        tensor = tf.math.reduce_prod(tensor, axis=2)
        tensor = tf.math.pow(tensor, 1.0/order)
    else:
        tensor = tf.math.reduce_sum(tensor, axis=2)
        tensor = 1.0 / tensor
        tensor = 1.0 - tensor
    return tensor


def triple_dot(x, b, c):
    x = tf.keras.layers.Dot(axes=(2, 1))([x, b])
    x = tf.keras.layers.Dot(axes=(2, 1))([x, c])
    return x


def parameter_bazier_curve_model(anchors, data):
    bc_order, dimension = anchors.shape
    data_size = data.shape[0]
    bc_c = bezier_curve_constant_matrix(bc_order)
    bc_c_inv = tf.linalg.inv(bc_c)
    p_inv = tf.linalg.pinv(anchors)
    p_inv = tf.expand_dims(p_inv, axis=0)
    a_in = tf.keras.Input(anchors.shape, batch_size=1, name='AnchorPoints')
    b_in = tf.keras.Input(data.shape, batch_size=1, name='DataPoints')

    t_ps = triple_dot(b_in, p_inv, bc_c_inv)
    t2 = tf.keras.layers.Lambda(lambda x: power_series_inv(x), name='PowerSeries_inv')(t_ps)
    t_new = tf.keras.layers.Dense(data_size, activation='relu')(t2)
    t_new = tf.keras.layers.Dropout(0.2)(t_new)
    t_new = tf.keras.layers.Dense(data_size, activation='relu')(t_new)
    get_t_new = tf.keras.Model(inputs=[a_in, b_in], outputs=[t_new])

    t_new_ps = power_series(t_new, bc_order)
    b_new = triple_dot(t_new_ps, bc_c, a_in)
    model = tf.keras.Model(inputs=[a_in, b_in], outputs=[b_new])
    # model.build([a_in.shape, b_in.shape])
    model.compile('adam', 'mse')
    tf.keras.utils.plot_model(model, "model.png")
    print(model.summary())
    return get_t_new, model


order, data_size, dimension = 3, 10, 2
b = np.random.random((data_size, dimension))
a = np.random.random((order, dimension))
parameter_bazier_curve_model(a, b)













