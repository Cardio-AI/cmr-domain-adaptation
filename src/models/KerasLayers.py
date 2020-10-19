from tensorflow.keras import backend as K
import tensorflow as tf
import math as m
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.layers import UpSampling2D as UpSampling2DInterpol
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as kl

__all__ = ['UpSampling2DInterpol', 'UpSampling3DInterpol', 'Euler2Matrix', 'ScaleLayer',
           'UnetWrapper', 'ConvEncoder', 'conv_layer_fn', 'downsampling_block_fn', 'upsampling_block_fn',
           'Inverse3DMatrix', 'ConvDecoder', 'ConvEncoder']


class UpSampling3DInterpol(UpSampling3D):

    def __init__(self, size=(1, 2, 2), interpolation='bilinear', **kwargs):
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.x = int(size[1])
        self.y = int(size[2])
        self.interpolation = interpolation
        super(self.__class__, self).__init__(**kwargs)

    def call(self, x):
        """
        :param x:
        :return:
        """
        target_size = (x.shape[2] * self.x, x.shape[3] * self.y)
        # traverse along the 3D volumes, handle the z-slices as batch
        return K.stack(
            tf.map_fn(lambda images:
                      tf.image.resize(
                          images=images,
                          size=target_size,
                          method=self.interpolation,  # define bilinear or nearest neighbor
                          preserve_aspect_ratio=True),
                      x))

    def get_config(self):
        config = super(UpSampling3DInterpol, self).get_config()
        config.update({'interpolation': self.interpolation, 'size': self.size})
        return config


class Inverse3DMatrix(Layer):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

    def call(self, m, **kwargs):
        """
        Calculate the inverse for an affine matrix
        :param m:
        :return:
        """
        # get the inverse of the affine matrix
        batch_size = tf.shape(m)[0]
        m_matrix = tf.keras.layers.Reshape(target_shape=(3, 4))(m)

        # Create a tensor with (b,1,4) and concat it to the affine matrix tensor (b,3,4)
        # and create a square tensor with (b,4,4)
        # (hack to slice the transformation matrix into an identity matrix)
        # don't know how to assign values to a tensor such as in numpy
        one = tf.ones((batch_size, 1, 1), dtype=tf.float16)
        zero = tf.zeros((batch_size, 1, 1), dtype=tf.float16)
        row = tf.concat([zero, zero, zero, one], axis=-1)
        ident = tf.concat([m_matrix, row], axis=1)

        m_matrix_inv = tf.linalg.inv(ident)

        m_inv = m_matrix_inv[:, :3, :]  # cut off the last row
        return tf.keras.layers.Flatten()(m_inv)

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        return config


class Euler2Matrix(Layer):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

    def call(self, theta, **kwargs):
        euler_1 = tf.expand_dims(theta[:, 0], -1)
        euler_2 = tf.expand_dims(theta[:, 1], -1)
        euler_3 = tf.expand_dims(theta[:, 2], -1)

        # clip values in a range -pi to pi, transformation is only defined within this range
        # clipping so far not necessary and not tested
        # pi = tf.constant(m.pi)
        # euler_1 = tf.clip_by_value(euler_1, -pi, pi)
        # euler_2 = tf.clip_by_value(euler_2, -pi, pi)
        # euler_3 = tf.clip_by_value(euler_3, -pi, pi)

        one = tf.ones_like(euler_1)
        zero = tf.zeros_like(euler_1)

        rot_x = tf.stack([tf.concat([one, zero, zero], axis=1),
                          tf.concat([zero, tf.cos(euler_1), tf.sin(euler_1)], axis=1),
                          tf.concat([zero, -tf.sin(euler_1), tf.cos(euler_1)], axis=1)], axis=1)

        rot_y = tf.stack([tf.concat([tf.cos(euler_2), zero, -tf.sin(euler_2)], axis=1),
                          tf.concat([zero, one, zero], axis=1),
                          tf.concat([tf.sin(euler_2), zero, tf.cos(euler_2)], axis=1)], axis=1)

        rot_z = tf.stack([tf.concat([tf.cos(euler_3), tf.sin(euler_3), zero], axis=1),
                          tf.concat([-tf.sin(euler_3), tf.cos(euler_3), zero], axis=1),
                          tf.concat([zero, zero, one], axis=1)], axis=1)

        rot_matrix = tf.matmul(rot_z, tf.matmul(rot_y, rot_x))

        # Extend matrix by the translation parameters
        translation = tf.expand_dims(tf.stack([theta[:, 3], theta[:, 4], theta[:, 5]], axis=-1), axis=-1)
        rot_matrix = tf.concat([rot_matrix, translation], axis=2)
        rot_matrix = tf.keras.layers.Flatten()(rot_matrix)
        return rot_matrix

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        return config


class ScaleLayer(Layer):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.scale = tf.Variable(1.)

    def call(self, inputs, **kwargs):
        return inputs * self.scale

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        return config


class UnetWrapper(Layer):
    def __init__(self, unet, unet_inplane=(224, 224), resize=True, trainable=False, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.unet = unet
        self.unet.trainable = trainable
        self.unet_inplane = unet_inplane

        self.resize = resize

    def call(self, x, **kwargs):
        # Call the unet with each slice, map_fn needs more memory
        # x = tf.transpose(x, [1,0,2,3,4])
        # x = tf.map_fn(self.unet, x,)
        # x = tf.transpose(x, [1, 0, 2, 3,4])

        # resize if input inplane resolution is different to unet input shape
        #if not tf.equal(tf.shape(x)[-3], self.unet_inplane[0]) and tf.equal(tf.shape(x)[-2], self.unet_inplane[1]):

        """ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in range(len(x)):
            img_resized = tf.compat.v1.image.resize(x[i],
                                                    size=self.unet_inplane,
                                                    method=tf.image.ResizeMethod.BILINEAR,
                                                    align_corners=True, name='resize')
            ta.write(i, self.unet(img_resized))
        ta = ta.stack()
        x = tf.transpose(ta, [1,0,2,3,4])"""

        x = tf.unstack(x, axis=1)
        if self.resize:
            x = [self.unet(
                tf.compat.v1.image.resize(
                    images,
                    size=self.unet_inplane,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=True,
                    name='resize')) for images in x]
        else:
            x = [self.unet(img) for img in x]

        x = tf.stack(x, axis=1)
        return x

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        config.update(self.unet.get_config())
        config.update({'unet_inplane': self.unet_inplane})
        return config


class ConvEncoder(Layer):
    def __init__(self, activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters,
                 kernel_init, m_pool, ndims, pad):
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.depth = depth
        self.drop_3 = drop_3
        self.dropouts = dropouts
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.m_pool = m_pool
        self.ndims = ndims
        self.pad = pad

        self.downsamplings = []
        filters = self.filters

        for l in range(self.depth):
            db = DownSampleBlock(filters=filters,
                                 f_size=self.f_size,
                                 activation=self.activation,
                                 drop=self.dropouts[l],
                                 batch_norm=self.batch_norm,
                                 kernel_init=self.kernel_init,
                                 pad=self.pad,
                                 m_pool=self.m_pool,
                                 bn_first=self.bn_first,
                                 ndims=self.ndims)
            self.downsamplings.append(db)
            filters *= 2

        self.conv1 = ConvBlock(filters=filters, f_size=self.f_size,
                               activation=self.activation, batch_norm=self.batch_norm, kernel_init=self.kernel_init,
                               pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

        self.bn = Dropout(self.drop_3)
        self.conv2 = ConvBlock(filters=filters, f_size=self.f_size,
                               activation=self.activation, batch_norm=self.batch_norm, kernel_init=self.kernel_init,
                               pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

    def call(self, inputs, **kwargs):

        encs = []
        skips = []

        self.first_block = True

        for db in self.downsamplings:

            if self.first_block:
                # first block
                input_tensor = inputs
                self.first_block = False
            else:
                # all other blocks, use the max-pooled output of the previous encoder block as input
                # remember the max-pooled output from the previous layer
                input_tensor = encs[-1]

            skip, enc = db(input_tensor)
            encs.append(enc)
            skips.append(skip)

        # return the last encoding block result
        encoding = encs[-1]
        encoding = self.conv1(inputs=encoding)
        encoding = self.bn(encoding)
        encoding = self.conv2(inputs=encoding)

        # work as u-net encoder or cnn encoder
        return encoding, skips

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""

        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'depth': self.depth,
                       'drop_3': self.drop_3,
                       'dropouts': self.dropouts,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'm_pool': self.m_pool,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class ConvDecoder(Layer):
    def __init__(self, activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters,
                 kernel_init, up_size, ndims, pad):
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.depth = depth
        self.drop_3 = drop_3
        self.dropouts = dropouts
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.up_size = up_size
        self.ndims = ndims
        self.pad = pad
        self.upsamplings = []

        filters = self.filters
        for layer in range(self.depth):
            ub = UpSampleBlock(filters=filters,
                               f_size=self.f_size,
                               activation=self.activation,
                               drop=self.dropouts[layer],
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init,
                               pad=self.pad,
                               up_size=self.up_size,
                               bn_first=self.bn_first,
                               ndims=self.ndims)
            self.upsamplings.append(ub)
            filters /= 2

    def call(self, inputs, **kwargs):

        enc, skips = inputs

        for upsampling in self.upsamplings:
            skip = skips.pop()
            enc = upsampling([enc, skip])

        return enc

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""

        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'depth': self.depth,
                       'drop_3': self.drop_3,
                       'dropouts': self.dropouts,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'up_size': self.up_size,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class ConvBlock(Layer):
    def __init__(self, filters=16, f_size=(3, 3, 3), activation='elu', batch_norm=True, kernel_init='he_normal',
                 pad='same', bn_first=False, ndims=2):
        """
        Wrapper for a 2/3D-conv layer + batchnormalisation
        Either with Conv,BN,activation or Conv,activation,BN

        :param filters: int, number of filters
        :param f_size: tuple of int, filterzise per axis
        :param activation: string, which activation function should be used
        :param batch_norm: bool, use batch norm or not
        :param kernel_init: string, keras enums for kernel initialisation
        :param pad: string, keras enum how to pad, the conv
        :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
        :param ndims: int, define the conv dimension
        :return: a functional tf.keras conv block
        expects an numpy or tensor object with (batchsize,z,x,y,channels)
        """
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.ndims = ndims
        self.pad = pad
        self.encoder = list()

        # create the layers
        Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        f_size = self.f_size[:self.ndims]

        self.conv = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                         padding=self.pad)
        self.conv_activation = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                                    padding=self.pad)
        self.bn = BatchNormalization(axis=-1)
        self.activation = Activation(self.activation)

    def call(self, inputs, **kwargs):

        if self.bn_first:
            # , kernel_regularizer=regularizers.l2(0.0001)
            conv1 = self.conv(inputs)
            conv1 = self.bn(conv1) if self.batch_norm else conv1
            conv1 = self.activation(conv1)

        else:
            # kernel_regularizer=regularizers.l2(0.0001),
            conv1 = self.conv_activation(inputs)
            conv1 = self.bn(conv1) if self.batch_norm else conv1

        return conv1

    def get_config(self):
        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class DownSampleBlock(Layer):
    def __init__(self, filters=16, f_size=(3, 3, 3), activation='elu', drop=0.3, batch_norm=True,
                 kernel_init='he_normal', pad='same', m_pool=(2, 2), bn_first=False, ndims=2):
        """
    Create an 2D/3D-downsampling block for the u-net architecture
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param m_pool: tuple of int, size of the max-pooling filters
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    Excpects a numpy or tensor input with (batchsize,z,x,y,channels)
    """

        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.drop = drop
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.m_pool = m_pool
        self.ndims = ndims
        self.pad = pad
        self.encoder = list()

        self.m_pool = self.m_pool[-self.ndims:]
        self.pool_fn = getattr(kl, 'MaxPooling{}D'.format(self.ndims))
        self.pool = self.pool_fn(m_pool)
        self.conf1 = ConvBlock(filters=self.filters, f_size=self.f_size, activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)
        self.dropout = Dropout(self.drop)
        self.conf2 = ConvBlock(filters=self.filters, f_size=self.f_size, activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

    def call(self, x, **kwargs):
        x = self.conf1(x)
        x = self.dropout(x)
        conv1 = self.conf2(x)
        p1 = self.pool(conv1)

        return [conv1, p1]


class UpSampleBlock(Layer):
    def __init__(self, use_upsample=True, filters=16, f_size=(3, 3, 3), activation='elu',
                 drop=0.3, batch_norm=True, kernel_init='he_normal', pad='same', up_size=(2, 2), bn_first=False,
                 ndims=2):
        """
        Create an upsampling block for the u-net architecture
        Each blocks consists of these layers: upsampling/transpose,concat,conv,dropout,conv
        Either with "upsampling,conv" or "transpose" upsampling
        :param use_upsample: bool, whether to use upsampling or transpose layer
        :param filters: int, number of filters per conv-layer
        :param f_size: tuple of int, filter size per axis
        :param activation: string, which activation function should be used
        :param drop: float, define the dropout rate between the conv layers of this block
        :param batch_norm: bool, use batch norm or not
        :param kernel_init: string, keras enums for kernel initialisation
        :param pad: string, keras enum how to pad, the conv
        :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
        :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
        :param ndims: int, define the conv dimension
        :return: a functional tf.keras upsampling block
        Expects an input with length 2 lower block: batchsize,z,x,y,channels, skip layers: batchsize,z,x,y,channels
        """

        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.drop = drop
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.use_upsample = use_upsample
        self.up_size = up_size
        self.ndims = ndims
        self.pad = pad
        self.encoder = list()

        Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        UpSampling = getattr(kl, 'UpSampling{}D'.format(self.ndims))
        ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(self.ndims))

        f_size = self.f_size[-self.ndims:]

        # use upsample&conv or transpose layer
        self.upsample = UpSampling(size=self.up_size)
        self.conv1 = Conv(filters=self.filters, kernel_size=f_size, padding=self.pad,
                          kernel_initializer=self.kernel_init,
                          activation=self.activation)

        self.convTranspose = ConvTranspose(filters=self.filters, kernel_size=f_size, strides=self.up_size,
                                           padding=self.pad,
                                           kernel_initializer=self.kernel_init,
                                           activation=self.activation)

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        self.convBlock1 = ConvBlock(filters=self.filters, f_size=f_size, activation=self.activation,
                                    batch_norm=self.batch_norm,
                                    kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first,
                                    ndims=self.ndims)
        self.dropout = Dropout(self.drop)
        self.convBlock2 = ConvBlock(filters=self.filters, f_size=f_size, activation=self.activation,
                                    batch_norm=self.batch_norm,
                                    kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first,
                                    ndims=self.ndims)

    def call(self, inputs, **kwargs):

        if len(inputs) == 2:
            skip = True
            lower_input, conv_input = inputs
        else:
            skip = False
            lower_input = inputs

        # use upsample&conv or transpose layer
        if self.use_upsample:

            deconv1 = self.upsample(lower_input)
            deconv1 = self.conv1(deconv1)

        else:
            deconv1 = self.convTranspose(lower_input)

        # if skip given, concat
        if skip:
            deconv1 = self.concatenate([deconv1, conv_input])
        deconv1 = self.convBlock1(inputs=deconv1)
        deconv1 = self.dropout(deconv1)
        deconv1 = self.convBlock2(inputs=deconv1)

        return deconv1

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'drop': self.drop,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'm_pool': self.m_pool,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


def conv_layer_fn(inputs, filters=16, f_size=(3, 3, 3), activation='elu', batch_norm=True, kernel_init='he_normal',
                  pad='same', bn_first=False, ndims=2):
    """
    Wrapper for a 2/3D-conv layer + batchnormalisation
    Either with Conv,BN,activation or Conv,activation,BN

    :param inputs: numpy or tensor object batchsize,z,x,y,channels
    :param filters: int, number of filters
    :param f_size: tuple of int, filterzise per axis
    :param activation: string, which activation function should be used
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras conv block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[:ndims]

    if bn_first:
        # , kernel_regularizer=regularizers.l2(0.0001)
        conv1 = Conv(filters=filters, kernel_size=f_size, kernel_initializer=kernel_init, padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1
        conv1 = Activation(activation)(conv1)

    else:
        # kernel_regularizer=regularizers.l2(0.0001),
        conv1 = Conv(filters=filters, kernel_size=f_size, activation=activation, kernel_initializer=kernel_init,
                     padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1

    return conv1


def downsampling_block_fn(inputs, filters=16, f_size=(3, 3, 3), activation='elu', drop=0.3, batch_norm=True,
                          kernel_init='he_normal', pad='same', m_pool=(2, 2), bn_first=False, ndims=2):
    """
    Create an 2D/3D-downsampling block for the u-net architecture
    :param inputs: numpy or tensor input with batchsize,z,x,y,channels
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param m_pool: tuple of int, size of the max-pooling layer
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras downsampling block
    """
    m_pool = m_pool[-ndims:]
    pool = getattr(kl, 'MaxPooling{}D'.format(ndims))

    conv1 = conv_layer_fn(inputs=inputs, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                          kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    conv1 = Dropout(drop)(conv1)
    conv1 = conv_layer_fn(inputs=conv1, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                          kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    p1 = pool(m_pool)(conv1)

    return (conv1, p1)


def upsampling_block_fn(lower_input, conv_input, use_upsample=True, filters=16, f_size=(3, 3, 3), activation='elu',
                        drop=0.3, batch_norm=True, kernel_init='he_normal', pad='same', up_size=(2, 2), bn_first=False,
                        ndims=2):
    """
    Create an upsampling block for the u-net architecture
    Each blocks consists of these layers: upsampling/transpose,concat,conv,dropout,conv
    Either with "upsampling,conv" or "transpose" upsampling
    :param lower_input: numpy input from the lower block: batchsize,z,x,y,channels
    :param conv_input: numpy input from the skip layers: batchsize,z,x,y,channels
    :param use_upsample: bool, whether to use upsampling or not
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[-ndims:]

    # use upsample&conv or transpose layer
    if use_upsample:
        # import src.models.KerasLayers as ownkl
        # UpSampling = getattr(ownkl, 'UpSampling{}DInterpol'.format(ndims))
        UpSampling = getattr(kl, 'UpSampling{}D'.format(ndims))
        deconv1 = UpSampling(size=up_size)(lower_input)
        deconv1 = Conv(filters=filters, kernel_size=f_size, padding=pad, kernel_initializer=kernel_init,
                       activation=activation)(deconv1)

    else:
        ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(ndims))
        deconv1 = ConvTranspose(filters=filters, kernel_size=f_size, strides=up_size, padding=pad,
                                kernel_initializer=kernel_init,
                                activation=activation)(lower_input)

    deconv1 = tf.keras.layers.Concatenate(axis=-1)([deconv1, conv_input])

    deconv1 = conv_layer_fn(inputs=deconv1, filters=filters, f_size=f_size, activation=activation,
                            batch_norm=batch_norm,
                            kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    deconv1 = Dropout(drop)(deconv1)
    deconv1 = conv_layer_fn(inputs=deconv1, filters=filters, f_size=f_size, activation=activation,
                            batch_norm=batch_norm,
                            kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)

    return deconv1


# Downsampling part of a U-net, could be used to extract feature maps from a input 2D or 3D volume
def encoder_fn(activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters, inputs,
               kernel_init, m_pool, ndims, pad):
    """
    Encoder for 2d or 3d data, could be used for a U-net.
    Implementation based on the functional tf.keras api
    :param activation:
    :param batch_norm:
    :param bn_first:
    :param depth:
    :param drop_3:
    :param dropouts:
    :param f_size:
    :param filters:
    :param inputs:
    :param kernel_init:
    :param m_pool:
    :param ndims:
    :param pad:
    :return:
    """

    encoder = list()
    dropouts = dropouts.copy()

    # build the encoder
    for l in range(depth):

        if len(encoder) == 0:
            # first block
            input_tensor = inputs
        else:
            # all other blocks, use the max-pooled output of the previous encoder block
            # remember the max-pooled output from the previous layer
            input_tensor = encoder[-1][1]
        encoder.append(
            downsampling_block_fn(inputs=input_tensor,
                                  filters=filters,
                                  f_size=f_size,
                                  activation=activation,
                                  drop=dropouts[l],
                                  batch_norm=batch_norm,
                                  kernel_init=kernel_init,
                                  pad=pad,
                                  m_pool=m_pool,
                                  bn_first=bn_first,
                                  ndims=ndims))
        filters *= 2
    # middle part
    input_tensor = encoder[-1][1]
    fully = conv_layer_fn(inputs=input_tensor, filters=filters, f_size=f_size,
                          activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                          pad=pad, bn_first=bn_first, ndims=ndims)
    fully = Dropout(drop_3)(fully)
    fully = conv_layer_fn(inputs=fully, filters=filters, f_size=f_size,
                          activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                          pad=pad, bn_first=bn_first, ndims=ndims)
    return fully


def inverse_affine_matrix_fn(m):
    """
    Calculate the inverse for an affine matrix
    :param m:
    :return:
    """
    # get the inverse of the affine matrix
    batch_size = tf.shape(m)[0]
    m_matrix = tf.reshape(m, (batch_size, 3, 4))

    # concat a row with b,1,4 to b,3,4 and create a b,4,4
    # (hack to slice the transformation matrix into an identity matrix)
    # don't know how to assign values to a tensor such as in numpy
    one = tf.ones((batch_size, 1, 1))
    zero = tf.zeros((batch_size, 1, 1))
    row = tf.concat([zero, zero, zero, one], axis=-1)
    ident = tf.concat([m_matrix, row], axis=1)

    m_matrix_inv = tf.linalg.inv(ident)
    m_inv = m_matrix_inv[:, :3, :]  # cut off the last row
    return tf.keras.layers.Flatten()(m_inv)


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix_fn(theta):
    """
    Calculate a rotation and translation matrix from 6 parameters given in theta
    The input is a list of tensors, the first three will be interpreted as euler angles
    The last three as translation params.
    :param theta: list of tensors with a length of 6 each with the shape b,1
    :return:
    """

    one = tf.ones_like(theta[0], dtype=tf.float32)
    zero = tf.zeros_like(theta[0], dtype=tf.float32)

    rot_x = tf.stack([tf.concat([one, zero, zero], axis=1),
                      tf.concat([zero, tf.cos(theta[0]), tf.sin(theta[0])], axis=1),
                      tf.concat([zero, -tf.sin(theta[0]), tf.cos(theta[0])], axis=1)], axis=1)

    rot_y = tf.stack([tf.concat([tf.cos(theta[1]), zero, -tf.sin(theta[1])], axis=1),
                      tf.concat([zero, one, zero], axis=1),
                      tf.concat([tf.sin(theta[1]), zero, tf.cos(theta[1])], axis=1)], axis=1)

    rot_z = tf.stack([tf.concat([tf.cos(theta[2]), tf.sin(theta[2]), zero], axis=1),
                      tf.concat([-tf.sin(theta[2]), tf.cos(theta[2]), zero], axis=1),
                      tf.concat([zero, zero, one], axis=1)], axis=1)

    rot_matrix = tf.matmul(rot_z, tf.matmul(rot_y, rot_x))

    # Apply learnable translation
    translation = tf.expand_dims(tf.stack([theta[3][:, 0], theta[4][:, 0], theta[5][:, 0]], axis=-1), axis=-1)
    # ignore translation
    # translation = tf.expand_dims(tf.stack([zero[:,0],zero[:, 0], zero[:, 0]], axis=-1), axis=-1)
    rot_matrix = tf.concat([rot_matrix, translation], axis=2)
    rot_matrix = tf.keras.layers.Flatten()(rot_matrix)

    # model = Model(inputs=[theta], outputs=rot_matrix, name='Affine_matrix_builder')

    return rot_matrix


def affineMatrixInverter_fn(m):
    """
    Concats an affine Matrix (b,12) to square shape, calculates the inverse and returns the sliced version
    :param m:
    :return: m inverted
    """
    # get the inverse of the affine matrix, rotate y back and compare it with AXtoSAXtoAX
    batch_size = tf.shape(m)[0]
    m_matrix = tf.reshape(m, (batch_size, 3, 4))
    # concat a row with b,1,4 to b,3,4 and create a b,4,4
    # (hack to slice the transformation matrix into an identity matrix)
    # don't know how to assign values to a tensor such as in numpy
    one = tf.ones((batch_size, 1, 1))
    zero = tf.zeros((batch_size, 1, 1))
    row = tf.concat([zero, zero, zero, one], axis=-1)
    ident = tf.concat([m_matrix, row], axis=1)

    m_matrix_inv = tf.linalg.inv(ident)
    m_inv = m_matrix_inv[:, :3, :]  # cut off the last row
    m_inv = tf.keras.layers.Flatten()(m_inv)

    return m_inv
