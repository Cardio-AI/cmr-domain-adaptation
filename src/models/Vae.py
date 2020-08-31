from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_vae(config={}, load_weights=False):

    """

    :param config:
    :param load_weights:
    :return:
    """

    # network parameters
    img_channels = config.get('IMG_CHANNELS', 1)
    input_shape = (*config.get('DIM', [224, 224]), img_channels)
    kernel_size = 3
    filters = 16
    latent_dim = config.get('LATENT_DIM', 10)
    intermediate_dim = 32
    ndims = len(config.get('DIM', [224, 224]))
    layers = 3

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    assert ndims in [2, 3]
    for i in range(layers):
        filters *= 2
        convL = getattr(KL, 'Conv%dD' % ndims)
        x = convL(filters=filters,
                   kernel_size=kernel_size,
                   activation='elu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(intermediate_dim, activation='tanh')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    if ndims == 3:
        x = Dense(shape[1] * shape[2] * shape[3] * shape[4], activation='elu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3], shape[4]))(x)
    else:
        x = Dense(shape[1] * shape[2] * shape[3], activation='elu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(layers):
        conv_transposeL = getattr(KL, 'Conv%dDTranspose' % ndims)
        x = conv_transposeL(filters=filters,
                            kernel_size=kernel_size,
                            activation='elu',
                            strides=2,
                            padding='same')(x)
        filters //= 2
    print(x.shape)

    outputs = conv_transposeL(filters=img_channels,
                              kernel_size=kernel_size,
                              activation='tanh',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    if load_weights:
        vae.load_weights(os.path.join(config.get('MODEL_PATH', './'), 'checkpoint.h5'))
    # VAE loss = mse_loss or binary crossentropy_loss + kl_loss

    vae_loss = vae_loss_factory(config, z_log_var, z_mean, inputs, outputs)
    vae.add_loss(vae_loss)
    lr = config.get('LEARNING_RATE', 0.0001)
    opt = Adam(lr=lr)

    vae.compile(optimizer=opt)
    vae.summary()
    return encoder, decoder, vae



def vae_loss_factory(config, z_log_var, z_mean, inputs, outputs):
    """
    Create a loss for the VAE model, could be mean squared error or binary crossentropy
    combined with KL loss
    :param config:
    :param z_log_var:
    :param z_mean:
    :param inputs:
    :param outputs:
    :return:
    """

    image_size = config.get('DIM', [224, 224])
    ndims = len(config.get('DIM', [224, 224]))

    from keras.losses import categorical_crossentropy

    def vae_loss_f(inputs, outputs):
        # VAE loss = mse_loss or binary crossentropy_loss + kl_loss
        if config.get('MSE', False):
            reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        else:
            # tried it with categorical crossentropy
            #reconstruction_loss = categorical_crossentropy(K.flatten(inputs),
             #                                         K.flatten(outputs))

            reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                      K.flatten(outputs))

        if ndims == 2:
            reconstruction_loss *= image_size[0] * image_size[1]
        elif ndims == 3:
            pass
            reconstruction_loss *= image_size[0] * image_size[1] #* image_size[2] # reduce loss to prevent exploding gradients

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    return vae_loss_f(inputs, outputs)



