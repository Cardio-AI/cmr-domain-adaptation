# spatial transformer architecture, based on the voxelmorph code
'''tensorflow/keras utilities for the neuron project

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.
'''

# main imports
import sys

# third party
import numpy as np
import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from tensorflow.keras.layers import LeakyReLU, Reshape, Lambda, Dropout
from src.models.KerasLayers import Euler2Matrix, ScaleLayer, UnetWrapper, ConvEncoder, Inverse3DMatrix
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.initializers
import tensorflow as tf
import logging
import src.utils.Metrics_own as metr

# import neuron layers, which will be useful for Transforming.
from tensorflow.keras.optimizers import Adam

from src.models.ModelUtils import get_optimizer
from src.models.KerasLayers import downsampling_block, conv_layer
from src.models.Unets import unet
from src.models.src import losses

sys.path.append('src/models/ext/neuron')
sys.path.append('src/models/ext/pynd-lib')
sys.path.append('src/models/ext/pytools-lib')
import src.models.ext.neuron.neuron.layers as nrn_layers
import src.models.ext.neuron.neuron.models as nrn_models
import src.models.ext.neuron.neuron.utils as nrn_utils


def create_affine_cycle_transformer_model(config, metrics=None, networkname='affine_cycle_transformer', unet=None):
    """
    Create a compiled Domain adaption (AX2SAX) spatial transformer model with three loss functions.

    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param unet: tf.keras.Model, pre-trained 2D U-net
    :return: compiled tf.keras.Model

    The returned tf.keras.Model expects the following input during training:
    [ax_cmr,  sax_cmr]
    and returns the following elements:
    [ax2sax_cmr, sax2ax_cmr, ax2sax_mod_cmr,sax_msk, ax_msk, m, m_mod]
    During inference sax_cmr could be None or a zero initialised ndarray

    This model has the following flow:
    inputs = [AX, SAX]
    m, m_mod = Encoder(AX)
    m_inv = Inverse(m)
    ax2sax = SpatialTransformer(AX, m)
    ax2sax = SpatialTransformer(AX, m_mod)
    sax2ax = SpatialTransformer(SAX, m_inv)
    sax_msk = Unet(SAX)
    ax_msk = SpatialTransformer(sax_msk, m_mod)
    outputs = [ax2sax, sax2ax, ax2sax_mod, sax_msk, ax_msk, m, m_mod]

    This model calculates the gradients according to the following loss functions:
    MSE_mod(ax2sax_gt, ax2sax_pred) # learn an affine transformation that fits to our gt
    MSE_mod(sax2ax_gt, sax2ax_pred) # apply the opposite transformation to align the cycle consistency
    Loss_focus(ax_msk) or Loss_focus(sax_msk)

    - MSE_mod = MSE(y_true, y_pred) * weighting[None, None,: , : ],
    Example in-plane weighting with an in-plane resolution of 10 x 10 pixels:
    weighting =
    [[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
     [0.   0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.  ]
     [0.   0.25 0.5  0.5  0.5  0.5  0.5  0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 0.75 0.75 0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 1.   1.   0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 1.   1.   0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.75 0.75 0.75 0.75 0.5  0.25 0.  ]
     [0.   0.25 0.5  0.5  0.5  0.5  0.5  0.5  0.25 0.  ]
     [0.   0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.  ]
     [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]

    - Loss_focus =
        # ignore background, we want to maximize the number of captured ventricle voxel
        y_pred = y_pred[...,1:]
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # keep only the highest prob
        sum_bigger_than = tf.reduce_max(y_pred, axis=-1)
        # create a mask of voxels greater than the threshold
        mask_bigger_than = tf.cast(sum_bigger_than > min_probabillity, tf.float32)
        # we cant use the mask directly as this creates no gradients, this keeps also the prob value
        sum_bigger_than = sum_bigger_than * mask_bigger_than
        # return a scalar between 0 and 1, the loss is typically close to 1, as the background class is overrepresented
        return 1- tf.reduce_mean(sum_bigger_than)
    """

    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        input_shape = config.get('DIM', [10, 224, 224])
        inputs_ax = Input((*input_shape, config.get('IMG_CHANNELS', 1)))
        inputs_sax = Input((*input_shape, config.get('IMG_CHANNELS', 1)))
        # define standard values according to the convention over configuration paradigm
        activation = config.get('ACTIVATION', 'elu')
        batch_norm = config.get('BATCH_NORMALISATION', False)
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        indexing = config.get('INDEXING', 'ij')

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        enc, _ = ConvEncoder(activation=activation,
                             batch_norm=batch_norm,
                             bn_first=bn_first,
                             depth=depth,
                             drop_3=drop_3,
                             dropouts=dropouts,
                             f_size=f_size,
                             filters=filters,
                             kernel_init=kernel_init,
                             m_pool=m_pool,
                             ndims=ndims,
                             pad=pad)(inputs_ax)

        # Shrink the encoding towards the euler angles and translation params,
        # no additional dense layers before the GAP layer
        m_raw = tensorflow.keras.layers.GlobalAveragePooling3D()(enc)  # m.shape --> b, 512
        m_raw = tensorflow.keras.layers.Dense(256, kernel_initializer=kernel_init, activation=activation,name='dense1')(m_raw)
        m_raw = tensorflow.keras.layers.Dense(9, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),activation=activation, name='dense2')(m_raw)
        m = Euler2Matrix(name='ax2sax_matrix')(m_raw[:, 0:6])

        # Cycle flow - use M and the inverse M to transform the SAX and AX input
        ax2sax = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='ax2sax')([inputs_ax, m])
        m_inv = Inverse3DMatrix()(m)
        sax2ax = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='sax2ax')([inputs_sax, m_inv])

        if unet:
            logging.info('unet given, use it to max probability')

            # concat the rotation parameters with the second set of translation params
            m_mod = tf.keras.layers.Concatenate(axis=-1)([m_raw[:, 0:3], m_raw[:, 6:9]])  # rot + translation
            m_mod = Euler2Matrix(name='ax2sax_mod_matrix')(m_mod)  #
            ax2sax_mod = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0, name='ax2sax_mod_st')([inputs_ax, m_mod])

            # we use the probabilities of a pre-trained U-net with fixed weights
            # to learn a second set of translation parameters which maximize the Unet probability
            mask_prob = UnetWrapper(unet, name='mask_prob')(ax2sax_mod)  #
            m_mod_inv = Inverse3DMatrix()(m_mod)  #
            mask2ax = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing, ident=False, fill_value=0, name='mask2ax')([mask_prob, m_mod_inv])

            # Define the model output
            outputs = [ax2sax, sax2ax, ax2sax_mod, mask_prob, mask2ax, m, m_mod]

            # Define the loss functions
            losses = {
                'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_inplane=True, xy_shape=input_shape[-2]),
                'sax2ax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_inplane=True, xy_shape=input_shape[-2]),
                'mask2ax': metr.max_volume_loss(min_probabillity=0.9)
            }

            # Define the loss weighting
            loss_w = {
                'ax2sax': 20.0,
                'sax2ax': 10.0,
                'mask2ax': 1.0
            }
        else: # no u-net given
            outputs = [ax2sax, sax2ax, m]
            losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_inplane=True, xy_shape=input_shape[-2]),
                      'sax2ax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_inplane=True, xy_shape=input_shape[-2])}
            loss_w = {'ax2sax': 20.0,
                      'sax2ax': 10.0}

        model = Model(inputs=[inputs_ax, inputs_sax], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname), loss=losses, loss_weights=loss_w)

        return model




