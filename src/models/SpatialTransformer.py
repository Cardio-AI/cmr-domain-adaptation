'''
Unsupervised Domain Adaption from Axial
Sven Koehler, Tarique Hussain, Zach Blair, Tyler Huffaker, Florian Ritzmann, Animesh Tandon,
Thomas Pickardt, Samir Sarikouch, Heiner Latus, Gerald Greil, Ivo Wolf, Sandy Engelhardt

Spatial transformer architecture, based on the voxelmorph code

tensorflow/keras utilities for the neuron project
Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.
'''

# main imports
import sys

# third party
import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from src.models.KerasLayers import Euler2Matrix, ScaleLayer, UnetWrapper, ConvEncoder, Inverse3DMatrix
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.initializers
import tensorflow as tf
import logging
import src.utils.Metrics_own as metr


from src.models.ModelUtils import get_optimizer


sys.path.append('src/models/ext/neuron')
sys.path.append('src/models/ext/pynd-lib')
sys.path.append('src/models/ext/pytools-lib')
import src.models.ext.neuron.neuron.layers as nrn_layers



def create_affine_cycle_transformer_model(config, metrics=None, networkname='affine_cycle_transformer', unet=None):
    """
    Create a compiled Domain adaption (AX2SAX) spatial transformer model with three loss functions.

    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param unet: tf.keras.Model, pre-trained 2D U-net
    :param use_mask2ax_prob: bool, use SAX or SAX2AX mask to max the probability
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
        y_pred = y_pred[...,1:] if background given by the U-Net predictions
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
    #tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
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
        dense_weights = config.get('DENSE_WEIGHTS', 256)
        indexing = config.get('INDEXING', 'ij')

        weight_mse_inplane = config.get('WEIGHT_MSE_INPLANE', True) # weight the MSE loss pixels in the center have greater weights
        mask_smaller_than_threshold = config.get('MASK_SMALLER_THAN_THRESHOLD', 0.01) # calc the MSe loss only where our image has values greater than
        ax_weight = config.get('AX_LOSS_WEIGHT', 2)

        cycle_loss = config.get('CYCLE_LOSS', False)
        sax_weight = config.get('SAX_LOSS_WEIGHT', 2)

        focus_loss = config.get('FOCUS_LOSS', False)
        focus_weight = config.get('FOCUS_LOSS_WEIGHT', 1)
        min_unet_probability = config.get('MIN_UNET_PROBABILITY', 0.9) # sum the foreground voxels with a prob higher than
        use_mask2ax_prob = config.get('USE_SAX2AX_PROB', True) # otherwise use the SAX probability




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
        m_raw = tensorflow.keras.layers.Dense(dense_weights, kernel_initializer=kernel_init, activation=activation,name='dense1')(m_raw)
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

            # baseline loss
            logging.info('adding ax2sax MSE loss with a weighting of {}'.format(ax_weight))
            losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=mask_smaller_than_threshold, weight_inplane=weight_mse_inplane, xy_shape=input_shape[-2])}
            loss_w = {'ax2sax': ax_weight}

            # extend losses by cycle MSE loss
            if cycle_loss:
                logging.info('adding cycle loss with a weighting of {}'.format(sax_weight))
                losses['sax2ax'] = metr.loss_with_zero_mask(mask_smaller_than=mask_smaller_than_threshold, weight_inplane=weight_mse_inplane, xy_shape=input_shape[-2])
                loss_w['sax2ax'] = sax_weight

            # extend losses by probability loss
            if focus_loss:
                # Use the SAX predictions or the SAX2AX predictions to maximise the unet probability
                # probability_object must fit a output-layer name
                if use_mask2ax_prob:
                    probability_object = 'mask2ax'
                else:
                    probability_object = 'mask_prob'
                logging.info('adding focus loss on {} with a weighting of {}'.format(probability_object, focus_weight))
                losses[probability_object] = metr.max_volume_loss(min_probabillity=min_unet_probability)
                loss_w[probability_object] = focus_weight


        else: # no u-net given
            outputs = [ax2sax, sax2ax, m]
            # baseline loss
            logging.info('adding ax2sax MSE loss with a weighting of {}'.format(ax_weight))
            losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=mask_smaller_than_threshold,
                                                         weight_inplane=weight_mse_inplane,
                                                         xy_shape=input_shape[-2])}
            loss_w = {'ax2sax': ax_weight}

            # extend losses by cycle MSE loss
            if cycle_loss:
                logging.info('adding cycle MSE loss with a weighting of {}'.format(sax_weight))
                losses['sax2ax'] = metr.loss_with_zero_mask(mask_smaller_than=mask_smaller_than_threshold,
                                                            weight_inplane=weight_mse_inplane,
                                                            xy_shape=input_shape[-2])
                loss_w['sax2ax'] = sax_weight


        model = Model(inputs=[inputs_ax, inputs_sax], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname), loss=losses, loss_weights=loss_w)

        return model


# ST to apply m to an volume
def create_affine_transformer_fixed(config, metrics=None, networkname='affine_transformer_fixed', fill_value=0, interp_method='linear'):
    """
    Apply a learned transformation matrix to an input image, no training possible
    :param config:  Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))

    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)))
        input_matrix = Input((12), dtype=np.float32)
        indexing = config.get('INDEXING','ij')

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=False, fill_value=fill_value)([inputs, input_matrix])

        model = Model(inputs=[inputs, input_matrix], outputs=[y, input_matrix], name=networkname)

        return model

