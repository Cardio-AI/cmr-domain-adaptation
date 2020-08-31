# spatial transformer architecture, based on the voxelmorph code
"""tensorflow/keras utilities for the neuron project

If you use this code, please cite
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018

or for the transformation/integration functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3"""

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

# encoder learns euler angles and translation param, builds m, ST applies m
def create_affine_transformer_max_unet_predictions(config, metrics=None, networkname='affine_transformer', single_model=True, supervision=False, unet=None):
    """
    create a trainable spatial transformer for AX2SAX Domain transformation
    MOdel expects an AX stack as input and returns AX2SAX, the affine transformation matrix M
    Additionally if a unet is injected, the model predicts the LV, MYO and RV on the AX2SAX transformed stack
    inverts the transformation and returns a segmentation in axial view.
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        inputs_ax = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)), dtype=np.float32)

        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        batch_norm = config.get('BATCH_NORMALISATION', False)
        use_upsample = config.get('USE_UPSAMPLE', 'False')
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[:ndims]
        lr = config.get('LR', 1e-4)
        indexing = config.get('INDEXING','ij')

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        outputs = Encoder_fn(activation=activation,
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
                               pad=pad,
                             inputs=inputs_ax)

        # direct convergence, no additional dense layers
        dense = tensorflow.keras.layers.GlobalAveragePooling3D()(outputs) # dense.shape --> b, 512
        dense = tensorflow.keras.layers.Dense(256, kernel_initializer=kernel_init, activation=activation,name='dense1')(dense)
        m_raw = tensorflow.keras.layers.Dense(9, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),activation=activation, name='dense2')(dense)

        m = Euler2Matrix()(m_raw[ :,0:6])

        # transform the source with the affine matrix (rotation + translation)
        ax2sax = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='ax2sax')([inputs_ax, m])

        if unet:
            # predict a mask, transform this mask back into the original orientation
            m_mod = tf.keras.layers.Concatenate(axis=-1)([m_raw[:, 0:3], m_raw[:, 6:9]])  #
            m_mod = Euler2Matrix()(m_mod)  #

            ax2sax_mod = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False,
                                                       fill_value=0, name='ax2sax_mod_st')([inputs_ax, m_mod])
            mask_prob = UnetWrapper(unet, name='mask_prob')(ax2sax_mod)  #
            m_mod_inv = Inverse3DMatrix()(m_mod)  #
            mask2ax = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing, ident=False,
                                                    fill_value=0, name='mask2ax')([mask_prob, m_mod_inv])  #


            outputs=[ax2sax,ax2sax_mod, mask_prob, mask2ax, m, m_mod] #
            losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_z=False, weight_inplane=True),
                      'mask_prob': metr.max_volume_loss(min_probabillity=0.9, z_weight=False)}
            loss_w = {'ax2sax': 1.0,
                      'mask_prob': 0.1}
        else:
            outputs = [ax2sax, m]
            losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_z=False, weight_inplane=True)}
            loss_w = {'ax2sax': 1.0}

        model = Model(inputs=[inputs_ax], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname),
                      loss=losses,
                      loss_weights=loss_w)

        return model




# rotation robust unet, encoder,angles and translation, m, ST, unet, ST with inverse m
def create_spatial_unet(config, metrics=None, networkname='enc_st_unet_st', single_model=True, supervision=False, unet_2d=None):

    """
        create a trainable spatial transformer with a cycle loss function
        :param config: Key value pairs for image size and other network parameters
        :param metrics: list of tensorflow or keras compatible metrics
        :param networkname: string, name of this model scope
        :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
        :return: compiled tf.keras model
        """
    from src.models.Unets import create_unet, create_3d_wrapper_for_2d_unet
    if tf.distribute.get_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        print('create a new strategy')
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)), dtype=np.float32)
        input_small = inputs#[:,::2,::2,::2,:] # downsample hard, we will loose data by this, otherwise interpolate in two dimensions

        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        batch_norm = config.get('BATCH_NORMALISATION', False)
        use_upsample = config.get('USE_UPSAMPLE', 'False')
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[:ndims]
        lr = config.get('LR', 1e-4)
        indexing = config.get('INDEXING', 'ij')

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]
        outputs = ConvEncoder(activation=activation,
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
                          pad=pad)(input_small)


        # global average pooling does not accept any weights, initialisation must be done in the dense layer, if wished
        m = tensorflow.keras.layers.GlobalAveragePooling3D()(outputs)  # m.shape --> b, 512
        m = tensorflow.keras.layers.Dense(6, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                                          activation=activation)(m)

        m = eulerAnglesToRotationMatrix([tf.expand_dims(m[:, 0], axis=-1),
                                         tf.expand_dims(m[:, 1], axis=-1),
                                         tf.expand_dims(m[:, 2], axis=-1),
                                         tf.expand_dims(m[:, 3], axis=-1),
                                         tf.expand_dims(m[:, 4], axis=-1),
                                         tf.expand_dims(m[:, 5], axis=-1)])

        # transform the input image with the affine matrix (rotation + translation)
        y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0)(
            [inputs, m])
        #unet3d = create_3d_wrapper_for_2d_unet(config=config, metrics=metrics,unet_2d=unet_2d,compile_=False)
        unet3d = create_unet(config=config, metrics=metrics, networkname='trainable_3D_unet',single_model=True)
        mask = unet3d(inputs)

        # invert the matrix
        m_inv = AffineMatrixInverter(m)

        # transform each label separately by the inverse rotation/translation
        masks_labels = tf.unstack(mask, axis=-1)
        rev_mask = [nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing, ident=False, fill_value=0)(
            [tf.expand_dims(msk,axis=-1), m_inv]) for msk in masks_labels]
        rev_mask = K.concatenate(rev_mask, axis=-1)

        rev = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0)(
            [y, m_inv])
        transformed = tf.keras.layers.Activation(activation='linear', name='internal_input_repr')(y)
        rev_transformed = tf.keras.layers.Activation(activation='linear', name='cycle_warped')(rev)
        #rev_mask = Conv(mask_classes, one_by_one, activation='softmax', name='target_mask')(rev_mask)
        rev_mask = tf.keras.layers.Activation(activation='linear', name='target_mask')(rev_mask)

        model = Model(inputs=[inputs], outputs=[mask], name=networkname)
        import src.utils.my_metrics as metr
        from tensorflow.keras.losses import MSE
        model.compile(optimizer=get_optimizer(config, networkname),
                      #metrics=metrics,
                      loss={'trainable_3D_unet': metr.bce_dice_loss},
                      loss_weights={'trainable_3D_unet': 1.0})

    return model

# encoder learns euler angle and translation params, builds m, ST applies m, second ST applies the inverse of m
def create_affine_cycle_transformer(config, metrics=None, networkname='affine_cycle_transformer', unet=None):
    """
    create a trainable spatial transformer with a cycle loss function
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        input_shape = config.get('DIM', [10, 224, 224])
        inputs_ax = Input((*input_shape, config.get('IMG_CHANNELS', 1)))
        inputs_sax = Input((*input_shape, config.get('IMG_CHANNELS', 1)))
        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        batch_norm = config.get('BATCH_NORMALISATION', False)
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[:ndims]
        lr = config.get('LR', 1e-4)
        indexing = config.get('INDEXING','ij')



        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        outputs = Encoder_fn(activation=activation,
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
                               pad=pad,
                             inputs=inputs_ax)

        # direct convergence, no dense layer
        m_raw = tensorflow.keras.layers.GlobalAveragePooling3D()(outputs) # m.shape --> b, 512
        m_raw = tensorflow.keras.layers.Dense(256, kernel_initializer=kernel_init, activation=activation,name='dense1')(m_raw)
        m_raw = tensorflow.keras.layers.Dense(9, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),activation=activation, name='dense2')(m_raw)

        m = Euler2Matrix(name='ax2sax_matrix')(m_raw[ :,0:6])

        # Cycle flow - use the inverse M to transform the SAX input to AX
        ax2sax = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='ax2sax')([inputs_ax, m])
        m_inv = Inverse3DMatrix()(m)
        sax2ax= nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0, name='sax2ax')([inputs_sax, m_inv])

        import src.utils.my_metrics as metr
        if unet:
            logging.info('unet given, use it to max probability')

            # concat the rotation parameters with the second set of translation params
            m_mod = tf.keras.layers.Concatenate(axis=-1)([m_raw[:,0:3],m_raw[:,6:9]]) # rot + translation
            m_mod = Euler2Matrix(name='ax2sax_mod_matrix')(m_mod) #

            ax2sax_mod = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='ax2sax_mod_st')([inputs_ax, m_mod])
            mask_prob = UnetWrapper(unet, name='mask_prob')(ax2sax_mod) #
            m_mod_inv = Inverse3DMatrix()(m_mod) #
            mask2ax = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing, ident=False, fill_value=0, name='mask2ax')([mask_prob, m_mod_inv]) #

            outputs=[ax2sax, sax2ax,ax2sax_mod, mask_prob, mask2ax, m, m_mod] #
            losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_inplane=True, xy_shape=input_shape[-2]),
             'sax2ax': metr.loss_with_zero_mask(mask_smaller_than=0.01,weight_inplane=True, xy_shape=input_shape[-2]),
             'mask2ax': metr.max_volume_loss(min_probabillity=0.9) #
                      }
            loss_w = {'ax2sax': 20.0,
                    'sax2ax' : 10.0,
                    'mask2ax': 1.0
                      }
        else:
            outputs = [ax2sax, sax2ax, m]
            losses = {'ax2sax': metr.loss_with_zero_mask(weight_inplane=True),
             'sax2ax': metr.loss_with_zero_mask(weight_inplane=True)}
            loss_w = {'ax2sax': 1.0,
                    'sax2ax' : 1.0}

        model = Model(inputs=[inputs_ax, inputs_sax], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname),
                      loss=losses,
                      loss_weights=loss_w)

        return model


# encoder learns euler angle and translation params, builds m, ST applies m, second ST applies the inverse of m
def create_affine_cycle_transformer_test(config, metrics=None, networkname='affine_cycle_transformer', unet=None):
    """
    create a trainable spatial transformer with a cycle loss function
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        input_shape = config.get('DIM', [10, 224, 224])
        inputs_ax = Input((*input_shape, config.get('IMG_CHANNELS', 1)))
        inputs_sax = Input((*input_shape, config.get('IMG_CHANNELS', 1)))
        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        batch_norm = config.get('BATCH_NORMALISATION', False)
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[:ndims]
        lr = config.get('LR', 1e-4)
        indexing = config.get('INDEXING','ij')



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

        # direct convergence, no dense layer
        m_raw = tensorflow.keras.layers.GlobalAveragePooling3D()(enc) # m.shape --> b, 512
        m_raw = tensorflow.keras.layers.Dense(256, kernel_initializer=kernel_init, activation=activation,name='dense1')(m_raw)
        m_raw = tensorflow.keras.layers.Dense(9, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),activation=activation, name='dense2')(m_raw)
        m = Euler2Matrix(name='ax2sax_matrix')(m_raw[ :,0:6])

        # Cycle flow - use the inverse M to transform the SAX input to AX
        ax2sax = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='ax2sax')([inputs_ax, m])
        m_inv = Inverse3DMatrix()(m)
        sax2ax= nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0, name='sax2ax')([inputs_sax, m_inv])

        import src.utils.my_metrics as metr
        if unet:
            logging.info('unet given, use it to max probability')

            # concat the rotation parameters with the second set of translation params
            m_mod = tf.keras.layers.Concatenate(axis=-1)([m_raw[:,0:3],m_raw[:,6:9]]) # rot + translation
            m_mod = Euler2Matrix(name='ax2sax_mod_matrix')(m_mod) #

            ax2sax_mod = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='ax2sax_mod_st')([inputs_ax, m_mod])
            mask_prob = UnetWrapper(unet, name='mask_prob')(ax2sax_mod) #
            m_mod_inv = Inverse3DMatrix()(m_mod) #
            mask2ax = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing, ident=False, fill_value=0, name='mask2ax')([mask_prob, m_mod_inv]) #

            outputs=[ax2sax, sax2ax,ax2sax_mod, mask_prob, mask2ax, m, m_mod] #
            losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_inplane=True, xy_shape=input_shape[-2]),
             'sax2ax': metr.loss_with_zero_mask(mask_smaller_than=0.01,weight_inplane=True, xy_shape=input_shape[-2]),
             'mask2ax': metr.max_volume_loss(min_probabillity=0.9) #
                      }
            loss_w = {'ax2sax': 20.0,
                    'sax2ax' : 10.0,
                    'mask2ax': 1.0
                      }
        else:
            outputs = [ax2sax, sax2ax, m]
            losses = {'ax2sax': metr.loss_with_zero_mask(weight_inplane=True),
             'sax2ax': metr.loss_with_zero_mask(weight_inplane=True)}
            loss_w = {'ax2sax': 1.0,
                    'sax2ax' : 1.0}

        model = Model(inputs=[inputs_ax, inputs_sax], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname),
                      loss=losses,
                      loss_weights=loss_w)

        return model

# encoder learns euler angle and translation params, builds m, ST applies m, second ST applies the inverse of m
def create_affine_transformer_baseline(config, metrics=None, networkname='affine_transformer_baseline', unet=None):
    """
    create a trainable spatial transformer for AX2SAX Domain transformation
    MOdel expects an AX stack as input and returns AX2SAX, the affine transformation matrix M
    Additionally if a unet is injected, the model predicts the LV, MYO and RV on the AX2SAX transformed stack
    inverts the transformation and returns a segmentation in axial view.
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        inputs_ax = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)), dtype=np.float32)

        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        batch_norm = config.get('BATCH_NORMALISATION', False)
        use_upsample = config.get('USE_UPSAMPLE', 'False')
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[:ndims]
        lr = config.get('LR', 1e-4)
        indexing = config.get('INDEXING','ij')

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        outputs = Encoder_fn(activation=activation,
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
                               pad=pad,
                             inputs=inputs_ax)

        # direct convergence, no additional dense layers
        dense = tensorflow.keras.layers.GlobalAveragePooling3D()(outputs) # dense.shape --> b, 512
        dense = tensorflow.keras.layers.Dense(256, kernel_initializer=kernel_init, activation=activation,name='dense1')(dense)
        m_raw = tensorflow.keras.layers.Dense(6, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),activation=activation, name='dense2')(dense)

        m = Euler2Matrix()(m_raw[ :,0:6])

        # transform the source with the affine matrix (rotation + translation)
        ax2sax = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, ident=False, fill_value=0,name='ax2sax')([inputs_ax, m])

        if unet:
            # predict a mask, transform this mask back into the original orientation
            mask_prob = UnetWrapper(unet, name='mask_prob')(ax2sax)
            m_inverse = Inverse3DMatrix()(m)
            mask_prob = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing, ident=False, fill_value=0,name='sax_pred2ax')([mask_prob, m_inverse])
            outputs = [ax2sax, m, mask_prob]
        else:
            outputs = [ax2sax, m]

        losses = {'ax2sax': metr.loss_with_zero_mask(mask_smaller_than=0.01, weight_z=False, weight_inplane=True)

                  }
        loss_w = {'ax2sax': 1.0
                  }

        model = Model(inputs=[inputs_ax], outputs=outputs, name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname),
                      loss=losses,
                      loss_weights=loss_w)

        return model

# unet as encoder learns m, and ST applies m
def create_affine_transformer_with_unet(config, metrics=None, networkname='affine_transformer', single_model=True, supervision=False):
    """
    create a 2D/3D u-net for image segmentation
    :param config: Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param stacked_model: bool, if this model will be used in another model, applies softmax and compile or not
    :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)))
        input_ident = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)))
        print(inputs.shape)

        # define standard values according to convention over configuration paradigm
        metrics = [keras.metrics.binary_accuracy] if metrics is None else metrics
        activation = config.get('ACTIVATION', 'elu')
        loss_f = config.get('LOSS_FUNCTION', keras.losses.categorical_crossentropy)
        batch_norm = config.get('BATCH_NORMALISATION', False)
        use_upsample = config.get('USE_UPSAMPLE', 'False')
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        mask_classes = config.get("MASK_CLASSES", 4)
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_min', 0.3)
        drop_3 = config.get('DROPOUT_max', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[:ndims]
        lr = config.get('LR', 1e-4)
        indexing = config.get('INDEXING','ij')

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        outputs = unet(activation=activation,
                       batch_norm=batch_norm,
                       bn_first=bn_first,
                       depth=depth,
                       drop_3=drop_3,
                       dropouts=dropouts,
                       f_size=f_size,
                       filters=filters,
                       inputs=inputs,
                       kernel_init=kernel_init,
                       m_pool=m_pool,
                       ndims=ndims,
                       pad=pad,
                       use_upsample=use_upsample,
                       mask_classes=mask_classes,
                       supervision=supervision)

        # transform the results into a flow field.
        #Conv = getattr(kl, 'Conv%dD' % ndims)
        #flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
         #           kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(outputs)

        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        outputs = Conv(filters=12, kernel_size=f_size, kernel_initializer=kernel_init, padding=pad)(outputs)

        # global average pooling does not accept any weights
        m = tensorflow.keras.layers.GlobalAveragePooling3D()(outputs)  # b, 12,14,14,12 --> b,12

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([inputs, m])

        model = Model(inputs=[inputs, input_ident], outputs=[y, m], name=networkname)

        #losses.Grad('l2').loss
        import src.utils.my_metrics as metr
        model.compile(optimizer=Adam(lr=lr),
                      loss=['mse', losses.Grad('l2').loss],
                      loss_weights=[1.0, 0.01])
        return model

# ST to apply m to an volume
def create_affine_transformer_fixed(config, metrics=None, networkname='affine_transformer_fixed', fill_value=0, interp_method='linear'):
    """
    Apply a learned transformation matrix to an input image, no training possible
    :param config:  Key value pairs for image size and other network parameters
    :param metrics: list of tensorflow or keras compatible metrics
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: :return: compiled tf.keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    #tf.print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), config.get('IMG_CHANNELS', 1)), dtype=np.float32)
        input_matrix = Input((12), dtype=np.float32)
        indexing = config.get('INDEXING','ij')

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=False, fill_value=fill_value)([inputs, input_matrix])

        model = Model(inputs=[inputs, input_matrix], outputs=[y, input_matrix], name=networkname)
        model.compile(optimizer=get_optimizer(config, networkname),
                      loss=['mse', losses.Grad('l2').loss],
                      loss_weights=[1.0, 0.01])

        return model

# Downsampling part of a U-net, could be used to extract feature maps from a input 2D or 3D volume
def Encoder_fn(activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters, inputs,
             kernel_init, m_pool, ndims, pad):

        """
        unet 2d or 3d for the functional tf.keras api
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
        :param use_upsample:
        :param mask_classes:
        :return:
        """

        filters_init = filters
        encoder = list()
        decoder = list()
        dropouts = dropouts.copy()

        Conv = getattr(kl, 'Conv{}D'.format(ndims))
        one_by_one = (1, 1, 1)[-ndims:]

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
                downsampling_block(inputs=input_tensor,
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
        fully = conv_layer(inputs=input_tensor, filters=filters, f_size=f_size,
                           activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                           pad=pad, bn_first=bn_first, ndims=ndims)
        fully = Dropout(drop_3)(fully)
        fully = conv_layer(inputs=fully, filters=filters, f_size=f_size,
                           activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                           pad=pad, bn_first=bn_first, ndims=ndims)
        return fully


def inverse_affine_matrix(m):
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
def eulerAnglesToRotationMatrix(theta):
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

    #model = Model(inputs=[theta], outputs=rot_matrix, name='Affine_matrix_builder')

    return rot_matrix

def AffineMatrixInverter(m):
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