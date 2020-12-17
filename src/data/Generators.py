import concurrent.futures
import logging
import os
import platform
from concurrent.futures import as_completed
from time import time

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras

from src.data.Dataset import describe_sitk
from src.data.Preprocess import resample_3D, clip_quantile, normalise_image, grid_dissortion_2D_or_3D, \
    transform_to_binary_mask, load_masked_img, random_rotate90_2D_or_3D, \
    augmentation_compose_2D_or3D, pad_and_crop
from src.visualization.Visualize import show_2D_or_3D


class BaseGenerator(tensorflow.keras.utils.Sequence):
    """
    Base generator class
    """

    def __init__(self, x=None, y=None, config={}):
        """
        Creates a datagenerator for a list of nrrd images and a list of nrrd masks
        :param x: list of nrrd image file names
        :param y: list of nrrd mask file names
        :param config:
        """

        logging.info('Create DataGenerator')

        # create dicts for index based access to the file names in the datagenerators
        X_dict = {}
        Y_dict = {}

        if y is not None:  # return x, y
            assert (len(x) == len(y)), 'len(X) != len(Y)'

        # linux/windows cleaning
        if platform.system() == 'Linux':
            x = [os.path.normpath(x) for x in x if type(x) is str]
            y = [os.path.normpath(y) for y in y if type(y) is str]

        ids = []
        for i in range(len(x)):
            ids.append(i)
            X_dict[i] = x[i]
            Y_dict[i] = y[i]

        # override if necessary
        self.SINGLE_OUTPUT = config.get('SINGLE_OUTPUT', False)

        self.labels = Y_dict
        self.images = X_dict
        self.LIST_IDS = ids

        # if streamhandler loglevel is set to debug, print each pre-processing step
        self.DEBUG_MODE = logging.getLogger().handlers[1].level == logging.DEBUG
        # self.DEBUG_MODE = False

        # read the config, set default values if param not given
        self.SCALER = config.get('SCALER', 'MinMax')
        self.AUGMENT = config.get('AUGMENT', False)
        self.AUGMENT_GRID = config.get('AUGMENT_GRID', False)
        self.SHUFFLE = config.get('SHUFFLE', True)
        self.RESAMPLE = config.get('RESAMPLE', False)
        self.SPACING = config.get('SPACING', [1.25, 1.25])
        self.SEED = config.get('SEED', 42)
        self.DIM = config.get('DIM', [256, 256])
        self.BATCHSIZE = config.get('BATCHSIZE', 32)
        self.IMG_CHANNELS = config.get('IMG_CHANNELS', 1)
        self.MASK_VALUES = config.get('MASK_VALUES', [0, 1, 2, 3])
        self.N_CLASSES = len(self.MASK_VALUES)
        # create one worker per image & mask (batchsize) for parallel pre-processing if nothing else is defined
        self.MAX_WORKERS = config.get('GENERATOR_WORKER', self.BATCHSIZE)
        self.MAX_WORKERS = min(32, self.MAX_WORKERS)
        if self.DEBUG_MODE:
            self.MAX_WORKERS = 1  # avoid parallelism when debugging, otherwise the blots are shuffled

        if not hasattr(self, 'X_SHAPE'):
            self.X_SHAPE = np.empty((self.BATCHSIZE, *self.DIM, self.IMG_CHANNELS), dtype=np.float32)
            self.Y_SHAPE = np.empty((self.BATCHSIZE, *self.DIM, self.N_CLASSES), dtype=np.bool)

        logging.info(
            'Datagenerator created with: \n shape: {}\n spacing: {}\n batchsize: {}\n Scaler: {}\n Images: {} \n Augment_grid: {} \n Thread workers: {}'.format(
                self.DIM,
                self.SPACING,
                self.BATCHSIZE,
                self.SCALER,
                len(
                    self.images),
                self.AUGMENT_GRID,
                self.MAX_WORKERS))

        self.on_epoch_end()

        if self.AUGMENT:
            logging.info('Data will be augmented (shift,scale and rotate) with albumentation')

        else:
            logging.info('No augmentation')

    def __plot_state_if_debug__(self, img, mask, start_time, step='raw'):

        if self.DEBUG_MODE:

            try:
                logging.debug('{}:'.format(step))
                logging.debug('{:0.3f} s'.format(time() - start_time))
                describe_sitk(img)
                describe_sitk(mask)
                if self.MASKS:
                    show_2D_or_3D(img, mask)
                    plt.show()
                else:
                    show_2D_or_3D(img)
                    plt.show()
                    # maybe this crashes sometimes, but will be caught
                    show_2D_or_3D(mask)
                    plt.show()

            except Exception as e:
                logging.debug('plot image state failed: {}'.format(str(e)))

    def __len__(self):

        """
        Denotes the number of batches per epoch
        :return: number of batches
        """
        return int(np.floor(len(self.LIST_IDS) / self.BATCHSIZE))

    def __getitem__(self, index):

        """
        Generate indexes for one batch of data
        :param index:
        :return:
        """

        t0 = time()
        indexes = self.indexes[index * self.BATCHSIZE: (index + 1) * self.BATCHSIZE]

        # Find list of IDs
        list_IDs_temp = [self.LIST_IDS[k] for k in indexes]
        logging.debug('index generation: {}'.format(time() - t0))
        # Generate data
        return self.__data_generation__(list_IDs_temp)

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        :return:
        """

        self.indexes = np.arange(len(self.LIST_IDS))
        if self.SHUFFLE:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, list_IDs_temp):

        """
        Generates data containing batch_size samples

        :param list_IDs_temp:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, *dim, number_of_classes)
        """

        # Initialization

        x = np.empty_like(self.X_SHAPE)
        y = np.empty_like(self.Y_SHAPE)

        futures = set()

        # spawn one thread per worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:

            t0 = time()
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                try:
                    # keep ordering of the shuffled indexes
                    futures.add(executor.submit(self.__preprocess_one_image__, i, ID))

                except Exception as e:
                    logging.error(
                        'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.images[ID],
                                                                                           self.labels[ID]))

        for i, future in enumerate(as_completed(futures)):
            # use the indexes i to place each processed example in the batch
            # otherwise slower images will always be at the end of the batch
            # Use the ID for exception handling as reference to the file name
            try:
                x_, y_, i, ID, needed_time = future.result()
                if self.SINGLE_OUTPUT:
                    x[i,], _ = x_, y_
                else:
                    x[i,], y[i,] = x_, y_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.images[ID],
                                                                                       self.labels[ID]))

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))
        if self.SINGLE_OUTPUT:
            return x.astype(np.float32), None
        else:
            return np.array(x.astype(np.float32)), np.array(y.astype(np.float32))

    def __preprocess_one_image__(self, i, ID):
        logging.error('not implemented error')


class DataGenerator(BaseGenerator):
    """
    Yields (X, Y) / image,mask for 2D and 3D U-net training
    could be used to yield (X, None)
    """

    def __init__(self, x=None, y=None, config={}):
        self.MASKING_IMAGE = config.get('MASKING_IMAGE', False)
        self.SINGLE_OUTPUT = False
        self.MASKING_VALUES = config.get('MASKING_VALUES', [1, 2, 3])

        # how to get from image path to mask path
        # the wildcard is used to load a mask and cut the images by one or more labels
        self.REPLACE_DICT = {}
        GCN_REPLACE_WILDCARD = ('img', 'msk')
        ACDC_REPLACE_WILDCARD = ('.nii.gz', '_gt.nii.gz')

        if 'ACDC' in x[0]:
            self.REPLACE_WILDCARD = ACDC_REPLACE_WILDCARD
        else:
            self.REPLACE_WILDCARD = GCN_REPLACE_WILDCARD
        # if masks are given
        if y is not None:
            self.MASKS = True
        super(DataGenerator, self).__init__(x=x, y=y, config=config)

    def __preprocess_one_image__(self, i, ID):

        t0 = time()
        # load image
        sitk_img = load_masked_img(sitk_img_f=self.images[ID], mask=self.MASKING_IMAGE,
                                   masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        # load mask/ax2sax
        sitk_msk = load_masked_img(sitk_img_f=self.labels[ID], mask=self.MASKING_IMAGE,
                                   masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD,
                                   mask_labels=self.MASK_VALUES)

        self.__plot_state_if_debug__(sitk_img, sitk_msk, t0, 'raw')
        t1 = time()

        if self.RESAMPLE:

            # calc new size after resample image with given new spacing
            # sitk.spacing has the opposite order than np.shape and tf.shape
            # we use the numpy order z, y, x
            old_spacing_img = list(reversed(sitk_img.GetSpacing()))
            old_size_img = list(reversed(sitk_img.GetSize()))  # after reverse: z, y, x

            old_spacing_msk = list(reversed(sitk_msk.GetSpacing()))
            old_size_msk = list(reversed(sitk_msk.GetSize()))  # after reverse: z, y, x

            if sitk_img.GetDimension() == 2:
                x_s_img = (old_size_img[1] * old_spacing_img[1]) / self.SPACING[1]
                y_s_img = (old_size_img[0] * old_spacing_img[0]) / self.SPACING[0]
                new_size_img = (int(np.round(x_s_img)), int(np.round(y_s_img)))

                x_s_msk = (old_size_msk[1] * old_spacing_msk[1]) / self.SPACING[1]
                y_s_msk = (old_size_msk[0] * old_spacing_msk[0]) / self.SPACING[0]
                new_size_msk = (int(np.round(x_s_msk)), int(np.round(y_s_msk)))

            elif sitk_img.GetDimension() == 3:
                # round up
                x_s_img = np.round((old_size_img[2] * old_spacing_img[2])) / self.SPACING[2]
                y_s_img = np.round((old_size_img[1] * old_spacing_img[1])) / self.SPACING[1]
                z_s_img = np.round((old_size_img[0] * old_spacing_img[0])) / self.SPACING[0]
                new_size_img = (int(np.round(x_s_img)), int(np.round(y_s_img)), int(np.round(z_s_img)))

                x_s_msk = np.round((old_size_msk[2] * old_spacing_msk[2])) / self.SPACING[2]
                y_s_msk = np.round((old_size_msk[1] * old_spacing_msk[1])) / self.SPACING[1]
                z_s_msk = np.round((old_size_msk[0] * old_spacing_msk[0])) / self.SPACING[0]
                new_size_msk = (int(np.round(x_s_msk)), int(np.round(y_s_msk)), int(np.round(z_s_msk)))

                # we can also resize with the resamplefilter from sitk
                # this cuts the image at the bottom and right
                # new_size = self.DIM
            else:
                raise ('dimension not supported: {}'.format(sitk_img.GetDimension()))

            logging.debug('dimension: {}'.format(sitk_img.GetDimension()))
            logging.debug('Size before resample: {}'.format(sitk_img.GetSize()))

            # resample the image to given spacing and size
            sitk_img = resample_3D(sitk_img=sitk_img, size=new_size_img, spacing=list(reversed(self.SPACING)),
                                   interpolate=sitk.sitkLinear)
            if self.MASKS:  # if y is a mask, interpolate with nearest neighbor
                sitk_msk = resample_3D(sitk_img=sitk_msk, size=new_size_msk, spacing=list(reversed(self.SPACING)),
                                       interpolate=sitk.sitkNearestNeighbor)
            else:
                sitk_msk = resample_3D(sitk_img=sitk_msk, size=new_size_msk, spacing=list(reversed(self.SPACING)),
                                       interpolate=sitk.sitkLinear)

        logging.debug('Spacing after resample: {}'.format(sitk_img.GetSpacing()))
        logging.debug('Size after resample: {}'.format(sitk_img.GetSize()))

        # transform to nda for further processing
        img_nda = sitk.GetArrayFromImage(sitk_img)
        mask_nda = sitk.GetArrayFromImage(sitk_msk)

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'resampled')
        t1 = time()

        img_nda = clip_quantile(img_nda, .999)
        if not self.MASKS:
            # yields two images
            mask_nda = clip_quantile(mask_nda, .999)

        img_nda = normalise_image(img_nda, normaliser=self.SCALER)
        if not self.MASKS:  # yields the image two times for an autoencoder
            mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, '{} normalized image:'.format(self.SCALER))

        if self.AUGMENT_GRID:  # augment with grid transform from albumenation
            # apply grid augmentation
            img_nda, mask_nda = grid_dissortion_2D_or_3D(img_nda, mask_nda, probabillity=0.8, is_y_mask=self.MASKS)
            img_nda, mask_nda = random_rotate90_2D_or_3D(img_nda, mask_nda, probabillity=0.1)

            self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'grid_augmented')
            t1 = time()

        if self.AUGMENT:  # augment data with albumentation
            # use albumentation to apply random rotation scaling and shifts
            img_nda, mask_nda = augmentation_compose_2D_or3D(img_nda, mask_nda, target_dim=self.DIM, probabillity=0.8)

            self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'augmented')
            t1 = time()

        img_nda, mask_nda = map(lambda x: pad_and_crop(x, target_shape=self.DIM),
                                [img_nda, mask_nda])


        img_nda = normalise_image(img_nda, normaliser=self.SCALER)
        # transform the labels to binary channel masks
        # if masks are given, otherwise keep image as it is (for vae models, masks == False)
        if self.MASKS:
            mask_nda = transform_to_binary_mask(mask_nda, self.MASK_VALUES)
        else:  # yields two images
            mask_nda = normalise_image(mask_nda, normaliser=self.SCALER)
            mask_nda = mask_nda[..., np.newaxis]

        self.__plot_state_if_debug__(img_nda, mask_nda, t1, 'after crop')

        return img_nda[..., np.newaxis], mask_nda, i, ID, time() - t0


class CycleMotionDataGenerator(DataGenerator):
    """
    yields ([AX], [AXtoSAX, AXtoSAXtoAX, m]) for cycle motion loss
    e.g.: AX --> AXtoSAX --> AXtoSAXtoAX
    """

    def __init__(self, x=None, y=None, config={}):
        super(CycleMotionDataGenerator, self).__init__(x=x, y=y, config=config)

        # change this to support different folder names, this is hardcoded to not break with the BaseGenerator api
        assert 'AX_3D' in x[0]
        assert 'AX_to_SAX_3D' in y[0]

        self.X_SHAPE = np.empty((self.BATCHSIZE, *self.DIM, self.IMG_CHANNELS), dtype=np.float32)
        self.MASKS = False

    def __data_generation__(self, list_IDs_temp):

        """
        Generates data containing batch_size samples

        :param list_IDs_temp:
        :return: X : (batchsize, *dim, n_channels), Y : (batchsize, *dim, number_of_classes)
        """

        # Initialization
        x = np.empty_like(self.X_SHAPE)  # ax
        y = np.empty_like(self.X_SHAPE)  # axtosax
        x2 = np.empty_like(self.X_SHAPE)  # sax
        y2 = np.empty_like(self.X_SHAPE)  # saxtoax
        empty = np.empty_like(self.X_SHAPE)  # saxtoax modified
        futures = set()

        # spawn one thread per worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:

            t0 = time()
            # Generate data
            for i, ID in enumerate(list_IDs_temp):

                try:
                    # keep ordering of the shuffled indexes
                    futures.add(executor.submit(self.__preprocess_one_image__, i, ID))

                except Exception as e:
                    logging.error(
                        'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.images[ID],
                                                                                           self.labels[ID]))

        for i, future in enumerate(as_completed(futures)):
            # use the indexes to order the batch
            # otherwise slower images will always be at the end of the batch
            try:
                x_, y_, x2_, y2_, i, ID, needed_time = future.result()
                x[i,], y[i,], x2[i,], y2[i,] = x_, y_, x2_, y2_
                logging.debug('img finished after {:0.3f} sec.'.format(needed_time))
            except Exception as e:
                logging.error(
                    'Exception {} in datagenerator with: image: {} or mask: {}'.format(str(e), self.images[ID],
                                                                                       self.labels[ID]))

        logging.debug('Batchsize: {} preprocessing took: {:0.3f} sec'.format(self.BATCHSIZE, time() - t0))

        ident = np.eye(4, dtype=np.float32)[:3, :]
        return tuple([[x.astype(np.float32), x2.astype(np.float32)],
                      [y.astype(np.float32), y2.astype(np.float32), empty.astype(np.float32), empty.astype(np.float32),
                       empty.astype(np.float32), ident, ident]])

    def __preprocess_one_image__(self, i, ID):

        t0 = time()
        # use the load_masked_img wrapper to enable masking of the images, not necessary for the TMI paper
        # load image
        sitk_ax = load_masked_img(sitk_img_f=self.images[ID], mask=self.MASKING_IMAGE,
                                  masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        # load ax2sax
        sitk_ax2sax = load_masked_img(sitk_img_f=self.labels[ID], mask=self.MASKING_IMAGE,
                                      masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        # load sax
        sitk_sax = load_masked_img(sitk_img_f=self.images[ID].replace('AX_3D', 'SAX_3D'),
                                   masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        # load sax2ax
        sitk_sax2ax = load_masked_img(sitk_img_f=self.images[ID].replace('AX_3D', 'SAX_to_AX_3D'),
                                      masking_values=self.MASKING_VALUES, replace=self.REPLACE_WILDCARD)
        # test to train on ax,sax image pairs without ax2sax transformation

        self.__plot_state_if_debug__(sitk_ax, sitk_ax2sax, t0, 'raw')
        t1 = time()

        if self.RESAMPLE:

            # calc new size after resample image with given new spacing
            # sitk.spacing has the opposite order than np.shape and tf.shape
            # we use the numpy order z, y, x
            old_spacing_img = list(reversed(sitk_ax.GetSpacing()))
            old_size_img = list(reversed(sitk_ax.GetSize()))  # after reverse: z, y, x

            old_spacing_msk = list(reversed(sitk_ax2sax.GetSpacing()))
            old_size_msk = list(reversed(sitk_ax2sax.GetSize()))  # after reverse: z, y, x

            old_spacing_sax3d = list(reversed(sitk_sax.GetSpacing()))
            old_size_sax3d = list(reversed(sitk_sax.GetSize()))  # after reverse: z, y, x

            old_spacing_saxtoax3d = list(reversed(sitk_sax2ax.GetSpacing()))
            old_size_saxtoax3d = list(reversed(sitk_sax2ax.GetSize()))  # after reverse: z, y, x

            if sitk_ax.GetDimension() == 3:
                # round up
                x_s_img = (old_size_img[2] * old_spacing_img[2]) / self.SPACING[2]
                y_s_img = (old_size_img[1] * old_spacing_img[1]) / self.SPACING[1]
                z_s_img = (old_size_img[0] * old_spacing_img[0]) / self.SPACING[0]
                new_size_img = (int(np.round(x_s_img)), int(np.round(y_s_img)), int(np.round(z_s_img)))

                x_s_msk = (old_size_msk[2] * old_spacing_msk[2]) / self.SPACING[2]
                y_s_msk = (old_size_msk[1] * old_spacing_msk[1]) / self.SPACING[1]
                z_s_msk = (old_size_msk[0] * old_spacing_msk[0]) / self.SPACING[0]
                new_size_msk = (int(np.round(x_s_msk)), int(np.round(y_s_msk)), int(np.round(z_s_msk)))

                x_s_sax3d = (old_size_sax3d[2] * old_spacing_sax3d[2]) / self.SPACING[2]
                y_s_sax3d = (old_size_sax3d[1] * old_spacing_sax3d[1]) / self.SPACING[1]
                z_s_sax3d = (old_size_sax3d[0] * old_spacing_sax3d[0]) / self.SPACING[0]
                new_size_sax3d = (int(np.round(x_s_sax3d)), int(np.round(y_s_sax3d)), int(np.round(z_s_sax3d)))

                x_s_saxtoax3d = (old_size_saxtoax3d[2] * old_spacing_saxtoax3d[2]) / self.SPACING[2]
                y_s_saxtoax3d = (old_size_saxtoax3d[1] * old_spacing_saxtoax3d[1]) / self.SPACING[1]
                z_s_saxtoax3d = (old_size_saxtoax3d[0] * old_spacing_saxtoax3d[0]) / self.SPACING[0]
                new_size_saxtoax3d = (
                    int(np.round(x_s_saxtoax3d)), int(np.round(y_s_saxtoax3d)), int(np.round(z_s_saxtoax3d)))


            else:
                raise NotImplementedError('dimension not supported: {}'.format(sitk_ax.GetDimension()))

            logging.debug('dimension: {}'.format(sitk_ax.GetDimension()))
            logging.debug('Size before resample: {}'.format(sitk_ax.GetSize()))

            # resample the image to given spacing increase/decrease the size according to new spacing
            sitk_ax = resample_3D(sitk_img=sitk_ax, size=new_size_img, spacing=list(reversed(self.SPACING)),
                                  interpolate=sitk.sitkLinear)

            # resample the image to given spacing and size
            sitk_sax = resample_3D(sitk_img=sitk_sax, size=new_size_sax3d, spacing=list(reversed(self.SPACING)),
                                   interpolate=sitk.sitkLinear)

            # resample the image to given spacing and size
            sitk_sax2ax = resample_3D(sitk_img=sitk_sax2ax, size=new_size_saxtoax3d,
                                      spacing=list(reversed(self.SPACING)),
                                      interpolate=sitk.sitkLinear)

            sitk_ax2sax = resample_3D(sitk_img=sitk_ax2sax, size=new_size_msk, spacing=list(reversed(self.SPACING)),
                                      interpolate=sitk.sitkLinear)

        logging.debug('Spacing after resample: {}'.format(sitk_ax.GetSpacing()))
        logging.debug('Size after resample: {}'.format(sitk_ax.GetSize()))

        # transform to nda for further processing
        nda_ax = sitk.GetArrayFromImage(sitk_ax)
        nda_sax = sitk.GetArrayFromImage(sitk_sax)
        nda_sax2ax = sitk.GetArrayFromImage(sitk_sax2ax)
        nda_ax2sax = sitk.GetArrayFromImage(sitk_ax2sax)

        self.__plot_state_if_debug__(nda_ax, t1, 'resampled')

        if self.AUGMENT_GRID:  # augment with grid transform from albumenation
            # TODO: implement augmentation, remember params and apply to the other images
            raise NotImplementedError

        if self.AUGMENT:  # augment data with albumentation
            # TODO: implement augmentation, remember params and apply to the other images
            raise NotImplementedError

        nda_ax, nda_ax2sax, nda_sax, nda_sax2ax = map(lambda x: clip_quantile(x, .999),
                                                      [nda_ax, nda_ax2sax, nda_sax, nda_sax2ax])

        nda_ax, nda_ax2sax, nda_sax, nda_sax2ax = map(lambda x: normalise_image(x, normaliser=self.SCALER),
                                                      [nda_ax, nda_ax2sax, nda_sax, nda_sax2ax])

        nda_ax, nda_ax2sax, nda_sax, nda_sax2ax = map(lambda x: pad_and_crop(x, target_shape=self.DIM),
                                                      [nda_ax, nda_ax2sax, nda_sax, nda_sax2ax])

        self.__plot_state_if_debug__(nda_ax, nda_ax2sax, t1, 'crop and pad')

        # clipping and normalise after cropping
        nda_ax, nda_ax2sax, nda_sax, nda_sax2ax = map(lambda x: normalise_image(x, normaliser=self.SCALER),
                                                      [nda_ax, nda_ax2sax, nda_sax, nda_sax2ax])

        self.__plot_state_if_debug__(nda_ax, nda_ax2sax, t1, 'clipped and normalized')

        return nda_ax[..., np.newaxis], \
               nda_sax[..., np.newaxis], \
               nda_sax[..., np.newaxis], \
               nda_ax[..., np.newaxis], \
               i, ID, time() - t0

"""        return nda_ax[..., np.newaxis], \
               nda_ax2sax[..., np.newaxis], \
               nda_sax[..., np.newaxis], \
               nda_sax2ax[..., np.newaxis], \
               i, ID, time() - t0"""
