import logging
import os
import sys

import SimpleITK as sitk
import cv2
import numpy as np
from albumentations import GridDistortion, RandomRotate90, Compose, RandomBrightnessContrast, ElasticTransform
from albumentations import ShiftScaleRotate
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from src.data.Dataset import get_metadata_maybe


def load_masked_img(sitk_img_f, mask=False, masking_values=None, replace=('img', 'msk'), mask_labels=None):
    """
    opens an sitk image, mask it if mask = True and masking values are given
    """

    if mask_labels is None:
        mask_labels = [0, 1, 2, 3]
    if masking_values is None:
        masking_values = [1, 2, 3]
    sitk_mask_f = sitk_img_f.replace(replace[0], replace[1])

    assert os.path.isfile(sitk_img_f), 'no valid image: {}'.format(sitk_img_f)

    img_original = sitk.ReadImage(sitk_img_f, sitk.sitkFloat32)

    if mask:
        msk_original = sitk.ReadImage(sitk_mask_f, sitk.sitkFloat32)

        img_nda = sitk.GetArrayFromImage(img_original)
        msk_nda = transform_to_binary_mask(sitk.GetArrayFromImage(msk_original), mask_values=mask_labels)

        # mask by different labels, sum up all masked channels
        temp = np.zeros(img_nda.shape)
        for c in masking_values:
            # mask by different labels, sum up all masked channels
            temp += img_nda * msk_nda[..., c].astype(np.bool)
        sitk_img = sitk.GetImageFromArray(temp)

        # copy metadata
        for tag in img_original.GetMetaDataKeys():
            value = get_metadata_maybe(img_original, tag)
            sitk_img.SetMetaData(tag, value)
        sitk_img.SetSpacing(img_original.GetSpacing())
        sitk_img.SetOrigin(img_original.GetOrigin())

        img_original = sitk_img

    return img_original

def resample_3D(sitk_img, size=(256, 256, 12), spacing=(1.25, 1.25, 8), interpolate=sitk.sitkNearestNeighbor):
    """
    resamples an 3D sitk image or numpy ndarray to a new size with respect to the giving spacing
    This method expects size and spacing in sitk format: x, y, z
    :param sitk_img:
    :param size:
    :param spacing:
    :param interpolate:
    :return: returns the same datatype as submitted, either sitk.image or numpy.ndarray
    """

    return_sitk = True

    if isinstance(sitk_img, np.ndarray):
        return_sitk = False
        sitk_img = sitk.GetImageFromArray(sitk_img)

    assert (isinstance(sitk_img, sitk.Image)), 'wrong image type: {}'.format(type(sitk_img))

    # if len(size) == 3 and size[0] < size[-1]: # 3D data, but numpy shape and size, reverse order for sitk
    # bug if z is lonnger than x or y
    #    size = tuple(reversed(size))
    #    spacing = tuple(reversed(spacing))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetInterpolator(interpolate)
    resampled = resampler.Execute(sitk_img)

    # return the same data type as input datatype
    if return_sitk:
        return resampled
    else:
        return sitk.GetArrayFromImage(resampled)


def random_rotate90_2D_or_3D(img, mask, probabillity=0.8):
    logging.debug('random rotate for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim is 2:

        aug = RandomRotate90(p=probabillity)

        params = aug.get_params()
        image_aug = aug.apply(img, **params)
        mask_aug = aug.apply(mask, interpolation=cv2.INTER_NEAREST, **params)

        # apply shift-scale and rotation augmentation on 2d data
        augmented['image'] = image_aug
        augmented['mask'] = mask_aug

    elif img.ndim is 3:
        # apply shif-scale and rotation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = RandomRotate90(p=probabillity)
        params = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z, ...], interpolation=cv2.INTER_LINEAR, factor=1, **params))
            masks.append(aug.apply(mask[z, ...], interpolation=cv2.INTER_NEAREST, factor=1, **params))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']


def augmentation_compose_2D_or3D(img, mask, target_dim, probabillity=1, spatial_transforms=True):
    # logging.debug('random rotate for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    targets = {}
    data = {}
    img_placeholder = 'image'
    mask_placeholder = 'mask'

    if img.ndim is 2:
        data = {"image": img, "mask": mask}

    if img.ndim is 3:
        data = {"image": img[0], "mask": mask[0]}

        for z in range(img.shape[0]):
            data['{}{}'.format(img_placeholder, z)] = img[z, ...]
            data['{}{}'.format(mask_placeholder, z)] = mask[z, ...]
            targets['{}{}'.format(img_placeholder, z)] = 'image'
            targets['{}{}'.format(mask_placeholder, z)] = 'mask'
    if spatial_transforms:
        aug = _create_aug_compose(p=probabillity, pad=max(img.shape[-2:]), target_dim=target_dim, targets=targets)
    else:
        aug = _create_aug_compose_only_brightness(p=probabillity, pad=max(img.shape[-2:]), target_dim=target_dim,
                                                  targets=targets)

    augmented = aug(**data)

    if img.ndim is 3:
        images = []
        masks = []
        for z in range(img.shape[0]):
            images.append(augmented['{}{}'.format(img_placeholder, z)])
            masks.append(augmented['{}{}'.format(mask_placeholder, z)])
        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    return augmented['image'], augmented['mask']


def _create_aug_compose_only_brightness(p=1, pad=256, target_dim=(256, 256), targets=None):
    if targets is None:
        targets = {}
    return Compose([
        # RandomRotate90(p=0.3),
        # Flip(0.1),
        # Transpose(p=0.1),
        # ShiftScaleRotate(p=0.8, rotate_limit=0,shift_limit=0.025, scale_limit=0.1,value=0, border_mode=cv2.BORDER_CONSTANT),
        # GridDistortion(p=0.8, value=0,border_mode=cv2.BORDER_CONSTANT),
        # PadIfNeeded(min_height=pad, min_width=pad, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
        # CenterCrop(height=target_dim[0], width=target_dim[1], p=1),
        # ToFloat(max_value=100,p=1),
        # HueSaturationValue(p=1)
        RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, always_apply=True)
        # RandomBrightness(limit=0.1,p=1),
        # GaussNoise(mean=image.mean(),p=1)
        # OneOf([
        # OpticalDistortion(p=1),
        # GridDistortion(p=0.1)
        # ], p=1),
    ], p=p,
        additional_targets=targets)


def _create_aug_compose(p=1, pad=256, target_dim=(256, 256), targets=None):
    if targets is None:
        targets = {}
    return Compose([
        RandomRotate90(p=0.3),
        # Flip(0.1),
        # Transpose(p=0.1),
        ShiftScaleRotate(p=0.8, rotate_limit=0, shift_limit=0.025, scale_limit=0.1, value=0,
                         border_mode=cv2.BORDER_CONSTANT),
        GridDistortion(p=0.8, value=0, border_mode=cv2.BORDER_CONSTANT),
        # PadIfNeeded(min_height=pad, min_width=pad, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
        # CenterCrop(height=target_dim[0], width=target_dim[1], p=1),
        # ToFloat(max_value=100,p=1),
        # HueSaturationValue(p=1)
        RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, always_apply=True)
        # RandomBrightness(limit=0.1,p=1),
        # GaussNoise(mean=image.mean(),p=1)
        # OneOf([
        # OpticalDistortion(p=1),
        # GridDistortion(p=0.1)
        # ], p=1),
    ], p=p,
        additional_targets=targets)


def random_rotate_2D_or_3D(img, mask, probabillity=0.8, shift_limit=0.0625, scale_limit=0.0, rotate_limit=0):
    """
    Rotate, shift and scale an image within a given range
    :param img: numpy.ndarray
    :param mask: numpy.ndarray
    :param probabillity: float, will be interpreted as %-value
    :param shift_limit:
    :param scale_limit:
    :param rotate_limit:
    :return:
    """

    logging.debug('random rotate for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim is 2:

        aug = ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
                               border_mode=cv2.BORDER_REFLECT_101, p=probabillity)

        params = aug.get_params()
        image_aug = aug.apply(img, interpolation=cv2.INTER_LINEAR, **params)
        mask_aug = aug.apply(mask, interpolation=cv2.INTER_NEAREST, **params)

        # apply shift-scale and rotation augmentation on 2d data
        augmented['image'] = image_aug
        augmented['mask'] = mask_aug

    elif img.ndim is 3:
        # apply shif-scale and rotation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
                               border_mode=cv2.BORDER_REFLECT_101, p=probabillity)
        params = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z, ...], interpolation=cv2.INTER_LINEAR, **params))
            masks.append(aug.apply(mask[z, ...], interpolation=cv2.INTER_NEAREST, **params))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']


def grid_dissortion_2D_or_3D(img, mask, probabillity=0.8, border_mode=cv2.BORDER_REFLECT_101, is_y_mask=True):
    """
    Apply grid dissortion
    :param img:
    :param mask:
    :return:
    """
    logging.debug('grid dissortion for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if is_y_mask:
        y_interpolation = cv2.INTER_NEAREST
    else:
        y_interpolation = cv2.INTER_LINEAR

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim is 2:
        # apply grid augmentation on 2d data
        aug = GridDistortion(p=probabillity, border_mode=border_mode, mask_value=0, value=0)
        if is_y_mask:
            augmented = aug(image=img, mask=mask)
        else:
            steps = aug.get_params()
            augmented['image'] = aug.apply(img, steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_LINEAR)
            augmented['mask'] = aug.apply(mask, steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_LINEAR)
    elif img.ndim is 3:

        # apply grid augmentation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = GridDistortion(p=probabillity, border_mode=border_mode)
        steps = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z, ...], steps['stepsx'], steps['stepsy'], interpolation=y_interpolation))
            masks.append(aug.apply(mask[z, ...], steps['stepsx'], steps['stepsy'], interpolation=y_interpolation))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']


def elastic_transoform_2D_or_3D(img, mask, probabillity=0.8):
    """
    Apply grid dissortion
    :param img:
    :param mask:
    :return:
    """
    logging.debug('grid dissortion for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim is 2:

        # apply grid augmentation on 2d data
        aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.09, alpha_affine=120 * 0.08,
                               border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
        augmented = aug(image=img, mask=mask)

    elif img.ndim is 3:

        # apply grid augmentation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.09, alpha_affine=120 * 0.08,
                               border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
        steps = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z, ...], steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_LINEAR))
            masks.append(aug.apply(mask[z, ...], steps['stepsx'], steps['stepsy'], interpolation=cv2.INTER_NEAREST))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']


def pad_and_crop(ndarray, target_shape=(10, 10, 10)):
    """
    Center pad and crop a np.ndarray with any shape to a given target shape
    Parameters
    In this implementation the pad and crop is invertible, ceil and round respects uneven shapes
    pad = floor(x),floor(x)+1
    crop = floor(x)+1, floor(x)
    ----------
    ndarray : numpy.ndarray - of any shape
    target_shape : tuple - must have the same length as ndarray.ndim

    Returns np.ndarray with each axis either pad or crop
    -------

    """
    empty = np.zeros(target_shape)
    target_shape = np.array(target_shape)
    logging.debug('input shape, crop_and_pad: {}'.format(ndarray.shape))
    logging.debug('target shape, crop_and_pad: {}'.format(target_shape))

    diff = ndarray.shape - target_shape

    # divide into summands to work with odd numbers
    # take the same numbers for left or right padding/cropping if the difference is dividable by 2
    # else take floor(x),floor(x)+1 for PAD (diff<0)
    # else take floor(x)+1, floor(x) for CROP (diff>0)
    d = list(
        (int(x // 2), int(x // 2)) if x % 2 == 0 else (int(np.floor(x / 2)), int(np.floor(x / 2) + 1)) if x < 0 else (
            int(np.floor(x / 2) + 1), int(np.floor(x / 2))) for x in diff)
    # replace the second slice parameter if it is None, which slice until end of ndarray
    d = list((abs(x), abs(y)) if y != 0 else (abs(x), None) for x, y in d)
    # create a bool list, negative numbers --> pad, else --> crop
    pad_bool = diff < 0
    crop_bool = diff > 0

    # create one slice obj for cropping and one for padding
    pad = list(i if b else (None, None) for i, b in zip(d, pad_bool))
    crop = list(i if b else (None, None) for i, b in zip(d, crop_bool))

    # Create one tuple of slice calls per pad/crop
    # crop or pad from dif:-dif if second param not None, else replace by None to slice until the end
    # slice params: slice(start,end,steps)
    pad = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in pad)
    crop = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in crop)

    # crop and pad in one step
    empty[pad] = ndarray[crop]
    return empty


def transform_to_binary_mask(mask_nda, mask_values=None):
    """
    Transform from a value-based representation to a binary channel based representation
    :param mask_nda:
    :param mask_values:
    :return:
    """
    # transform the labels to binary channel masks

    if mask_values is None:
        mask_values = [0, 1, 2, 3]
    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype=np.bool)
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask


def from_channel_to_flat(binary_mask, start_c=0):
    """
    Transform a tensor or numpy nda from a channel-wise (one channel per label) representation
    to a value-based representation
    :param binary_mask:
    :return:
    """
    # convert to bool nda to allow later indexing
    binary_mask = binary_mask >= 0.5

    # reduce the shape by 1
    temp = np.zeros(binary_mask.shape[:-1], dtype=np.uint8)
    labels = list(range(binary_mask.shape[-1]))
    # swap the last two elements, from either 0-4 or 0-3
    labels[-1], labels[-2] = labels[-2], labels[-1]

    # order: RV, LV, MYo
    for c in labels:
    #for c in range(binary_mask.shape[-1]):
        temp[binary_mask[..., c]] = c + start_c
    return temp


def clip_quantile(img_nda, upper_quantile=.999, lower_boundary=0):
    """
    clip to values between 0 and .999 quantile
    :param img_nda:
    :param upper_quantile:
    :return:
    """
    # calc the background average voxel value
    #backround_size = 5
    #background_slice = tuple(slice(0,backround_size) for _ in range(len(img_nda.shape)))
    #background = img_nda[background_slice].max()

    ninenine_q = np.quantile(img_nda.flatten(), upper_quantile, overwrite_input=False)
    return np.clip(img_nda, lower_boundary, ninenine_q)


def normalise_image(img_nda, normaliser='minmax'):
    """
    Normalise Images to a given range,
    normaliser string repr for scaler, possible values: 'MinMax', 'Standard' and 'Robust'
    if no normalising method is defined use MinMax normalising
    :param img_nda:
    :param normaliser:
    :return:
    """
    # ignore case
    normaliser = normaliser.lower()

    if normaliser == 'standard':
        return StandardScaler(copy=False, with_mean=True, with_std=True).fit_transform(img_nda)
    elif normaliser == 'robust':
        return RobustScaler(copy=False, quantile_range=(0.0, 95.0), with_centering=True,
                            with_scaling=True).fit_transform(img_nda)
    else:
        return (img_nda - img_nda.min()) / (img_nda.max() - img_nda.min() + sys.float_info.epsilon)


def get_metadata_maybe(key, sitk_img, default='not_found'):
    try:
        value = sitk_img.GetMetaData(key)
    except Exception as e:
        # logging.debug('key not found: {}, {}'.format(key, e))
        value = default
        pass

    return value


import matplotlib.pyplot as plt


def show_3D(sitk_img):
    img_array = sitk.GetArrayFromImage(sitk_img)
    max_depth = img_array.shape[0]
    fig = plt.figure(figsize=(30, max_depth))
    for i in range(max_depth):
        fig.add_subplot(1, max_depth, i + 1)
        plt.imshow(img_array[i, :, :])
        plt.axis('off')


def Image3D_from_Time(sitk_img, time):
    img_array = sitk.GetArrayFromImage(sitk_img)
    sitk_array_3D = img_array[time - 1, :, :, :]
    new_sitk_img = sitk.GetImageFromArray(sitk_array_3D)
    return new_sitk_img


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# returns only one label
def get_single_label_img(sitk_img, label):
    img_array = sitk.GetArrayFromImage(sitk_img)
    label_array = (img_array == label).astype(int)
    label_array = label_array * 100
    label_img = sitk.GetImageFromArray(label_array)
    label_img.CopyInformation(sitk_img)
    return label_img


# returns resampled img1 by referncing img2
def resample_img(sitk_img1, sitk_img2):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img2)
    resampled_img = resampler.Execute(sitk_img1)
    return resampled_img


# returns resampled_img back to normal label
def convert_back_to_label(sitk_img, label):
    img_array = sitk.GetArrayFromImage(sitk_img)
    label_array = (img_array > 25).astype(int)
    label_array = label_array * label
    label_img = sitk.GetImageFromArray(label_array)
    label_img.CopyInformation(sitk_img)
    return label_img


# returns resampled_img back to normal label with percentage
def advanced_convert_back_to_label(sitk_img, label, percentage):
    img_array = sitk.GetArrayFromImage(sitk_img)
    percent_array = img_array.flatten()
    percent_array = percent_array[percent_array > 0]
    try:
        threshold = - np.percentile(percent_array * -1, q=percentage)
    except IndexError:
        print("IndexError")
        threshold = 50
    label_array = (img_array >= threshold).astype(int)
    label_array = label_array * label
    label_img = sitk.GetImageFromArray(label_array)
    label_img.CopyInformation(sitk_img)
    return label_img


# returns all labels added and with priority from label 3 to 1
def add_labels(label_array1, label_array2, label_array3):
    final_array = label_array3
    temp_array = final_array + label_array2
    # if you add the label, there is only a 2 if there was a 0 there before, so now oferwriting
    temp_array = (temp_array == 2).astype(int)
    temp_array = temp_array * 2
    final_array = final_array + temp_array
    temp_array = final_array + label_array1
    temp_array = (temp_array == 1).astype(int)
    temp_array = temp_array * 1
    final_array = final_array + temp_array
    return final_array


def add_labels_max_thres(label_array1, label_array2, label_array3, threshold):
    max_label_1_2_array = np.maximum(label_array1, label_array2)
    max_label_2_3_array = np.maximum(label_array2, label_array3)
    max_label_array = np.maximum(max_label_1_2_array, max_label_2_3_array)
    thres_label_array = (max_label_array >= threshold).astype(int)
    max_label_array = thres_label_array * max_label_array

    max_label_array1 = label_array1 * np.equal(max_label_array, label_array1).astype(int)
    # max_label_array1 =  (max_label_array1 > threshold).astype(int) * 1
    max_label_array1 = max_label_array1.astype(bool).astype(int) * 1
    max_label_array2 = label_array2 * np.equal(max_label_array, label_array2).astype(int)
    # max_label_array2 =  (max_label_array2 > threshold).astype(int) * 2
    max_label_array2 = max_label_array2.astype(bool).astype(int) * 2
    max_label_array3 = label_array3 * np.equal(max_label_array, label_array3).astype(int)
    # max_label_array3 =  (max_label_array3 > threshold).astype(int) * 3
    max_label_array3 = max_label_array3.astype(bool).astype(int) * 3

    return add_labels(max_label_array1, max_label_array2, max_label_array3)


# reutrns resampled img1 by referencing img2
def resample_label_img(sitk_img1, sitk_img2):
    label1_img1 = get_single_label_img(sitk_img1, 1)
    label2_img1 = get_single_label_img(sitk_img1, 2)
    label3_img1 = get_single_label_img(sitk_img1, 3)

    resampled_label1 = resample_img(label1_img1, sitk_img2)
    resampled_label2 = resample_img(label2_img1, sitk_img2)
    resampled_label3 = resample_img(label3_img1, sitk_img2)
    resampled_label1 = convert_back_to_label(resampled_label1, 1)
    resampled_label2 = convert_back_to_label(resampled_label2, 2)
    resampled_label3 = convert_back_to_label(resampled_label3, 3)

    resampled_array1 = sitk.GetArrayFromImage(resampled_label1)
    resampled_array2 = sitk.GetArrayFromImage(resampled_label2)
    resampled_array3 = sitk.GetArrayFromImage(resampled_label3)
    resampled_array = add_labels(resampled_array1, resampled_array2, resampled_array3)
    resampled_img = sitk.GetImageFromArray(resampled_array)
    resampled_img.CopyInformation(resampled_label1)
    return resampled_img


# reutrns resampled img1 by referencing img2 with percentage
def percentage_resample_label_img(sitk_img1, sitk_img2, percentage):
    label1_img1 = get_single_label_img(sitk_img1, 1)
    label2_img1 = get_single_label_img(sitk_img1, 2)
    label3_img1 = get_single_label_img(sitk_img1, 3)

    resampled_label1 = resample_img(label1_img1, sitk_img2)
    resampled_label2 = resample_img(label2_img1, sitk_img2)
    resampled_label3 = resample_img(label3_img1, sitk_img2)
    resampled_label1 = advanced_convert_back_to_label(resampled_label1, 1, percentage)
    resampled_label2 = advanced_convert_back_to_label(resampled_label2, 2, percentage)
    resampled_label3 = advanced_convert_back_to_label(resampled_label3, 3, percentage)

    resampled_array1 = sitk.GetArrayFromImage(resampled_label1)
    resampled_array2 = sitk.GetArrayFromImage(resampled_label2)
    resampled_array3 = sitk.GetArrayFromImage(resampled_label3)
    resampled_array = add_labels(resampled_array1, resampled_array2, resampled_array3)
    resampled_img = sitk.GetImageFromArray(resampled_array)
    resampled_img.CopyInformation(resampled_label1)
    return resampled_img


# reutrns resampled img1 by referencing img2 with percentage
def max_thres_resample_label_img(sitk_img1, sitk_img2, threshold):
    label1_img1 = get_single_label_img(sitk_img1, 1)
    label2_img1 = get_single_label_img(sitk_img1, 2)
    label3_img1 = get_single_label_img(sitk_img1, 3)

    resampled_label1 = resample_img(label1_img1, sitk_img2)
    resampled_label2 = resample_img(label2_img1, sitk_img2)
    resampled_label3 = resample_img(label3_img1, sitk_img2)

    resampled_array1 = sitk.GetArrayFromImage(resampled_label1)
    resampled_array2 = sitk.GetArrayFromImage(resampled_label2)
    resampled_array3 = sitk.GetArrayFromImage(resampled_label3)

    resampled_array = add_labels_max_thres(resampled_array1, resampled_array2, resampled_array3, threshold)
    resampled_img = sitk.GetImageFromArray(resampled_array)
    resampled_img.CopyInformation(resampled_label1)
    return resampled_img


# returns resampled img1 by referencing img2 with percentage.
def max_thres_resample2_label_img(sitk_img1, sitk_img2, threshold):
    label1_img1 = get_single_label_img(sitk_img1, 1)
    label2_img1 = get_single_label_img(sitk_img1, 2)
    label3_img1 = get_single_label_img(sitk_img1, 3)

    resampled_label1 = resample_direcion_origin_spacing(label1_img1, sitk_img2, interpolate=sitk.sitkLinear)
    resampled_label2 = resample_direcion_origin_spacing(label2_img1, sitk_img2, interpolate=sitk.sitkLinear)
    resampled_label3 = resample_direcion_origin_spacing(label3_img1, sitk_img2, interpolate=sitk.sitkLinear)

    resampled_array1 = sitk.GetArrayFromImage(resampled_label1)
    resampled_array2 = sitk.GetArrayFromImage(resampled_label2)
    resampled_array3 = sitk.GetArrayFromImage(resampled_label3)

    resampled_array = add_labels_max_thres(resampled_array1, resampled_array2, resampled_array3, threshold)
    resampled_img = sitk.GetImageFromArray(resampled_array)
    resampled_img = sitk.Cast(resampled_img, sitk.sitkUInt8)
    resampled_img.CopyInformation(resampled_label1)
    return resampled_img


def resample_direcion_origin_spacing(sitk_img, reference_sitk, interpolate=sitk.sitkLinear):
    """
    Resample a sitk img, copy direction, origin and spacing of the reference image
    Keep the size (resolution) of the target image
    :param sitk_img: sitk.Image
    :param reference_sitk: sitk.Image
    :param interpolate: fn
    :return:
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_sitk.GetSpacing())
    resampler.SetInterpolator(interpolate)
    resampler.SetOutputDirection(reference_sitk.GetDirection())
    resampler.SetOutputOrigin(reference_sitk.GetOrigin())
    resampler.SetSize(reference_sitk.GetSize())
    resampled = resampler.Execute(sitk_img)
    # copy metadata
    for key in reference_sitk.GetMetaDataKeys():
        resampled.SetMetaData(key, get_metadata_maybe(reference_sitk, key))
    return resampled


# returns resampled img1 by referencing img2 with percentage, modified by sven !!!
def max_thres_resample2_iso_label_img(sitk_img1, threshold, spacing_=(1.5, 1.5, 1.5), file_path='temp.nrrd',
                                      interpol=sitk.sitkLinear):
    from src.visualization.Visualize import show_2D_or_3D
    import numpy as np
    import scipy

    debug = False

    label1_img1 = get_single_label_img(sitk_img1, 1)
    label2_img1 = get_single_label_img(sitk_img1, 2)
    label3_img1 = get_single_label_img(sitk_img1, 3)

    resampled_label1, _ = transform_to_isotrop_voxels(label1_img1, interpolate=interpol, spacing_=spacing_)
    resampled_label2, _ = transform_to_isotrop_voxels(label2_img1, interpolate=interpol, spacing_=spacing_)
    resampled_label3, _ = transform_to_isotrop_voxels(label3_img1, interpolate=interpol, spacing_=spacing_)

    resampled_array1 = sitk.GetArrayFromImage(resampled_label1)
    resampled_array2 = sitk.GetArrayFromImage(resampled_label2)
    resampled_array3 = sitk.GetArrayFromImage(resampled_label3)

    kernel = np.ones((5, 5, 7), np.uint8)
    kernel_small = np.ones((3, 3, 3), np.uint8)
    # maybe use a bigger/smaller kernel?

    # close holes, smooth in z,x,y
    resampled_array1 = scipy.ndimage.morphology.grey_closing(resampled_array1, structure=kernel, mode='constant')
    resampled_array2 = scipy.ndimage.morphology.grey_closing(resampled_array2, structure=kernel, mode='constant')
    resampled_array3 = scipy.ndimage.morphology.grey_closing(resampled_array3, structure=kernel, mode='constant')

    resampled_array1 = scipy.ndimage.morphology.grey_closing(resampled_array1, structure=kernel_small, mode='constant')
    resampled_array2 = scipy.ndimage.morphology.grey_closing(resampled_array2, structure=kernel_small, mode='constant')
    resampled_array3 = scipy.ndimage.morphology.grey_closing(resampled_array3, structure=kernel_small, mode='constant')

    if debug: show_2D_or_3D(resampled_array1[::10])
    plt.show()
    if debug: show_2D_or_3D(resampled_array2[::10])
    plt.show()
    if debug: show_2D_or_3D(resampled_array3[::10])
    plt.show()

    resampled_array = add_labels_max_thres(resampled_array1, resampled_array2, resampled_array3, threshold)

    resampled_img = sitk.GetImageFromArray(resampled_array)
    resampled_img = sitk.Cast(resampled_img, sitk.sitkUInt8)
    resampled_img.CopyInformation(resampled_label1)
    return resampled_img, file_path


# reutrns resampled img1 by referencing img2 with percentage, original from the TMI paper!
def max_thres_resample2_iso_label_img_original(sitk_img1, threshold):
    from src.visualization.Visualize import show_2D_or_3D
    debug = True

    label1_img1 = get_single_label_img(sitk_img1, 1)
    label2_img1 = get_single_label_img(sitk_img1, 2)
    label3_img1 = get_single_label_img(sitk_img1, 3)

    resampled_label1 = transform_to_isotrop_voxels(label1_img1, interpolate=sitk.sitkLinear, spacing_=(1.5, 1.5, 1.5))
    resampled_label2 = transform_to_isotrop_voxels(label2_img1, interpolate=sitk.sitkLinear, spacing_=(1.5, 1.5, 1.5))
    resampled_label3 = transform_to_isotrop_voxels(label3_img1, interpolate=sitk.sitkLinear, spacing_=(1.5, 1.5, 1.5))

    resampled_array1 = sitk.GetArrayFromImage(resampled_label1)
    resampled_array2 = sitk.GetArrayFromImage(resampled_label2)
    resampled_array3 = sitk.GetArrayFromImage(resampled_label3)

    if debug: show_2D_or_3D(resampled_array1[::10])
    plt.show()
    if debug: show_2D_or_3D(resampled_array2[::10])
    plt.show()
    if debug: show_2D_or_3D(resampled_array3[::10])
    plt.show()

    resampled_array = add_labels_max_thres(resampled_array1, resampled_array2, resampled_array3, threshold)
    resampled_img = sitk.GetImageFromArray(resampled_array)
    resampled_img = sitk.Cast(resampled_img, sitk.sitkUInt8)
    resampled_img.CopyInformation(resampled_label1)
    return resampled_img


# origin shift with transformindex
def resample_direcion_origin_spacing_shift(sitk_img, reference_sitk, shift, interpolate=sitk.sitkLinear,
                                           file_path='temp.nrrd'):
    """
    Resample a sitk img, copy direction, origin and spacing of the reference image
    Keep the size (resolution) of the target image
    :param sitk_img: sitk.Image
    :param reference_sitk: sitk.Image
    :param interpolate: fn
    :return: falsch muss umgeschrieben werden
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_sitk.GetSpacing())
    resampler.SetInterpolator(interpolate)
    resampler.SetOutputDirection(reference_sitk.GetDirection())
    # if coupled with resampler.SetSize only shift origin if negativ
    resampler.SetOutputOrigin(reference_sitk.TransformIndexToPhysicalPoint(tuple([abs(x) * -1 for x in shift])))
    # resampler.SetSize(sitk_img.GetSize())
    resampler.SetSize(tuple(map(sum, zip(reference_sitk.GetSize(), tuple([abs(x) * 2 for x in shift])))))
    # resampler.SetSize(tuple(map(sum,zip(sitk_img.GetSize(), tuple([abs(x) for x in shift])))))
    resampled = resampler.Execute(sitk_img)
    # copy metadata
    for key in reference_sitk.GetMetaDataKeys():
        resampled.SetMetaData(key, get_metadata_maybe(reference_sitk, key))
    return resampled, file_path


# reutrns resampled img1 by referencing img2 with percentage
def max_thres_resample2_label_img_shift(sitk_img1, sitk_img2, threshold, shift, file_path='temp.nrrd'):
    label1_img1 = get_single_label_img(sitk_img1, 1)
    label2_img1 = get_single_label_img(sitk_img1, 2)
    label3_img1 = get_single_label_img(sitk_img1, 3)

    resampled_label1, _ = resample_direcion_origin_spacing_shift(label1_img1, sitk_img2, shift,
                                                                 interpolate=sitk.sitkLinear)
    resampled_label2, _ = resample_direcion_origin_spacing_shift(label2_img1, sitk_img2, shift,
                                                                 interpolate=sitk.sitkLinear)
    resampled_label3, _ = resample_direcion_origin_spacing_shift(label3_img1, sitk_img2, shift,
                                                                 interpolate=sitk.sitkLinear)

    resampled_array1 = sitk.GetArrayFromImage(resampled_label1)
    resampled_array2 = sitk.GetArrayFromImage(resampled_label2)
    resampled_array3 = sitk.GetArrayFromImage(resampled_label3)

    resampled_array = add_labels_max_thres(resampled_array1, resampled_array2, resampled_array3, threshold)
    resampled_img = sitk.GetImageFromArray(resampled_array)
    resampled_img = sitk.Cast(resampled_img, sitk.sitkUInt8)
    resampled_img.CopyInformation(resampled_label1)
    return resampled_img, file_path


def transform_to_isotrop_voxels(sitk_img, interpolate=sitk.sitkBSpline, spacing_=(1., 1., 1.), file_path='temp.nrrd'):
    size = sitk_img.GetSize()
    spacing = sitk_img.GetSpacing()
    size_new = tuple([int((s * space_old) // space_new) for s, space_old, space_new in zip(size, spacing, spacing_)])

    # resample to isotrop voxels
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size_new)
    resampler.SetOutputSpacing(spacing_)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetInterpolator(interpolate)
    resampled = resampler.Execute(sitk_img)
    # copy metadata
    for key in sitk_img.GetMetaDataKeys():
        resampled.SetMetaData(key, get_metadata_maybe(sitk_img, key))
    return resampled, file_path
