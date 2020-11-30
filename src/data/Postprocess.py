import logging

import numpy as np
from skimage import measure
import SimpleITK as sitk


def clean_3d_prediction_3d_cc(pred, background=0):
    """
    Find the biggest connected component per label
    This is a debugging method, which will plot each step
    returns: a tensor with the same shape as pred, but with only one cc per label
    """

    # avoid labeling images with float values
    assert len(np.unique(pred)) < 10, 'to many labels: {}'.format(len(np.unique(pred)))

    cleaned = np.zeros_like(pred)

    def clean_3d_label(val):

        """
        has access to pred, no passing required
        """

        # create a placeholder
        biggest = np.zeros_like(pred)
        biggest_size = 0

        # find all cc for this label
        # tensorflow operation is only in 2D
        # all_labels = tfa.image.connected_components(np.uint8(pred==val)).numpy()
        all_labels = measure.label(np.uint8(pred == val), background=background)

        for c in np.unique(all_labels)[1:]:
            mask = all_labels == c
            mask_size = mask.sum()
            if mask_size > biggest_size:
                biggest = mask
                biggest_size = mask_size
        return biggest

    for val in np.unique(pred)[1:]:
        biggest = clean_3d_label(val)
        cleaned[biggest] = val
    return cleaned
import cv2

def clean_3d_prediction_2d_cc(pred):
    cleaned = []
    # for each slice
    for s in pred:
        new_img = np.zeros_like(s)  # step 1
        # for each label
        for val in np.unique(s)[1:]:  # step 2
            mask = np.uint8(s == val)  # step 3
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
            new_img[labels == largest_label] = val  # step 6
        cleaned.append(new_img)
    return np.stack(cleaned, axis=0)


def undo_generator_steps(ndarray, cfg, interpol=sitk.sitkLinear, orig_sitk=None):
    """
    Undo the generator steps for a 3D volume
    # load 3d volume, to copy the original metadata
    Parameters
    ----------
    ndarray :
    p :
    cfg :
    interpol :
    orig_sitk :

    Returns
    -------

    """
    from src.data.Preprocess import resample_3D, center_crop_or_pad_2d_or_3d
    try:
        # file_4d = glob.glob('{}{}'.format(orig_file_path, p))[0]
        # orig_sitk = sitk.ReadImage(file_4d)
        orig_size_ = orig_sitk.GetSize()
        orig_spacing_ = orig_sitk.GetSpacing()
        orig_size = list(reversed(orig_size_))
        orig_spacing = list(reversed(orig_spacing_))

        logging.debug('original shape: {}'.format(orig_size_))
        logging.debug('original spacing: {}'.format(orig_spacing_))
        logging.debug('reverse original shape: {}'.format(orig_size))
        logging.debug('reverse original spacing: {}'.format(orig_spacing))
    except Exception as e:
        logging.error('no metadata and spacing copied: {}'.format(str(e)))
        orig_sitk = None
    # numpy has the following order: h,w,c (or z,h,w,c for 3D)

    h_w_size = orig_size[:]
    h_w_spacing = orig_spacing[:]
    w_h_size_sitk = list(reversed(h_w_size))
    w_h_spacing_sitk = list(reversed(h_w_spacing))

    # calculate the size of the image before crop or pad
    new_size = (np.array(h_w_size) * np.array(h_w_spacing)) / np.array(cfg['SPACING'])
    new_size = [int(np.round(i)) for i in new_size]

    # pad, crop to original physical size in current spacing
    logging.debug('orig height/width size: {}'.format(h_w_size))
    logging.debug('orig height/width spacing: {}'.format(h_w_spacing))
    logging.debug('pred shape: {}'.format(ndarray.shape))
    logging.debug('intermediate size after : {}'.format(new_size))

    ndarray, _, _ = center_crop_or_pad_2d_or_3d(ndarray, ndarray, new_size)
    logging.debug(ndarray.shape)

    # resample, set current spacing
    img_ = sitk.GetImageFromArray(ndarray)
    img_.SetSpacing(cfg['SPACING'])

    img_ = resample_3D(img_, size=w_h_size_sitk, spacing=w_h_spacing_sitk, interpolate=interpol)

    logging.info('Size after undo: {}'.format(img_.GetSize()))
    logging.info('Spacing after undo: {}'.format(img_.GetSpacing()))

    return img_