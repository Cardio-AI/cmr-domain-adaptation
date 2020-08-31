import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def show_slice_transparent(img=None, mask=None, show=True, f_size=(5, 5), ax=None):
    """
    Plot image + masks in one figure
    """
    mask_values = [1, 2, 3]

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        return

    # replace mask with empty slice if none is given
    if mask is None:
        shape = img.shape
        mask = np.zeros((shape[0], shape[1], 3))

    # replace image with empty slice if none is given
    if img is None:
        shape = mask.shape
        img = np.zeros((shape[0], shape[1], 1))

    # check image shape
    if len(img.shape) == 2:
        # image already in 2d shape take it as it is
        x_ = (img).astype(np.float32)
    elif len(img.shape) == 3:
        # take only the first channel, grayscale - ignore the others
        x_ = (img[..., 0]).astype(np.float32)
    else:
        logging.error('invalid dimensions for image: {}'.format(img.shape))
        return

    # check masks shape, handle mask without channel per label
    if len(mask.shape) == 2:  # mask with int as label values
        mask_ = np.zeros((*(mask.shape), len(mask_values)), dtype=np.float32)
        for ix, mask_value in enumerate(mask_values):
            mask_[..., ix] = np.maximum(mask_[..., ix], mask == mask_value)
        y_ = mask_.astype(np.float32)

    elif len(mask.shape) == 3 and mask.shape[2] == 1:  # handle mask with additional channel
        mask = mask[..., 0]
        mask_ = np.zeros((*(mask.shape), len(mask_values)), dtype=np.float32)
        for ix, mask_value in enumerate(mask_values):
            mask_[..., ix] = np.maximum(mask_[..., ix], mask == mask_value)
        y_ = mask_.astype(np.float32)
    elif len(mask.shape) == 3 and mask.shape[2] == 3:  # handle mask with three channels
        y_ = (mask).astype(np.float32)
    elif len(mask.shape) == 3 and mask.shape[2] == 4:  # handle mask with 4 channels (backround = first channel)
        # ignore backround channel for plotting
        y_ = (mask[..., 1:] > 0.5).astype(np.float32)
    else:
        logging.error('invalid dimensions for masks: {}'.format(mask.shape))
        return
    if not ax: # no axis given
        fig = plt.figure(figsize=f_size)
        ax = fig.add_subplot(1, 1, 1, frameon=False)
    else: # axis given get the current fig
        fig = plt.gcf()
    fig.tight_layout(pad=0)
    ax.axis('off')
    ax.imshow(x_, 'bone', interpolation='none')
    ax.imshow(y_, interpolation='none', alpha=.2)
    

    if show:
        """
                logging.info('Image-shape: {}'.format(x_.shape))
        logging.info('Image data points: {}'.format((x_ > 0).sum()))
        logging.info('Image mean: {:.3f}'.format(x_.mean()))
        logging.info('Image max: {:.3f}'.format(x_.max()))
        logging.info('Image min: {:.3f}'.format(x_.min()))
        logging.info('Mask-shape: {}'.format(y_.shape))
        logging.info('RV mask data points: {}'.format((y_[..., 0] > 0.0).sum()))
        logging.info('Myo mask data points: {}'.format((y_[..., 1] > 0.0).sum()))
        logging.info('LV mask data points: {}'.format((y_[..., 2] > 0.0).sum()))
        logging.info('RV mask {}% of total pixels.'.format(y_[..., 0].mean()))
        logging.info('Myo mask {}% of total pixels.'.format(y_[..., 1].mean()))
        logging.info('LV mask {}% of total pixels.'.format(y_[..., 2].mean()))
        """

        return fig
    else:
        # fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data