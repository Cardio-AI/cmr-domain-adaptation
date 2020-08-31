import numpy as np
from skimage import measure


def clean_3d_prediction_3d_cc(pred):
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
        all_labels = measure.label(np.uint8(pred == val), background=0)

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