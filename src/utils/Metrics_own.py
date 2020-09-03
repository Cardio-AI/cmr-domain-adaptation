from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from functools import partial
from tensorflow.keras.losses import mse
from src.data.Dataset import get_metadata_maybe, ensure_dir

def max_volume_loss(min_probabillity=0.8,):
    """
    Create a callable loss function which maximizes the probability values of y_pred
    There is additionally the possibility to weight high probabillities in
    :param min_probabillity:
    :return: loss function which maximize the number of voxels with a probabillity higher than the threshold
    """

    def max_loss(y_true, y_pred):
        """
        Maximize the foreground voxels in the middle slices with a probability heigher than a given threshold.
        :param y_true:
        :param y_pred:
        :param weights:
        :return:
        """

        y_pred = y_pred[...,1:] # ignore background, we want to maximize the number of captured ventricle voxel
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        sum_bigger_than = tf.reduce_max(y_pred, axis=-1)
        mask_bigger_than = tf.cast(sum_bigger_than > min_probabillity, tf.float32)
        sum_bigger_than = sum_bigger_than * mask_bigger_than

        return 1- tf.reduce_mean(sum_bigger_than)

    return max_loss


def loss_with_zero_mask(loss=mse, mask_smaller_than=0.01, weight_inplane=False,xy_shape=224):
    """
    Loss-factory returns a loss which calculates a given loss-function (e.g. MSE) only for the region where y_true is greater than a given threshold
    This is necessary for our AX2SAX comparison, as we have different length of CMR stacks (AX2SAX gt is cropped at z = SAX.z + 20mm)
    Example inplane weighting which is multiplied to each slice of the volume
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
    :param loss: any callable loss function. e.g. tf.keras.losses
    :param mask_smaller_than: float, threshold to calculate the loss only for voxels where gt is greater
    :param weight_inplane: bool, apply in-plane weighting
    :param xy_shape: int, number of in-plane pixels
    :return:
    """

    # in-plane weighting, which helps to focus on the voxels close to the center
    x_shape = xy_shape
    y_shape = xy_shape
    temp = np.zeros((x_shape, y_shape))
    weights_distribution = np.linspace(0, 1, x_shape // 2)
    for i, l in enumerate(weights_distribution):
        temp[i:-i, i:-i] = l
    weights = temp[None, None, :, :]
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    def my_loss(y_true, y_pred, weights_inplane=weights):
        """
        wrapper to either calculate a loss only on areas where the gt is greater than mask_smaller_than
        and additionally weight the loss along the z axis to increase the importance of the first slices or the last ones
        :param y_true:
        :param y_pred:
        :return:
        """
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, dtype=tf.float32)
        mask = tf.squeeze(tf.cast((y_true > mask_smaller_than),tf.float32),axis=-1)

        if weight_inplane:
            return (loss(y_true, y_pred) * mask) * weights_inplane + K.epsilon()
        else:
            return loss(y_true, y_pred) * mask

    return my_loss

def loss_with_margin(loss=mse, z_margin=-10,inplane_margin=20):
    """
    Wrapper to calcuate the loss only for a specific area
    e.g.:
    Finetune a AXtoSAX SpatialTransformer network on the first n slices,
    This loss uses only the first n slices, and within those slices only the center area
    This could be helpfull if the network should further focus on aligning the spatial translation along z
    :param y_true:
    :param y_pred:
    :param loss:
    :param z_margin:
    :param inplane_margin:
    :return:
    """
    def my_loss(y_true, y_pred):
        y_true_ = y_true[...,:z_margin, inplane_margin: -inplane_margin, inplane_margin:-inplane_margin,:]
        y_pred_ = y_pred[...,:z_margin, inplane_margin: -inplane_margin, inplane_margin:-inplane_margin,:]
        return loss(y_true_, y_pred_)
    return my_loss

#https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True) # -1
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1) # -1
        return loss
    
    return loss

# modified with dice coef applied
# a weighted cross entropy loss function combined with the dice coef for faster learning
def weighted_cce_dice_coef(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probs of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    def cat_cross_entropy_dice_coef(y_true, y_pred):
        return loss(y_true, y_pred)- dice_coef(y_true, y_pred)
    
    return cat_cross_entropy_dice_coef

# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044

def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss(true,pred):  #this is a loss
    """

    :param true:
    :param pred:
    :return:
    """

    return 1-iou_core(true, pred)


def iou_core(true,pred):
    """

    :param true:
    :param pred:
    :return:
    """
    #this can be used as a loss if you make it negative

    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)
    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def competitionMetric2(true, pred):
    """

    :param true:
    :param pred:
    :return:
    """
    #any shape can go - can't be a loss function

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

# Define IoU metric
# https://www.tensorflow.org/api_docs/python/tf/metrics/mean_iou
# Used in Kaggle (https://github.com/daicoolb/unet_kaggle2018/blob/master/unet_full.py)
# background ignored
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 4) # define the number of classes
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# returns the mean iou for class 3 / LV
# running vars --> http://ronny.rest/blog/post_2017_09_11_tf_metrics/
def mean_iou_lv(y_true, y_pred):
    prec = []
    y_pred = tf.expand_dims(y_pred[...,3], axis=-1)
    y_true = tf.expand_dims(y_true[...,3],axis=-1)
    
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true,y_pred_, 2) 
        # define the number of classes
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# returns the mean iou for class 2 / myo
def mean_iou_myo(y_true, y_pred):
    prec = []
    y_pred = tf.expand_dims(y_pred[...,2], axis=-1)
    y_true = tf.expand_dims(y_true[...,2],axis=-1)
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true,y_pred_, 2) 
        # define the number of classes
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# returns the mean iou for class 1 / RV
def mean_iou_rv(y_true, y_pred):
    prec = []
    y_pred = tf.expand_dims(y_pred[...,1], axis=-1)
    y_true = tf.expand_dims(y_true[...,1],axis=-1)
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true,y_pred_, 2) 
        # define the number of classes
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# returns the mean iou for class 0 / Background
def mean_iou_background(y_true, y_pred):
    prec = []
    y_pred = tf.expand_dims(y_pred[...,0], axis=-1)
    y_true = tf.expand_dims(y_true[...,0],axis=-1)
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true,y_pred_, 2) 
        # define the number of classes
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef_background(y_true, y_pred):
    y_pred = y_pred[...,0]
    y_true = y_true[...,0]
    return dice_coef(y_true, y_pred)

def dice_coef_rv(y_true, y_pred):
    y_pred = y_pred[...,-3]
    y_true = y_true[...,-3]
    return dice_coef(y_true, y_pred)

def dice_coef_myo(y_true, y_pred):
    y_pred = y_pred[...,-2]
    y_true = y_true[...,-2]
    return dice_coef(y_true, y_pred)

def dice_coef_lv(y_true, y_pred):
    y_pred = y_pred[...,-1]
    y_true = y_true[...,-1]
    return dice_coef(y_true, y_pred)


# ignore background score
def dice_coef_labels(y_true, y_pred):

    # ignore background, slice from the back to work with and without background channels
    y_pred = y_pred[...,-3:-1]
    y_true = y_true[...,-3:-1]
    
    return dice_coef(y_true, y_pred)

# ignore background score

def dice_coef(y_true, y_pred):
    smooth = 1.
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_squared(y_true, y_pred):
    smooth = 1.
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)


def dice_numpy(y_true, y_pred, empty_score=1.0):

    """

    :param y_true:
    :param y_pred:
    :param empty_score:
    :return:
    """

    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def cce_dice_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return  0.5 * tf.keras.losses.categorical_crossentropy(y_true, y_pred)- dice_coef(y_true, y_pred)


def bce_dice_iou_loss(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    return (0.1 * iou_loss(y_true, y_pred) + 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred))- dice_coef(y_true, y_pred)


def bce_jac_loss(y_true_, y_pred_):

    if y_true_[0].shape[-1] > 4:  # make sure to work with channel first batches
        y_pred_ = tf.transpose(y_pred_, [0, 4, 2, 3, 1])
        y_true_ = tf.transpose(y_true_, [0, 4, 2, 3, 1])
    
    y_pred = y_pred_[...,1:]
    y_true = y_true_[...,1:]
    return (0.1 * jaccard_distance_loss(y_true, y_pred) + 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred))

def jaccard_distance_label_loss(y_true, y_pred):
    
    y_pred = y_pred[...,-3:]
    y_true = y_true[...,-3:]
    
    return jaccard_distance_loss(y_true, y_pred)

def bce_labels_loss(y_true, y_pred):

    if y_pred.shape[-1] == 4:
        y_pred = y_pred[...,1:]
        y_true = y_true[...,1:]
    
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def dice_coef_labels_loss(y_true, y_pred):

    if y_pred.shape[-1] == 4:
    #if tf.shape(y_pred)[-1] == 4:
        y_pred = y_pred[..., 1:]
        y_true = y_true[..., 1:]
    
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_squared_labels_loss(y_true, y_pred):
    if y_pred.shape[-1] == 4:
        y_pred = y_pred[..., 1:]
        y_true = y_true[..., 1:]
    
    return 1 - dice_coef_squared(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):

    
    return 1 - dice_coef(y_true, y_pred)

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, [1,2,3])
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean(numerator / (denominator + epsilon)) # average over classes and batch


def bce_dice_jac_loss(y_true, y_pred):

    #if y_true[0].shape[-1] > 4:  # make sure to work with channel first batches
    #    y_pred = tf.transpose(y_pred, [0, 4, 2, 3, 1])
    #    y_true = tf.transpose(y_true, [0, 4, 2, 3, 1])
    
    y_pred = y_pred[...,-3:]
    y_true = y_true[...,-3:]
    #return jaccard_distance_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)- dice_coef(y_true, y_pred)
    return (0.1 * jaccard_distance_loss(y_true, y_pred) + 0.5 * keras.losses.binary_crossentropy(y_true, y_pred))- dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """
    weighted binary cross entropy - dice coef loss
    uses all labels if shale labels ==3
    otherwise slice the background to ignore over-represented background class
    :param y_true:
    :param y_pred:
    :return:
    """

    # use all channels for the binary crossentropy
    # use only the labels for the dice loss
    if y_pred.shape[-1] == 4:
        y_pred = y_pred[...,-3:]
        y_true = y_true[...,-3:]


    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

# https://github.com/keras-team/keras/issues/9395
def generalized_dice_coef(y_true, y_pred):
    n_el = 1
    for dim in y_pred.shape:
        if dim != None:
            n_el *= int(dim)
    n_cl = y_pred.shape[-1]
    w = K.zeros(shape=(n_cl,))
    w = (K.sum(y_true, axis=(0,1,2)))/(n_el)
    w = 1/(w**2+0.000001)
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2))
    numerator = K.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2))
    denominator = K.sum(denominator)
    return 2*numerator/denominator

def generalized_dice_loss(y_true, y_pred): # no gradients, not possible as loss
    y_pred = y_pred[...,1:]
    y_true = y_true[...,1:]
    return -1 * generalized_dice_coef(y_true, y_pred)


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def iou_coef_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return -1 * iou

def iou_coef_labels_loss(y_true, y_pred, smooth=1):
    
    if y_true[0].shape[-1] > 4:  # make sure to work with channel first batches
        y_pred = tf.transpose(y_pred, [0, 4, 2, 3, 1])
        y_true = tf.transpose(y_true, [0, 4, 2, 3, 1])
    
    y_pred = y_pred[...,-3:-1]
    y_true = y_true[...,-3:-1]
    return iou_coef_loss(y_true, y_pred)



# https://github.com/vuptran/cardiac-segmentation/blob/master/helpers.py
# jaccard coef (IOU)
def jaccard_coef(y_true, y_pred, smooth=100):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return jac

# jaccard coef (IOU)
def jaccard_coef_background(y_true, y_pred):
    
    y_pred = tf.expand_dims(y_pred[...,0], axis=-1)
    y_true = tf.expand_dims(y_true[...,0],axis=-1)
    return jaccard_coef(y_pred, y_true)

# jaccard coef (IOU)
def jaccard_coef_rv(y_true, y_pred):
    
    y_pred = tf.expand_dims(y_pred[...,1], axis=-1)
    y_true = tf.expand_dims(y_true[...,1],axis=-1)
    return jaccard_coef(y_pred, y_true)

# jaccard coef (IOU)
def jaccard_coef_myo(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred[...,2], axis=-1)
    y_true = tf.expand_dims(y_true[...,2],axis=-1)
    return jaccard_coef(y_pred, y_true)

# jaccard coef (IOU)
def jaccard_coef_lv(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred[...,3], axis=-1)
    y_true = tf.expand_dims(y_true[...,3],axis=-1)
    return jaccard_coef(y_pred, y_true)

# use as loss, is unstable, needs to be combined with other losses
def jaccard_distance(y_true, y_pred, smooth=100):
    
    jac = jaccard_coef(y_pred, y_true, smooth=smooth)
    return (1 - jac) * smooth


def jaccard_coef_labels(y_true, y_pred):
    
    # ignore background
    y_pred = y_pred[...,-3:]
    y_true = y_true[...-3:]
    return jaccard_coef(y_pred, y_true)


# https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
def jaccard_distance_loss(y_true, y_pred, smooth=100): # does not learn anything
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


