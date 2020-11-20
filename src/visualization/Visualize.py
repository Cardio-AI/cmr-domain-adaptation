import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix
from src.utils.Utils_io import save_plot, ensure_dir
import SimpleITK as sitk
from matplotlib.ticker import PercentFormatter
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
import numpy as np

def my_autopct(pct):
    """
    Helper to filter % values of a pie chart, which are smaller than 1%
    :param pct:
    :return:
    """
    return ('%1.0f%%'% pct) if pct > 1 else ''

def get_metadata_maybe(sitk_img, key, default='not_found'):
    # helper for unicode decode errors
    try:
        value = sitk_img.GetMetaData(key)
    except Exception as e:
        logging.debug('key not found: {}, {}'.format(key, e))
        value = default
    # need to encode/decode all values because of unicode errors in the dataset
    if not isinstance(value, int):
        value = value.encode('utf8', 'backslashreplace').decode('utf-8').replace('\\udcfc', 'ue')
    return value


def show_2D_or_3D(img=None, mask=None, save=False, file_name='reports/figure/temp.png',dpi=200,f_size=(5,5), interpol='bilinear'):
    """
    Debug wrapper for 2D or 3D image/mask vizualisation
    wrapper checks the ndim and calls shoow_transparent or plot 3d
    :param img:
    :param mask:
    :param show:
    :param f_size:
    :return:
    """

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        return
    if img is not None:
        dim = img.ndim
    else:
        dim = mask.ndim


    if dim == 2:
        return show_slice_transparent(img, mask)
    elif dim == 3 and img.shape[-1] == 1:  # data from the batchgenerator
        return show_slice_transparent(img, mask)
    elif dim == 3:
        return plot_3d_vol(img, mask, save=save, path=file_name,dpi=dpi,fig_size=f_size, interpol=interpol)
    elif dim == 4 and img.shape[-1] == 1:  # data from the batchgenerator
        return plot_3d_vol(img, mask, save=save, path=file_name,dpi=dpi,fig_size=f_size, interpol=interpol)
    elif dim == 4 and img.shape[-1] in [3,4]: # only mask
        return plot_3d_vol(img, save=save, path=file_name,dpi=dpi,fig_size=f_size, interpol=interpol)
    elif dim == 4:
        return plot_4d_vol(img, mask)
    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise NotImplementedError('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))


def create_eval_plot(df_dice, df_haus=None, df_hd=None, df_vol=None, eval=None):

    # create a violinplot with an integrated bland altmann plot
    # nobs = median
    import seaborn as sns
    outliers = False
    my_pal_1 = {"Dice LV": "dodgerblue", "Dice MYO": "g", "Dice RV":"darkorange"}
    my_pal_2 = {"Err LV(ml)": "dodgerblue", "Err MYO(ml)": "g", "Err RV(ml)":"darkorange"}
    my_pal_3 = {"Volume LV": "dodgerblue", "Volume MYO": "g", "Volume RV":"darkorange"}
    hd_pal = {"Hausdorff LV": "dodgerblue", "Hausdorff MYO": "g", "Hausdorff RV": "darkorange"}


    plt.rcParams.update({'font.size': 20})
    if df_haus is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25,8), sharey=False)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8), sharey=False)

    ax1 = sns.violinplot(x= 'variable',y = 'value', data=df_dice,order=["Dice LV", "Dice MYO", "Dice RV"],palette=my_pal_1 , showfliers = outliers, ax=ax1)
    medians = df_dice.groupby(['variable'])['value'].mean().round(2)
    sd = df_dice.groupby(['variable'])['value'].std().round(2)
    nobs = ['{}+/-{}'.format(m,s) for m,s in zip(medians, sd)]

    for tick,label in zip(range(len(ax1.get_xticklabels())),ax1.get_xticklabels()):
        _ = ax1.text(tick, medians[tick], nobs[tick],horizontalalignment='center', size='x-small', color='black', weight='semibold')
    plt.setp(ax1, ylim=(0,1))
    plt.setp(ax1, ylabel=('DICE'))
    plt.setp(ax1, xlabel='')
    ax1.set_xticklabels(['LV','MYO', 'RV'])

    # create bland altmannplot from vol diff
    ax2 = bland_altman_metric_plot(eval, ax2)

    # create violin plot for the volume
    ax3 = sns.violinplot(x= 'variable',y = 'value',order=["Volume LV", "Volume MYO", "Volume RV"], palette=my_pal_3, showfliers = outliers, data=df_vol, ax=ax3)

    medians = df_vol.groupby(['variable'])['value'].mean().round(2)
    sd = df_vol.groupby(['variable'])['value'].std().round(2)
    nobs = ['{}+/-{}'.format(m,s) for m,s in zip(medians, sd)]

    for tick,label in zip(range(len(ax3.get_xticklabels())),ax3.get_xticklabels()):
        _ = ax3.text(tick, medians[tick], nobs[tick],horizontalalignment='center', size='x-small', color='black', weight='semibold')
    #plt.setp(ax3, ylim=(0,500))
    plt.setp(ax3, ylabel=('Vol size in ml'))
    plt.setp(ax3, xlabel='')
    ax3.set_xticklabels(['LV','MYO', 'RV'])

    ax4 = sns.violinplot(x='variable', y='value', order=["Hausdorff LV", "Hausdorff MYO", "Hausdorff RV"], palette=hd_pal,
                         showfliers=outliers, data=df_haus, ax=ax4)

    medians = df_haus.groupby(['variable'])['value'].mean().round(2)
    sd = df_haus.groupby(['variable'])['value'].std().round(2)
    nobs = ['{}+/-{}'.format(m, s) for m, s in zip(medians, sd)]

    for tick, label in zip(range(len(ax4.get_xticklabels())), ax4.get_xticklabels()):
        _ = ax4.text(tick, medians[tick], nobs[tick], horizontalalignment='center', size='x-small', color='black',
                     weight='semibold')
    plt.setp(ax4, ylim=(0, 50))
    plt.setp(ax4, ylabel=('Hausdorff distance'))
    plt.setp(ax4, xlabel='')
    ax4.set_xticklabels(['LV','MYO', 'RV'])
    plt.tight_layout()
    return fig

def show_slice(img=[], mask=[], show=True, f_size=(15, 5)):
    """
    Plot image + masks in one figure
    """
    mask_values = [1, 2, 3]

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if len(img) == 0 and len(mask) == 0:
        print('no images given')
        return

    # replace mask with empty slice if none is given
    if len(mask) == 0:
        shape = img.shape
        mask = np.zeros((shape[0], shape[1], 3))

    # replace image with empty slice if none is given
    if len(img) == 0:
        shape = mask.shape
        img = np.zeros((shape[0], shape[1], 1))

    # check image shape
    if len(img.shape) == 2:
        # image already in 2d shape take it as it is
        x_ = img.astype(np.float32)
    elif len(img.shape) == 3:
        # take only the first channel, grayscale - ignore the others
        x_ = (img[..., 0]).astype(np.float32)
    else:
        logging.info('invalid dimensions for image: {}'.format(img.shape))
        return

    # check masks shape, handle mask without channel per label
    if len(mask.shape) == 2:  # mask with int as label values
        y_ = transform_to_binary_mask(mask, mask_values=mask_values)

    elif len(mask.shape) == 3 and mask.shape[2] == 3:  # handle mask with three channels
        y_ = mask

    elif len(mask.shape) == 3 and mask.shape[2] in [4]:  # handle mask with 4 channels (backround = first channel)
        y_ = mask[..., 1:]  # ignore backround channel for plotting

    else:
        logging.info('invalid dimensions for masks: {}'.format(mask.shape))
        return

    y_ = (y_).astype(np.float32)  # set a threshold for slices during training

    # scale image between 0 and 1
    x_ = (x_ - x_.min()) / (x_.max() - x_.min() + sys.float_info.epsilon)

    # draw mask and image as rgb image, 
    # use the green channel for mask and image
    temp = np.zeros((x_.shape[0], x_.shape[1], 3), dtype=np.float32)
    temp[..., 1] = np.maximum(x_, y_[..., 1] > 0.5)  # green
    temp[..., 0] = np.maximum(x_, y_[..., 0] > 0.5)  # red
    temp[..., 2] = np.maximum(x_, y_[..., 2] > 0.5)  # blue

    if show:
        # define figure size
        rows = 1
        columns = 3
        fig = plt.figure(figsize=f_size)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(temp)

        # RV = 1 = Y[:,:,0] # Myo = 2 = Y[:,:,1] # LV = 3 = Y[:,:,2]
        # draw all mask channels as rgb image
        fig.add_subplot(rows, columns, 2)
        temp2 = np.zeros((x_.shape[0], x_.shape[1], 3))
        temp2[..., 1] = y_[..., 1]
        temp2[..., 0] = y_[..., 0]
        temp2[..., 2] = y_[..., 2]
        plt.imshow(temp2)

        # draw the plain training image
        fig.add_subplot(rows, columns, 3)
        plt.imshow(x_)

        fig.tight_layout(pad=0)
        #plt.show()
        logging.info('Image-shape: {}'.format(x_.shape))
        logging.info('Image data points: {}'.format((x_ > 0).sum()))
        logging.info('Image mean: {:.3f}'.format(x_.mean()))
        logging.info('Image max: {:.3f}'.format(x_.max()))
        logging.info('Image min: {:.3f}'.format(x_.min()))
        logging.info('Mask-shape: {}'.format(y_.shape))
        logging.info('RV mask data points: {}'.format((y_[:, :, 0] > 0.0).sum()))
        logging.info('Myo mask data points: {}'.format((y_[:, :, 1] > 0.0).sum()))
        logging.info('LV mask data points: {}'.format((y_[:, :, 2] > 0.0).sum()))
        logging.info('RV mask mean: {}'.format(y_[:, :, 0].mean()))
        logging.info('Myo mask mean: {}'.format(y_[:, :, 1].mean()))
        logging.info('LV mask mean: {}'.format(y_[:, :, 2].mean()))

    else:
        return temp

    """
    experimental: convert figure to numpy array, works but bad quality
    
            fig.canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    """


def show_slice_transparent(img=None, mask=None, show=True, f_size=(5, 5), ax=None):
    """
    Plot image + masks in one figure
    """
    mask_values = [1, 2, 3]

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

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
        y_ = transform_to_binary_mask(mask, mask_values=[1,2,3]).astype(np.float32)
        #print(y_.shape)
    elif len(mask.shape) == 3 and mask.shape[2] == 1:  # handle mask with empty additional channel
        mask = mask[..., 0]
        y_ = transform_to_binary_mask(mask, mask_values=[1,2,3]).astype(np.float32)
        
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
    
    # normalise image
    x_ = (x_ - x_.min()) / (x_.max() - x_.min() + sys.float_info.epsilon)
    ax.imshow(x_, 'gray',vmin=0,vmax=0.4)
    ax.imshow(y_, interpolation='none', alpha=.3)

    if show:
        return ax
    else:
        return fig


def bland_altman_metric_plot(metric, ax = None):
    """
    Plots a Bland Altmann plot for a evaluation dataframe from the acdc scripts
    :param metric: pd.Dataframe
    :return: plt.ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 15))

    my_colors = {"LV": "dodgerblue", "MYO": "g", "RV": "darkorange"}

    def bland_altman_plot(data1, data2, shifting,*args, **kwargs):
        """
        Create a single bland altmann plot into the ax object of the surounding wrapper function,
        this functions will be called for each label
        :param data1: list - prediction
        :param data2: list - ground truth
        :param shifting: float - relative y-axis shift from 0
        :param args:
        :param kwargs:
        :return: None
        """
        from scipy.stats import wilcoxon, ttest_ind
        #stat, p = wilcoxon(data1, data2)
        #print('wilcoxon rank test: {}, {}'.format(stat, p))
        stat, p = ttest_ind(data1, data2)
        print('T-test - stats: {}, p: {}'.format(stat, p))

        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        # plot points and lines
        line_size = 4
        ax.scatter(mean, diff, alpha=0.4, s=200,*args, **kwargs)
        ax.axhline(md, **kwargs, linestyle='-', alpha=0.5, lw=line_size)
        ax.axhline(md + 1.96 * sd, **kwargs, linestyle='--', alpha=0.5, lw=line_size)
        ax.axhline(md - 1.96 * sd, **kwargs, linestyle='--', alpha=0.5, lw=line_size)

        # calculate Properties for text
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        limitOfAgreement = 1.96
        limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement * sd)
        offset = (limitOfAgreementRange / 100.0) * 1.5
        text_size = plt.rcParams['font.size']
        ax.annotate(f'{md:.2f}' + ' ± ' + f'{sd:.2f}', xy=(0.05, shifting), xycoords='axes fraction', fontsize=text_size, fontname='Cambria',weight='semibold', **kwargs)

    # plot 3 different metrics into the same plot
    pred = metric['Volume LV']
    gt = pred - metric['Err LV(ml)']
    bland_altman_plot(pred, gt, shifting=0.95, color=my_colors['LV'])

    pred = metric['Volume MYO']
    gt = pred - metric['Err MYO(ml)']
    bland_altman_plot(pred, gt, shifting=0.90, color=my_colors['MYO'])

    pred = metric['Volume RV']
    gt = pred - metric['Err RV(ml)']
    bland_altman_plot(pred, gt, shifting=0.85, color=my_colors['RV'])

    # set labels
    label_size = plt.rcParams['font.size']
    ax.set_ylabel('Vol diff \n(pred - gt) in ml', fontsize=label_size, fontname='Cambria')
    ax.set_xlabel('Mean vol in ml', fontsize=label_size, fontname='Cambria')

    # set legend
    legend_size = plt.rcParams['font.size']
    LV_patch = mpatches.Patch(color=my_colors['LV'], label='LV')
    MYO_patch = mpatches.Patch(color=my_colors['MYO'], label='MYO')
    RV_patch = mpatches.Patch(color=my_colors['RV'], label='RV')
    ax.legend(handles=[LV_patch, MYO_patch, RV_patch], prop={'size': legend_size})

    # set axis
    ax.tick_params(axis='both', labelsize=plt.rcParams['font.size'])
    # to test if fixed axes is ok just comment the next line and add back, if it didn't get bigger
    ax.set_ylim(-250, 250)
    ax.set_xlim(0, 550)

    return ax

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_4d_vol(img_4d, timesteps=[0], save=False, path='temp/', mask_4d=None, f_name='4d_volume.png'):
    # creates a grid with # timesteps * z-slices
    # saves all slices as fig
    # expects nda with t, z, x, y

    if isinstance(img_4d, sitk.Image):
        img_4d = sitk.GetArrayFromImage(img_4d)

    if len(timesteps) <= 1:  # add first volume if no timesteps found
        logging.info('No timesteps given for: {}, use img.shape[0]'.format(path))
        timesteps = list(range(0, img_4d.shape[0]))
    assert (len(timesteps) == img_4d.shape[0]), 'timeteps does not match'

    if img_4d.shape[-1] == 4:
        img_4d = img_4d[..., 1:]  # ignore background if 4 channels are given
    
    elif img_4d.shape[-1] == 1:
        img_4d = img_4d[..., 0]  # ignore single channels at the end, matpotlib cant plot this shape

    
    if mask_4d is not None: # if images and masks are provided
        if mask_4d.shape[-1] == 4:
            mask_4d = mask_4d[..., 1:]  # ignore background for masks if 4 channels are given
    
    
    # define the number of subplots
    # timesteps * z-slices
    z_size = min(int(2*img_4d.shape[1]), 30)
    t_size = min(int(2*len(timesteps)), 20)
    logging.info('figure: {} x {}'.format(z_size, t_size))

    # long axis volumes have only one z slice squeeze=False is neccessary to avoid sqeezing the axes
    fig, ax = plt.subplots(len(timesteps), img_4d.shape[1], figsize=[z_size, t_size],squeeze=False)
    #print(timesteps)
    for t_, img_3d in enumerate(img_4d):  # traverse trough time

        for z, slice in enumerate(img_3d):  # traverse through the z-axis
            # show slice and delete ticks
            if mask_4d is not None:
                
                ax[t_][z] = show_slice_transparent(slice, mask_4d[t_, z, ...], show=True, ax=ax[t_][z])
            else:
                ax[t_][z] = show_slice_transparent(slice, show=True, ax=ax[t_][z])
            ax[t_][z].set_xticks([])
            ax[t_][z].set_yticks([])
            # ax[t_][z].set_aspect('equal')
            if t_ == 0:  # set title before first row
                ax[t_][z].set_title('z-axis: {}'.format(z), color='r')
            if z == 0:  # set ylabel before first column
                ax[t_][z].set_ylabel('t-axis: {}'.format(timesteps[t_]), color='r')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.tight_layout()

    if save:
        ensure_dir(path)
        save_plot(fig, path, f_name, override=True, tight=False)
    else:
        return fig
        #fig.show()


def plot_3d_vol(img_3d, mask_3d=None, timestep=0, save=False, path='reports/figures/temp.png',
                fig_size=[25, 8], dpi=200, interpol='nearest'):
    """
    plots a 3D nda, if a mask is given combine mask and image slices
    :param show:
    :param img_3d:
    :param mask_3d:
    :param timestep:
    :param save:
    :param path:
    :param fig_size:
    :return: plot figure
    """

    if isinstance(img_3d, sitk.Image):
        img_3d = sitk.GetArrayFromImage(img_3d)

    if isinstance(mask_3d, sitk.Image):
        mask_3d = sitk.GetArrayFromImage(mask_3d)

    # use float as dtype for all plots
    if img_3d is not None:
        img_3d = img_3d.astype(np.float32)
    if mask_3d is not None:
        mask_3d = mask_3d.astype(np.float32)

    if img_3d.max() == 0:
        logging.debug('timestep: {} - no values'.format(timestep))
    else:
        logging.debug('timestep: {} - plotting'.format(timestep))

    if img_3d.shape[-1] in [3,4]: # this image is a mask
        img_3d = img_3d[..., -3:]  # ignore background
        mask_3d = img_3d # handle this image as mask
        img_3d = np.zeros((mask_3d.shape[:-1]))

    elif img_3d.shape[-1] == 1:
        img_3d = img_3d[..., 0]  # matpotlib cant plot this shape

    if mask_3d is not None:
        if mask_3d.shape[-1] == 4:
            mask_3d = mask_3d[..., 1:]  # ignore background if 4 channels are given
        elif mask_3d.shape[-1] > 4:
            mask_3d = transform_to_binary_mask(mask_3d)

    slice_n = 1
    # slice very huge 3D volumes, otherwise they are too small on the plot
    if (img_3d.shape[0] > 20) and (img_3d.ndim == 3):
        slice_n = img_3d.shape[0] // 20

    img_3d = img_3d[::slice_n]
    mask_3d = mask_3d[::slice_n]if mask_3d is not None else mask_3d

    # number of subplots = no of slices in z-direction
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    for idx, slice in enumerate(img_3d):  # iterate over all slices
        ax = fig.add_subplot(1, img_3d.shape[0], idx + 1)

        if mask_3d is not None:
            ax = show_slice_transparent(img=slice, mask=mask_3d[idx], show=True, ax=ax)
        else:
            mixed = show_slice(img=slice, mask=[], show=False)
            ax.imshow(mixed)

        ax.set_xticks([])
        ax.set_yticks([])
        #real_index = idx + (idx * slice_n)
        #ax.set_title('z-axis: {}'.format(idx), color='r', fontsize=plt.rcParams['font.size'])


    fig.subplots_adjust(wspace=0, hspace=0)
    if save:
        save_plot(fig, path, str(timestep), override=False)

    else:
        return fig
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data


def transform_to_binary_mask(mask_nda, mask_values=[0, 1, 2, 3]):
    # transform the labels to binary channel masks

    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype=np.bool)
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask

def plot_value_histogram(nda, f_name='histogram.jpg', image=True, reports_path='reports/figures/4D_description'):
    '''
    plot 4 histograms for a numpy array of any shape
    1st plot with all values (100 buckets)
    2nd plot with .999 quantile (20 buckets)
    3rd plot with .75 quantile (20 buckets)
    4th plot with .5 quantile (bucketsize = Counter.most_common())
    y axis is percentual scaled
    x axis linear spaced - logspaced buckets are possible but not so visual clear
    '''
    ensure_dir(reports_path)
    nda_img_flat = nda.flatten()
    plt.close('all')

    if not image:
        fig = plt.figure(figsize=[6, 6])
        ax1 = fig.add_subplot(111)
        nda_img_flat_filter = nda_img_flat[nda_img_flat > 0]
        c = Counter(nda_img_flat_filter)
        ax1.hist(nda_img_flat_filter, weights=np.ones(len(nda_img_flat_filter)) / len(nda_img_flat_filter), bins=3)
        ax1.set_title("Mask with  = {0:.2f} values".format(len(c)))
        ax1.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        fig = plt.figure(figsize=[20, 6])
        ax1 = fig.add_subplot(141)
        ax1.hist(nda_img_flat, weights=np.ones(len(nda_img_flat)) / len(nda_img_flat), bins=100)
        ax1.set_title("1. quantile = {0:.2f}".format(nda_img_flat.max()))
        ax1.yaxis.set_major_formatter(PercentFormatter(1))

        ax2 = fig.add_subplot(142)
        ninenine_q = np.quantile(nda_img_flat, .999)
        nda_img_flat_nine = nda_img_flat[nda_img_flat <= ninenine_q]
        ax2.hist(nda_img_flat_nine, weights=np.ones(len(nda_img_flat_nine)) / len(nda_img_flat_nine), bins=20)
        ax2.set_title("0.999 quantile = {0:.2f}".format(ninenine_q))
        ax2.yaxis.set_major_formatter(PercentFormatter(1))

        ax3 = fig.add_subplot(143)
        seven_q = np.quantile(nda_img_flat, .75)
        nda_img_flat_seven = nda_img_flat[nda_img_flat <= seven_q]
        ax3.hist(nda_img_flat_seven, weights=np.ones(len(nda_img_flat_seven)) / len(nda_img_flat_seven), bins=20)
        ax3.set_title("0.75 quantile = {0:.2f}".format(seven_q))
        ax3.yaxis.set_major_formatter(PercentFormatter(1))

        ax4 = fig.add_subplot(144)
        mean_q = np.quantile(nda_img_flat, .5)
        nda_img_flat_mean = nda_img_flat[nda_img_flat <= mean_q]
        c = Counter(nda_img_flat_mean)
        ax4.hist(nda_img_flat_mean, weights=np.ones(len(nda_img_flat_mean)) / len(nda_img_flat_mean),
                 bins=len(c.most_common()))
        ax4.set_title("0.5 quantile = {0:.2f}".format(mean_q))
        ax4.set_xticks([key for key, _ in c.most_common()])
        ax4.yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle(f_name, y=1.08)
    fig.tight_layout()
    plt.savefig(os.path.join(reports_path, f_name))
    plt.show()


def create_quiver_plot(flowfield_2d=None, ax=None, N=5, scale=0.3, linewidth=.5):
    """
    Function to create an easy flowfield from the voxelmorph output
    Needs a 2D flowfield, function can handle 2D or 3D vectors as channels
    :param N: take only ever n vector
    :param flowfield_2d: numpy array with shape x, y, vectors
    :param ax: matplotlib ax object which should be used for plotting,
    create a new ax object if none is given
    :return: ax to plot or save
    """
    from matplotlib import cm
    from src.data.Preprocess import normalise_image
    import matplotlib

    if not ax:
        fig, ax = plt.subplots(figsize=(15, 15))

    # extract flowfield for x and y
    if flowfield_2d.shape[-1] == 3:  # originally a 3d flowfield
        Z_ = flowfield_2d[..., 0]
        X_ = flowfield_2d[..., 1]
        Y_ = flowfield_2d[..., 2]
    elif flowfield_2d.shape[-1] == 2:  # 2d flowfield
        X_ = flowfield_2d[..., 0]
        Y_ = flowfield_2d[..., 1]

    border = 0
    start_x = border
    start_y = border
    nz = Z_.shape[0] - border
    nx = X_.shape[0] - border  # define ticks in x
    ny = Y_.shape[1] - border # define ticks in y

    # slice flowfield, take every N value
    Fz = Z_[::N, ::N]
    Fx = X_[::N, ::N]
    Fy = Y_[::N, ::N]
    nrows, ncols = Fx.shape

    # create a grid with the size nx/ny and ncols/nrows
    z_ = np.linspace(start_x, nz, ncols)
    x_ = np.linspace(start_x, nx, ncols)
    y_ = np.linspace(start_y, ny, nrows)
    xi, yi = np.meshgrid(x_, y_, indexing='xy')
    zi, _ = np.meshgrid(z_, nx, indexing='xy')

    # working, use z as color
    # this way is not as clear as the test 3
    #norm = normalise_image(Fz)
    #colors = cm.copper(norm)
    #colors = colors.reshape(-1, 4)

    # test 3
    occurrence = Fz.flatten() / np.sum(Fz)
    norm = matplotlib.colors.Normalize()
    norm.autoscale(occurrence)
    cm = matplotlib.cm.copper
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    colors = cm(norm(Fz)).reshape(-1, 4)

    # plot
    ax.set_title('Flowfield')
    #ax.quiver(xi, -yi, Fx, Fy, units='xy', scale=.5, alpha=.5)
    ax.quiver(xi, -yi, Fx, Fy, color=colors, units='xy', angles='xy', scale=scale, linewidth=linewidth, minshaft=2,headwidth=6, headlength=7)
    #plt.colorbar(sm)
    return ax
