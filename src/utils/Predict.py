import cv2
import tensorflow as tf
from src.data.Dataset import copy_meta_and_save
from src.data.Preprocess import normalise_image, ensure_dir, from_channel_to_flat, transform_to_binary_mask
from src.data.Postprocess import undo_generator_steps, clean_3d_prediction_3d_cc
from src.data.Dataset import get_reference_nrrd
from src.models.SpatialTransformer import create_affine_transformer_fixed
from src.visualization.Visualize import show_2D_or_3D
import src.utils.Loss_and_metrics as metr
tf.get_logger().setLevel('ERROR')
import logging
import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def predict_on_one_3d_file(full_file_name,
                           filename,
                           ax_small,
                           ax2sax_small_gt_,
                           ax_full_,
                           ax_msk_full_gt,
                           debug,
                           slice_n,
                           export_path,
                           save,
                           save_plots,
                           postprocess,
                           use_mod_translation,
                           cfg,
                           msk_cfg,
                           model,
                           unet_model,
                           path_to_original_sitk = '/mnt/ssd/data/gcn/ax_sax_from_flo/ax3d/'):
    """
    Use the spatial transformer to transform an AX CMR into the SAX domain,
    apply a task network (in our case a U-Net),
    transform the masks back into the AX domain,
    undo all generator steps and save the AX-CMR, AX-GT mask and AX-prediction mask as nrrd files (image, gt and pred).
    This method should be able to undo all generator steps,
    meaning that we can either use the gt files which went through the generator
    or the original gt files before any preprocessing was applied for the evaluation of this model
    Parameters
    ----------
    full_file_name : str - current file path
    filename : - short version of the current file
    ax_small : - np.ndarray - downsampled AX CMR
    ax2sax_small_gt_ : np.ndarray - downsampled AX mask
    ax_full_ : np.ndarray - AX CMR in full resolution
    ax_msk_full_gt : np.ndarray - AX CMR mask in full resolution
    debug : bool - plot intermediate steps
    slice_n : int - slice every nth slice of the volumes before plotting
    export_path : string - path to save the predicted files
    save : bool - save the prediction or not
    save_plots : bool - save the main intermediate plots to the figure_export path
    postprocess : bool . apply postprocessing or not
    use_mod_translation : bool - use the modified (True) or MSE-base set of translation parameters
    cfg : dict - experiment config
    msk_cfg : dict - modified experiment config to load the files in full resolution (DataGenerator)
    model : tf.keras.model - pre-trained AX-SAX model
    unet_model : tf.keras.model - 3D wrapper for any task specific model, such as a segmentation model

    Returns ax_cmr, ax_pred_msk, ax_gt_msk
    -------

    """

    transformer_spacing = cfg['SPACING'][0]
    full_spacing = msk_cfg['SPACING'][0]
    mask_threshold = 0.5
    figure_export = 'reports/figures/temp'
    ensure_dir(figure_export)


    # define a different logging level and plot the generator steps
    if debug: logging.info('Prediction on AX volume:')
    if debug: show_2D_or_3D(ax_full_[::slice_n * 3], save=save_plots, file_name=os.path.join(figure_export, 'ax'))
    plt.show()

    # Predict rotation of AX_small and get transformation matrix
    pred, inv, pred_mod, prob, inv_msk, m, m_mod = model.predict(
        x=[np.expand_dims(ax_small, axis=0), np.expand_dims(ax_small, axis=0)])
    if debug: logging.info('AX --> SAX rotated by the model')
    if debug: show_2D_or_3D(pred[0][::slice_n])
    plt.show()
    if debug: logging.info('AX --> SAX with modified m rotated by the model')
    if debug: show_2D_or_3D(pred_mod[0][::slice_n], prob[0][::slice_n])
    plt.show()
    inv_msk = inv_msk >= 0.5

    # scale the translation parameter of the affine matrix from spacing 5 to 1.5
    # These lines change m
    m_temp = m.copy()
    m_mod_temp = m_mod.copy()
    if use_mod_translation:
        m_scaled = np.reshape(m_mod, (3, 4))
    else:
        m_scaled = np.reshape(m, (3, 4))
    m_t = m_scaled[:, 3]  # slice translation
    m_t = m_t * (transformer_spacing / full_spacing)  # scale translation
    m_scaled[:, 3] = m_t  # slice scaled translation back into m
    m_scaled_flatten = m_scaled.flatten()

    # show the target AXtoSAX volume
    if debug: logging.info('Target (AX2SAX):')
    if debug: show_2D_or_3D(ax2sax_small_gt_[::slice_n])
    plt.show()

    # Repeat the transformation on ax with full resolution
    if debug: logging.info('Repeat the transformation on the full resolution')
    transformer = create_affine_transformer_fixed(config=msk_cfg, interp_method='linear')
    ax2sax_full, m_ = transformer.predict(
        x=[np.expand_dims(ax_full_, axis=0), np.expand_dims(m_scaled_flatten, axis=0)])
    if debug: show_2D_or_3D(ax2sax_full[0][::slice_n * 3])
    plt.show()

    # create a square ident matrix slice m into it
    m_matrix = np.identity(4)
    # slice m (3,4) into identity (4,4)
    m_matrix[:3, :] = m_scaled  # this m is already scaled
    # calc inverse, flatten the matrix and cut off the last row to fit the spatial transformer input shape
    m_matrix_inverse = np.linalg.inv(m_matrix)
    m_matrix_inverse_flatten = m_matrix_inverse.flatten()[:-4]

    ax2sax_full = normalise_image(ax2sax_full)
    msk = unet_model.predict(x=[ax2sax_full])
    msk_binary = msk >= mask_threshold
    msk_binary = msk_binary.astype(np.float32)

    if debug: logging.info('Predicted mask')
    if debug: show_2D_or_3D(ax2sax_full[0][::slice_n * 3], msk_binary[0][::slice_n * 3], save=save_plots,
                            file_name=os.path.join(figure_export, 'ax2sax'))
    plt.show()

    # apply inverse to our msk and plot it together with the inverse AXtoSAX
    m_transformer = create_affine_transformer_fixed(config=msk_cfg, interp_method='linear')
    inv_msk = list()

    # compatible with three-channels in the unet - without background channel
    if msk.shape[-1] == 3:
        zero = np.zeros_like(msk[..., 0])
        inv_msk.append(zero)

    for c in range(msk.shape[-1]):
        inv_m, _ = m_transformer.predict(
            x=[msk_binary[..., c], np.expand_dims(m_matrix_inverse_flatten, axis=0)])
        inv_msk.append(inv_m[..., 0] >= mask_threshold)
    inv_msk = np.stack(inv_msk, axis=-1)

    # postprocessing
    if debug: logging.info('Predicted mask rotated to AX on original AX image - before postprocessing')
    if debug: show_2D_or_3D(ax_full_[::slice_n * 3], inv_msk[0][::slice_n * 3])
    plt.show()

    inv_msk = from_channel_to_flat(inv_msk[0])

    logging.info('DICE LV: {}'.format(metr.dice_coef_lv(ax_msk_full_gt.astype(np.float32),
                                                        transform_to_binary_mask(inv_msk).astype(np.float32)).numpy()))
    logging.info('DICE RV: {}'.format(metr.dice_coef_rv(ax_msk_full_gt.astype(np.float32),
                                                        transform_to_binary_mask(inv_msk).astype(np.float32)).numpy()))
    logging.info('DICE MYO: {}'.format(metr.dice_coef_myo(ax_msk_full_gt.astype(np.float32),
                                                          transform_to_binary_mask(inv_msk).astype(
                                                              np.float32)).numpy()))

    if postprocess:
        kernel = np.ones((5, 5), np.uint8)
        kernel_small = np.ones((3, 3), np.uint8)

        # close small holes
        inv_msk = np.stack([cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) for img in inv_msk], axis=0)
        if debug: logging.info('Predicted mask rotated to AX on original AX image - after closing')
        if debug: show_2D_or_3D(ax_full_[::slice_n * 3], inv_msk[::slice_n * 3])
        plt.show()

        # make it thinner
        inv_msk = np.stack([cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1) for img in inv_msk], axis=0)
        if debug: logging.info('Predicted mask rotated to AX on original AX image - after opening')
        if debug: show_2D_or_3D(ax_full_[::slice_n * 3], inv_msk[::slice_n * 3])
        plt.show()

    # Finally keep only one CC per label
    inv_msk = clean_3d_prediction_3d_cc(inv_msk)

    logging.info('Predicted mask rotated to AX on original AX image')
    show_2D_or_3D(ax_full_[::slice_n * 3], inv_msk[::slice_n * 3], save=save_plots,
                  file_name=os.path.join(figure_export, 'ax2sax2ax'))
    plt.show()

    # get the AX target segmentation, processed by the generator to have it in the same shape
    msk_gt_flatten = from_channel_to_flat(ax_msk_full_gt, start_c=1)
    logging.info('GT on AX')
    show_2D_or_3D(ax_full_[::slice_n * 3], msk_gt_flatten[::slice_n * 3], save=save_plots,
                  file_name=os.path.join(figure_export, 'ax_gt'))
    plt.show()

    ax_full_ = ax_full_[..., 0]
    # if debug: globals().update(locals())

    p = os.path.basename(full_file_name)

    reference_sitk = get_reference_nrrd(p, path_to_original_sitk)

    sitk_pred = undo_generator_steps(ndarray=inv_msk, cfg=msk_cfg, interpol=sitk.sitkNearestNeighbor,
                                     orig_sitk=reference_sitk)
    sitk_ax_img = undo_generator_steps(ndarray=ax_full_, cfg=msk_cfg, interpol=sitk.sitkLinear,
                                       orig_sitk=reference_sitk)
    sitk_ax_msk = undo_generator_steps(ndarray=msk_gt_flatten, cfg=msk_cfg, interpol=sitk.sitkNearestNeighbor,
                                       orig_sitk=reference_sitk)

    nda_ax = sitk.GetArrayFromImage(sitk_ax_img).astype(np.float32)
    nda_pred = transform_to_binary_mask(sitk.GetArrayFromImage(sitk_pred)).astype(np.float32)
    nda_gt = transform_to_binary_mask(sitk.GetArrayFromImage(sitk_ax_msk)).astype(np.float32)

    if debug: show_2D_or_3D(nda_ax, nda_pred)

    ensure_dir(os.path.join(export_path, 'pred'))
    ensure_dir(os.path.join(export_path, 'image'))
    ensure_dir(os.path.join(export_path, 'gt'))

    if save:
        copy_meta_and_save(sitk_pred, reference_sitk, os.path.join(export_path, 'pred', filename))
        copy_meta_and_save(sitk_ax_img, reference_sitk,
                           os.path.join(export_path, 'image', filename.replace('msk', 'img')))
        copy_meta_and_save(sitk_ax_msk, reference_sitk, os.path.join(export_path, 'gt', filename))

    logging.info('inv mask shape: {}, gt mask shape: {}'.format(nda_pred.shape, nda_gt.shape))
    logging.info('DICE LV: {}'.format(metr.dice_coef_lv(nda_gt, nda_pred).numpy()))
    logging.info('DICE RV: {}'.format(metr.dice_coef_rv(nda_gt, nda_pred).numpy()))
    logging.info('DICE MYO: {}'.format(metr.dice_coef_myo(nda_gt, nda_pred).numpy()))
    try:
        logging.info('m: {}'.format(np.reshape(m_temp[0], (3, 4))))
        logging.info('m_mod: {}'.format(np.reshape(m_mod_temp[0], (3, 4))))
        logging.info('m_scaled: {}'.format(np.reshape(m_scaled, (3, 4))))
    except Exception as e:
        logging.error(str(e))
        pass
    return nda_ax, nda_pred, nda_gt
