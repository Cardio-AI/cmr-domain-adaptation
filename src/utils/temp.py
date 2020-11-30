import glob
import logging


def undo_generator_steps(ndarray, p, orig_file_path='/mnt/ssd/data/gcn/gcn_05_2020_ax_sax_86_2nd/AX/'):
    # undo the generator steps
    # load 4d volume, to copy the right metadata
    # file_4d = glob.glob('data/raw/GCN_2nd/4D/**/masks/{}*.nrrd'.format(p))[0]
    try:
        # file_4d = glob.glob('/mnt/ssd/data/ACDC/original/all/**/{}*4d.nii.gz'.format(p))[0]
        file_4d = glob.glob('{}{}*.nrrd'.format(orig_file_path, p))[0]
        # file_4d = glob.glob('/mnt/ssd/git/3d-mri-domain-adaption/data/raw/gcn_05_2020_ax_sax_86/SAX/{}*.nrrd'.format(p))[0]
        # file_4d = glob.glob('data/raw/GCN/4D/**/images/{}*.nrrd'.format(p))[0]
        # file_4d = glob.glob('data/raw/peters/4D/all/images/{}*nrrd'.format(p))[0]
        # file_4d = glob.glob('data/raw/gcn_05_2020_ax_sax_86/AX/{}*clean.nrrd'.format(p))[0]
        orig_sitk = sitk.ReadImage(file_4d)
        orig_size_ = orig_sitk.GetSize()
        orig_spacing_ = orig_sitk.GetSpacing()
        orig_size = list(reversed(orig_size_))
        orig_spacing = list(reversed(orig_spacing_))

        logging.info('original shape: {}'.format(orig_size_))
        logging.info('original spacing: {}'.format(orig_spacing_))
        logging.info('reverse original shape: {}'.format(orig_size))
        logging.info('reverse original spacing: {}'.format(orig_spacing))
    except Exception as e:
        logging.error('no metadata and spacing copied: {}'.format(str(e)))
        img = None
    # numpy has the following order: h,w,c (or z,h,w,c for 3D)

    h_w_size = orig_size[2:]
    h_w_spacing = orig_spacing[2:]
    w_h_size_sitk = list(reversed(h_w_size))
    w_h_spacing_sitk = list(reversed(h_w_spacing))

    # calculate the size of the image before crop or pad
    new_size = (np.array(h_w_size) * np.array(h_w_spacing)) / np.array(config['SPACING'])
    new_size = [int(np.ceil(i)) for i in new_size]

    # pad, crop to original physical size in current spacing, slice wise
    from src.data.Preprocess import center_crop_or_pad_2d, resample_3D
    predictions = list()
    gts = list()
    images = list()

    logging.info('orig height/width size: {}'.format(h_w_size))
    logging.info('orig height/width spacing: {}'.format(h_w_spacing))
    logging.info('pred shape: {}'.format(pred.shape))
    logging.info('intermediate size after : {}'.format(new_size))

    #
    for i in range(pred.shape[0]):
        # get intermediate size, undo resample and crop/pad
        # by first crop and pad with intermediate size(which reflects the size we have in the generator after resampling)
        # second resample with original spacing

        img_ = volume[i]
        msk_ = pred[i]
        gt_ = gt[i]

        temp = img_.copy()
        # logging.error('before crop_pad: {}'.format(img_.shape))
        img_, msk_, _ = center_crop_or_pad_2d(img_, msk_, new_size)
        _, gt_, _ = center_crop_or_pad_2d(temp, gt_, new_size)
        # logging.error('after crop_pad: {}'.format(img_.shape))

        # resample, set current spacing
        img_ = sitk.GetImageFromArray(img_)
        img_.SetSpacing(config['SPACING'])
        msk_ = sitk.GetImageFromArray(msk_)
        msk_.SetSpacing(config['SPACING'])
        gt_ = sitk.GetImageFromArray(gt_)
        gt_.SetSpacing(config['SPACING'])

        img_ = resample_3D(img_, size=w_h_size_sitk, spacing=w_h_spacing_sitk, interpolate=sitk.sitkLinear)
        msk_ = resample_3D(msk_, size=w_h_size_sitk, spacing=w_h_spacing_sitk)
        gt_ = resample_3D(gt_, size=w_h_size_sitk, spacing=w_h_spacing_sitk)

        img_ = sitk.GetArrayFromImage(img_)
        msk_ = sitk.GetArrayFromImage(msk_)
        gt_ = sitk.GetArrayFromImage(gt_)
        # logging.error('after resample: {}'.format(img_.shape))

        images.append(img_)
        predictions.append(msk_)
        gts.append(gt_)
        # logging.info(len(images))

    volume = np.stack(images, axis=0)
    pred = np.stack(predictions, axis=0)
    gt = np.stack(gts, axis=0)

    # show_2D_or_3D(volume, gt)

    logging.info('predictions size after resize: {}'.format(pred.shape))

def get_reference_nrrd(p, orig_file_path):
    try:
        file_4d = glob.glob('{}{}'.format(orig_file_path, p))[0]
        orig_sitk = sitk.ReadImage(file_4d)
        orig_size_ = orig_sitk.GetSize()
        orig_spacing_ = orig_sitk.GetSpacing()
        orig_size = list(reversed(orig_size_))
        orig_spacing = list(reversed(orig_spacing_))

        logging.info('original shape: {}'.format(orig_size_))
        logging.info('original spacing: {}'.format(orig_spacing_))
        logging.info('reverse original shape: {}'.format(orig_size))
        logging.info('reverse original spacing: {}'.format(orig_spacing))
    except Exception as e:
        logging.error('no file found: {}'.format(str(e)))
        orig_sitk = None
    return orig_sitk