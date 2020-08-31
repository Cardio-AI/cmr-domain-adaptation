#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os, errno
import logging
import logging
import numpy as np
import SimpleITK as sitk
from bs4 import BeautifulSoup
from collections import Counter
import cv2
from time import time
import glob
import concurrent.futures
from concurrent.futures import as_completed
#Console_and_file_logger('extract_segmentations_new_fast', logging.INFO)

# In[2]:

def describe_sitk(sitk_img):
    """
    log some basic informations for a sitk image
    :param sitk_img:
    :return:
    """
    if isinstance(sitk_img, np.ndarray):
        sitk_img = sitk.GetImageFromArray(sitk_img.astype(np.float32))

    logging.debug('size: {}'.format(sitk_img.GetSize()))
    logging.debug('spacing: {}'.format(sitk_img.GetSpacing()))
    logging.debug('origin: {}'.format(sitk_img.GetOrigin()))
    logging.debug('direction: {}'.format(sitk_img.GetDirection()))
    logging.debug('pixel type: {}'.format(sitk_img.GetPixelIDTypeAsString()))
    logging.debug('number of pixel components: {}'.format(sitk_img.GetNumberOfComponentsPerPixel()))

# define some helper classes
# Define an individual logger
class Console_and_file_logger():
    def __init__(self, logfile_name='Log', log_lvl=logging.INFO, path='./logs/'):
        """
        Create your own logger
        log debug messages into a logfile
        log info messages into the console
        log error messages into a dedicated *_error logfile
        :param logfile_name:
        :param log_dir:
        """

        # Define the general formatting schema
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger()

        # define a general logging level,
        # each handler has its own logging level
        # the console handler ist selectable by log_lvl
        logger.setLevel(logging.DEBUG)

        log_f = os.path.join(path, logfile_name + '.log')
        ensure_dir(os.path.dirname(os.path.abspath(log_f)))

        # delete previous handlers and overwrite with given setup
        logger.handlers = []
        if not logger.handlers:

            # Define debug logfile handler
            hdlr = logging.FileHandler(log_f)
            hdlr.setFormatter(formatter)
            hdlr.setLevel(logging.DEBUG)

            # Define info console handler
            hdlr_console = logging.StreamHandler()
            hdlr_console.setFormatter(formatter)
            hdlr_console.setLevel(log_lvl)

            # write error messages in a dedicated logfile
            log_f_error = os.path.join(path, logfile_name + '_errors.log')
            ensure_dir(os.path.dirname(os.path.abspath(log_f_error)))
            hdlr_error = logging.FileHandler(log_f_error)
            hdlr_error.setFormatter(formatter)
            hdlr_error.setLevel(logging.ERROR)

            # Add all handlers to our logger instance
            logger.addHandler(hdlr)
            logger.addHandler(hdlr_console)
            logger.addHandler(hdlr_error)

        cwd = os.getcwd()
        logging.info('{} {} {}'.format('--' * 10, 'Start', '--' * 10))
        logging.info('Working directory: {}.'.format(cwd))
        logging.info('Log file: {}'.format(log_f))
        logging.info('Log level for console: {}'.format(logging.getLevelName(log_lvl)))

def ensure_dir(file_path):
    """
    Make sure a directory exists or create it
    :param file_path:
    :return:
    """
    if not os.path.exists(file_path):
        logging.debug('Creating directory {}'.format(file_path))

        try:# necessary for parallel workers
            os.makedirs(file_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Contour():

    """
    Contour object for one pointlist of the cvi42wsx file
    represents usually one contour (RV, LV or MYO) of one 2D slice
    """

    def __init__(self, uid, image_size, pixel_size, points, sub_pixel_res, tag):
        self.uid = uid
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.points = points
        self.sub_pixel_res = sub_pixel_res
        self.tag = tag

    uid = 0
    image_size = (0, 0)
    pixel_size = (0, 0)
    sub_pixel_res = 0
    points = []


def contour_factory(contours, tag):
    """
    Expects a list of segmentation-elements
    Reads/collects all neccessary data for each contour-element
    Wraps each contour information in a contour-object,
    builds a list of Contours and returns them
    Params: list of contour-xml tags:
    e.g. call:
    xml_file = 'D:\\git\\cardio\\data\\processed\\Segmentations\\0AE4R74L190001011238291.3.12.2.1107.5.99.2.1013.30000008030612131778100072179.cvi42wsx'
    with open(xml_file) as fp:
    soup = BeautifulSoup(fp, 'lxml')
    saendocardialContour = soup.find_all('hash:item', {'hash:key' : 'saendocardialContour'})

    """
    cont = {}
    logging.debug('building {} contours'.format(len(contours)))
    for contour in contours:
        # print(contour)
        uid = contour.parent.parent.attrs.get('hash:key')
        image_width = contour.find('hash:item', {'hash:key': 'ImageSize'}).find('size:width').text
        image_height = contour.find('hash:item', {'hash:key': 'ImageSize'}).find('size:height').text
        pixel_width = contour.find('hash:item', {'hash:key': 'PixelSize'}).find('size:width').text
        pixel_height = contour.find('hash:item', {'hash:key': 'PixelSize'}).find('size:height').text
        sub_pixel = contour.find('hash:item', {'hash:key': 'SubpixelResolution'}).text
        points_t = contour.find('hash:item', {'hash:key': 'Points'})

        points_x = [int(int(point.text) / int(sub_pixel)) for point in points_t.find_all('point:x')]
        points_y = [int(int(point.text) / int(sub_pixel)) for point in points_t.find_all('point:y')]

        points = list(zip(points_x, points_y))

        # create a Contour class for each contour
        cont[str(uid)] = Contour(uid,
                                 (image_width, image_height),
                                 (pixel_width, pixel_height),
                                 points,
                                 sub_pixel, tag)
    return cont

def extract_la_contours(soup):
    """
        Expects the Segmentation-XML-File as Beautifulsup-object
        Returns: a dictionary with segmentation-tag : list of segmentation-tag elements
        """
    c = {}
    contours = {'lalaContour': [],
                'laraContour': []}

    all_items = list(soup.find_all('hash:item'))
    lala_c = []
    lara_c = []

    for item in all_items:
        if item.attrs.get('hash:key', False) == "lalaContour":
            lala_c.append(item)
        elif item.attrs.get('hash:key', False) == "laraContour":
            lara_c.append(item)


    contours['lalaContour'] = lala_c
    contours['laraContour'] = lara_c

    """
    convert to contour-objects
    Expects a dictionary object with:
    segmentation-tag : list of segmentation-tag elements
    converts the list of segmentation-elements
    into a flat dictionary with {uid : [contour]}
    """

    for tag, contour_elements in contours.items():
        if len(contour_elements) > 0:
            logging.debug('extract all {} segmentations'.format(tag))
            # transform all contour elements in contour objects
            contours = contour_factory(contour_elements, tag)
            # update contours dictionary
            # we might have multiple contours per slice
            # we have a list of contours with tuples of (uid, contour-obj)
            # we need to group all contours of one image with a list of contours
            for uid, cont in contours.items():
                # if this uid/image has already an
                if c.get(uid, False):
                    c[uid].append(cont)
                else:
                    c[uid] = [cont]
    return c

def extract_sa_contours(soup):
    """
    Expects the Segmentation-XML-File as Beautifulsup-object
    Returns: a dictionary with segmentation-tag : list of segmentation-tag elements
    'sarvepicardialContour': [] is not contoured in the current datasets
    """
    c = {}
    contours = {'saendocardialContour': [],
                'saepicardialContour': [],
                'sarvendocardialContour': [],
                'lineRoiContour0001': [],
                'lineRoiContour0002': [],
                'lineRoiContour': [],
                }

    all_items = list(soup.find_all('hash:item'))
    myo_c = []
    rv_c = []
    lv_c = []
    marker1_c = []
    marker2_c = []
    marker_c = []
    for item in all_items:
        if item.attrs.get('hash:key', False) == "sarvendocardialContour":
            rv_c.append(item)
        elif item.attrs.get('hash:key', False) == "saendocardialContour":
            lv_c.append(item)
        elif item.attrs.get('hash:key', False) == "saepicardialContour":
            myo_c.append(item)
        elif item.attrs.get('hash:key', False) == "lineRoiContour0001":
            marker1_c.append(item)
        elif item.attrs.get('hash:key', False) == "lineRoiContour0002":
            marker2_c.append(item)
        elif item.attrs.get('hash:key', False) == "lineRoiContour":
            marker_c.append(item)


    contours['sarvendocardialContour'] = rv_c
    contours['saendocardialContour'] = lv_c
    contours['saepicardialContour'] = myo_c
    contours['lineRoiContour0001'] = marker1_c
    contours['lineRoiContour0002'] = marker2_c
    contours['lineRoiContour'] = marker_c

    """
    convert to contour-objects
    Expects a dictionary object with:
    segmentation-tag : list of segmentation-tag elements
    converts the list of segmentation-elements
    into a flat dictionary with {uid : [contour]}
    """

    for tag, contour_elements in contours.items():
        if len(contour_elements) > 0:
            logging.debug('extract all {} segmentations'.format(tag))
            # transform all contour elements in contour objects
            contours = contour_factory(contour_elements, tag)
            # update contours dictionary
            # we might have multiple contours per slice
            # we have a list of contours with tuples of (uid, contour-obj)
            # we need to group all contours of one image with a list of contours
            for uid, cont in contours.items():
                # if this uid/image has already a contour, make always a list of contours
                if c.get(uid, False):
                    c[uid].append(cont)
                else:
                    c[uid] = [cont]
    return c

def extract_contours(soup, args):
    """
    Decide if this file is a SA or LA exported xml file, should be given in the XML file with SAX_... or LAX_...
    If nothing is given (original GCN dataset and first export, SAX is default
    :param soup:
    :param args:
    :return:
    """
    if args.get('view', 'sax').lower() == 'lax':
        return extract_la_contours(soup)
    else:
        return extract_sa_contours(soup)


def sort_dicoms(dicom_images, sort_for_time=True):
    """
    Sort all slices by trigger-time and origin or origin

    :param dicom_images:
    :param sort_for_time: Bool, if True: return (triggertime,*origin), else return origin
    :return:
    """

    logging.debug('images to sort: {}'.format(len(dicom_images)))

    def origin_sort(dicom_image):
        """
        Helper, which returns a tuple as argument for sorted()
        if sort_for_time: return (triggertime,*origin), else return origin
        :param dicom_image:
        :return:
        """
        origin = dicom_image.GetOrigin()

        if sort_for_time:
            # triggertime
            time = dicom_image.GetMetaData('0018|1060')
            #logging.debug('trigger time: {}'.format(time))
            return (float(time), origin[2], origin[1], origin[0])
        else:
            return (origin[2], origin[1], origin[0])

    sorted_dicom_images = sorted(dicom_images, key=origin_sort)

    return sorted_dicom_images

def get_timesteps(dicom_images):
    """
    calculate the timesteps of one volume by summing up the origins
    all slices with same origin represents the timesteps of one slice
    """

    origins = [img.GetOrigin() for idx, img in enumerate(dicom_images)]
    counter = Counter(origins)
    # make sure to get another timestep size if this one is 1
    iter_ = iter(counter)
    steps = counter[next(iter_)]

    # in some cases we have a volume where each origin occures n times
    # but one slice/origin is missplaced and occures only once
    # to avoid volumes with 1 timestep (except of LA volumes)
    # try to get a greater number of timesteps, this is a hack
    if steps == 1:
        steps = counter[next(iter_)]

    return steps

def get_contour(sitk_img, contour_dict, mixed_mode=False, mask=False, args={}):
    """
    Check if there is a contour object for this dicom slice given
    writes contours/masks into dicom slice,
    returns an image, a mask, or a contour slice

    sitk_img = sitk-image
    contour_dict = a flat dictionary with {uid : [contours]}
    class Contour():
        self.uid = uid
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.points = points
        self.sub_pixel_res = sub_pixel_res
        self.tag = tag


    mixed_mode = True --> Use image slice as background
    mixed_mode = False --> Use emtpy slice as background

    mask = True --> fill contour, create masks
    mask = False --> Use plain contours

    returns (modified) dicom slice
    """

    # values to use for contour drawing

    tags = {
            'sarvepicardialContour': 0,
            'sarvendocardialContour': 1,
            'saepicardialContour': 2,
            'saendocardialContour': 3,
            'lineRoiContour0001': 4,
            'lineRoiContour0002': 5,
            'lineRoiContour': 6,
            'laraContour': 7,
            'lalaContour': 8

    }


    # check if this slice SOP-UID is in the contour dictionary
    # check if contour is given for this slice
    contour = contour_dict.get(sitk_img.GetMetaData('0008|0018'), False)

    if contour:  # if there is a contour for this slice given, this contour will be a list of contour objects,
        # if no contour is given for this slice contour will be False

        # go through contours in reverse order, to draw first the outer line, than inner
        contour = reversed(contour)  # rvepi, rvendo, epi, endo

        # which background should be used for this slice
        if mixed_mode:  # use image slice as background
            nda = sitk.GetArrayFromImage(sitk_img)
        else:  # use empty slice as background
            temp_nda = sitk.GetArrayFromImage(sitk_img)
            shape = temp_nda.shape
            dtype = temp_nda.dtype
            nda = np.zeros(shape, dtype=dtype)

        # create mask or contour slice
        if mask:  # create masks from point list with given tag values
            nda_ = nda[0, :, :]
            for cont in contour:
                cv2.fillPoly(nda_, pts=[np.array(cont.points)], color=tags[cont.tag])
                nda = nda_[np.newaxis, :, :]

        else:  # create contour points with given tag-values
            for cont in contour:
                for point in cont.points:
                    nda[0, point[1], point[0]] = tags[cont.tag]

        contour_img = nda


    else:  # no contour given

        if mixed_mode:  # mixed mode = True --> return image slice
            contour_img = sitk_img

        else:  # mixed_mode = False --> return empty slice
            temp_nda = sitk.GetArrayFromImage(sitk_img)
            nda = np.zeros(temp_nda.shape, dtype=temp_nda.dtype)
            contour_img = nda

    return contour_img


def get_series_id(path_, uids):
    """
    checks if image uid is in uids
    returns the series id for a dicom image if true,
    returns None otherwise
    :param path_: full file path to a dicom image
    :param uids: sitk uid
    :return: series or None
    """

    reader = sitk.ImageFileReader()
    reader.SetFileName(path_)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    # read the image uid compares with given uids list
    if reader.GetMetaData('0008|0018') in uids:
        # if this is a valid dicom image return the series id to find the slices next to this slice
        return reader.GetMetaData('0020|000e')


def is_valid_dicom(path_, seried_ids):

    """
    Gets a full path to a dicom image and a list of series ids
    loads the dicom image and reads the series id,
    returns True if this series id is in the given list
    :param path_:
    :param seried_ids:
    :return:
    """

    reader = sitk.ImageFileReader()
    reader.SetFileName(path_)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    series_id = reader.GetMetaData('0020|000e')
    if series_id in seried_ids:
        return True
    else:
        return False


def get_valid_dicom_image(contours, args):

    """
    Load all files within a study folder
    load each file, check if file is valid by the sop-uids from the xml file
    load the series id of this file
    load all files with that series id
    by this we are able to load the slices between the labels

    :param contours:
    :param args:
    :return: a list of dicom images
    """

    logging.debug(args['src'])
    # get all dicom file names in the current folder
    dicom_f_names = set(sorted(glob.glob(os.path.join(args['src'], '**/*.dcm'), recursive=True)))

    # compare each sitk uid with the uids from the xml file, if they match, collect the series ids
    # filter None from list
    logging.debug(
        '----------------    Search for series-ids in dicom directory with image-uids from xml file ----------------')
    series_ids = set(filter(None, [get_series_id(p_, contours) for p_ in dicom_f_names]))
    logging.debug('----------------    Found {} series-ids in dicom directory ----------------'.format(len(series_ids)))

    # compare each sitk series id with the collected valid series ids, if they match, load the dicom image
    logging.debug(
        '----------------    Load dicom images with given series-ids {} -----------------'.format(series_ids))
    dicom_images = [sitk.ReadImage(path_) for path_ in dicom_f_names if is_valid_dicom(path_, series_ids)]

    # sometimes no series ids could be found, or mapped
    # if we know that all dicom files are related to this volume
    # we can load all of them and try to build the volume
    if len(dicom_images) == 0:
        logging.debug(
            '----------------    No uids could be matched, or no valid series id found, load all dicom images! {} -----------------'.format(series_ids))
        dicom_images = [sitk.ReadImage(path_) for path_ in dicom_f_names]

    logging.debug('----------------    Dicoms loaded: {} -----------------'.format(len(dicom_images)))

    return dicom_images

def build_volume(contour_as_dict, dicoms_sorted, args):
    """
    Builds 3 x 4D nrrd volumes:
    - dicom images sorted
    - contours sorted (currently excluded/ not necessary)
    - masks sorted
    :param contour_as_dict:
    :param dicoms_sorted:
    :param args:
    :return:
    """

    image_nda = []
    #contour_nda = []
    mask_nda = []

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
    timesteps = int(get_timesteps(dicoms_sorted))
    # check if we might miss some dicom images
    if not (len(dicoms_sorted) / timesteps).is_integer():
        logging.error('number of dicom slices {}/ timesteps {} is not an integer, trying to round z but maybe there are wrong dicom files, check: \n'
                      'patient: {} \n'
                      'view: {} \n'
                      'xml_file: {}\n'
                      'dicom-src: {}'.format(len(dicoms_sorted), timesteps, args['patient'], args['view'], args['xml_file'], args['src']))
    slices = int(len(dicoms_sorted) / timesteps)
    logging.debug('Building volumes for patient: {}'.format(args['patient']))
    logging.debug('images: {}'.format(len(dicoms_sorted)))
    logging.debug('timesteps: {}'.format(timesteps))
    logging.debug('slices: {}'.format(slices))
    logging.debug('copy metadata from:')
    describe_sitk(dicoms_sorted[0])
    lower_boundary = 0
    upper_boundary = timesteps

    # build new spacing with z = slice thickness and t = 1
    spacing_3d = dicoms_sorted[0].GetSpacing()
    origin = dicoms_sorted[0].GetOrigin()
    direction = dicoms_sorted[0].GetDirection()

    z_spacing = spacing_3d[2]
    if z_spacing == 1:  # some volumes dont have z spacing, they have one series per slice, use the slice thickness (0018,0050)
        z_spacing = int(get_metadata_maybe(dicoms_sorted[0], '0018|0050', 6))  # default is 6
        logging.debug('no spacing given, use slice thickness: {} as z-spacing '.format(z_spacing))
    spacing = (spacing_3d[0], spacing_3d[1], z_spacing, 1)

    # init origin
    origin_3d = None

    # iterate over all slices
    for idx in range(slices):
        # take all timesteps of this slice,
        # sort timesteps of this slice by Triggertime
        image_volume_aslist = dicoms_sorted[lower_boundary:upper_boundary].copy()
        image_volume_aslist = sort_dicoms(image_volume_aslist, sort_for_time=True)

        # image_volume_aslist[0] = lowest slice of first volume, use it as origin
        if origin_3d == None:
            origin_3d = image_volume_aslist[0].GetOrigin()
            origin = (origin_3d[0], origin_3d[1], origin_3d[2], 0)

        img_t = []
        #contour_t = []
        mask_t = []
        # iterate over all timesteps of this slice
        for sitk_img in image_volume_aslist:
            #contour_t.append(
            #    np.squeeze(get_contour(sitk_img, contour_dict=contour_as_dict, mixed_mode=False, mask=False, args=args), axis=0))
            mask_t.append(
                np.squeeze(get_contour(sitk_img, contour_dict=contour_as_dict, mixed_mode=False, mask=True, args=args), axis=0))
            img_t.append(np.squeeze(sitk.GetArrayFromImage(sitk_img), axis=0))

        image_nda.append(np.stack(img_t, axis=0))
        #contour_nda.append(np.stack(contour_t, axis=0))
        mask_nda.append(np.stack(mask_t, axis=0))

        lower_boundary = lower_boundary + timesteps
        upper_boundary = upper_boundary + timesteps

    # stack along the z axis (t,z,x,y)
    new_img_clean = np.stack(image_nda, axis=1).astype(np.float32)
    #new_contour = np.stack(contour_nda, axis=1).astype(np.int64)
    new_mask = np.stack(mask_nda, axis=1).astype(np.uint8)

    # sitk.GetImageFromArray cant handle 4d images
    sitk_images = [sitk.GetImageFromArray(vol) for vol in new_img_clean]
    # copy rotation/direction
    _ = [img.SetDirection(direction) for img in sitk_images]
    new_img_clean = sitk.JoinSeries(sitk_images)
    #new_contour = sitk.JoinSeries([sitk.GetImageFromArray(vol) for vol in new_contour])
    sitk_masks = [sitk.GetImageFromArray(vol) for vol in new_mask]
    # copy rotation/direction
    _ = [img.SetDirection(direction) for img in sitk_masks]
    new_mask = sitk.JoinSeries(sitk_masks)

    size = new_img_clean.GetSize()
    dimension = new_img_clean.GetDimension()
    logging.debug('volumes created for {}'.format(args['xml_file']))
    logging.info("Image size: {}".format(size))
    logging.debug("Image dimension: {}".format(dimension))
    logging.info("Image Spacing: {}".format(spacing))
    logging.debug('Writing images ...')

    # anyway, copy image information to new volume
    sitk_img = dicoms_sorted[0]
    for tag in sitk_img.GetMetaDataKeys():
        value = get_metadata_maybe(sitk_img, tag)
        new_img_clean.SetMetaData(tag, value)
        #new_contour.SetMetaData(tag, value)
        new_mask.SetMetaData(tag, value)

    new_img_clean.SetSpacing(spacing)
    #new_contour.SetSpacing(spacing)
    new_mask.SetSpacing(spacing)

    new_img_clean.SetOrigin(origin)
    #new_contour.SetOrigin(origin)
    new_mask.SetOrigin(origin)

    # direction cant be copied, matrix length does not match (sitk exception)
    #new_img_clean.SetDirection(direction)
    # new_contour.SetOrigin(origin)
    #new_mask.SetDirection(direction)

    sitk.WriteImage(new_img_clean, args['out_file_clean'])
    logging.debug('Image: {} done'.format(args['out_file_clean']))
    #sitk.WriteImage(new_contour, args['out_file_c'])
    #logging.debug('Image: {} done'.format(args['out_file_c']))
    sitk.WriteImage(new_mask, args['out_file_m'])
    logging.debug('Image: {} done'.format(args['out_file_m']))

    logging.debug('Patient: {} done!'.format(args['patient']))
    #describe_sitk(new_img_clean)


# In[12]:


def convert_xml_to_nrrd(args):
    """
    Entry-point for one extraction, extract contours from one xml file
    and maps them to the corresponding dicom images
    :param args:
    :return:
    """

    path = './reports/'

    # open xml file and create soup
    with open(args['xml_file']) as fp:
        soup = BeautifulSoup(fp, 'lxml')

    # load all segmentation contours
    contour_as_dict = extract_contours(soup, args)
    logging.debug('Converted: {} contours'.format(len(contour_as_dict)))

    # load related dicom images
    dicom_images = get_valid_dicom_image(contours=contour_as_dict, args=args)
    dicoms_sorted = sort_dicoms(dicom_images, sort_for_time=False)

    if len(dicoms_sorted) <= 1: # log patients with a mapped dicom folder, but only one or zero mapped dicom file
        logging.error('No dicoms found for patient: {} in view: {}'.format(args['patient'], args['view']))
        logging.error('extracted contour: {}'.format(contour_as_dict))
        return None
    build_volume(contour_as_dict, dicoms_sorted, args)


def build_config(xml_file, dicom_folder, path_to_export, patient_id):
    """
    build a dictionary with the directory paths for one Contour-Volume matching/export
    :param xml_file:
    :param dicom_folder:
    :param path_to_export:
    :param patient_id:
    :return:
    """
    try:
        f_name_clean = 'volume_clean.nrrd'
        f_name_segment = 'volume_contour.nrrd'
        f_name_mask = 'volume_mask.nrrd'
        # default is sa
        view = 'sax'

        # some dicom folder are with '.' in the name some not, make sure to get the patient folder name
        #dicom_patient = os.path.basename(os.path.normpath(dicom_folder))
        dicom_patient = os.path.split(dicom_folder)[1]

        # for the included
        if len(dicom_patient) is 0:
            xml_f = os.path.basename(os.path.normpath(xml_file))
            view = xml_f.split('_')[0]
            if len(view) ==0:
                view = 'undefined'
            # make sure to work if the view has a different length
            patient_uid = xml_f.split('_')[1][0:8]
            study_date_raw = xml_f.split('_')[1][8:17]
            study_date = study_date_raw[0:4] + "-" + study_date_raw[4:6] + "-" + study_date_raw[6:8]

            dicom_patient = '{:04d}-{}_{}'.format(0, patient_uid, study_date)

        # define subdirectories for the LAX & SAX
        path_to_export = os.path.join(path_to_export, view)

        args = {}
        args['view'] = view
        args['patient'] = dicom_patient
        args['xml_file'] = xml_file
        args['src'] = dicom_folder
        args['dest'] = os.path.join(path_to_export, dicom_patient)
        args['out_file_clean'] = os.path.join(path_to_export, dicom_patient + "_" + f_name_clean)
        args['out_file_c'] = os.path.join(path_to_export, dicom_patient + "_" + f_name_segment)
        args['out_file_m'] = os.path.join(path_to_export, dicom_patient + "_" + f_name_mask)

        # ensure export dir exists
        ensure_dir(path_to_export)

        logging.debug('Build config for:')
        for key, value in args.items():
            logging.debug('{}: {}'.format(key, value))
    except Exception as e:
        logging.error('Config init failed! xml file: {}, dicom-folder: {}\n Local variable stack: {}'.format(xml_file, dicom_folder, locals()))
        raise e
    return args


# In[14]:


def map_xml_to_study(xml, dicom_study_folders=None, dicom_included=False):
    """
    extracts patient id etc. from xml file name
    transforms the format and search for that expression in all dicom folders
    :param xml:
    :param dicom_study_folders:
    :param dicom_included:
    :return:
    """
    xml = os.path.abspath(xml)

    # parse patient uid, date from xml file and format it
    if dicom_included: # match by view (LAX/SAX) and study date raw
        xml_file = os.path.basename(xml)
        view = xml_file.split('_')[0]
        patient_uid = xml_file[4:12]
        study_date_raw = xml_file[12:21]
        study_date = study_date_raw[0:4] + "-" + study_date_raw[4:6] + "-" + study_date_raw[6:8]
    else: # match by patient uid and study date
        xml_file = os.path.basename(xml)
        patient_uid = xml_file[:8]

        study_date = xml_file[8:17]
        study_date = study_date[0:4] + "-" + study_date[4:6] + "-" + study_date[6:8]
        logging.debug(patient_uid)
        logging.debug(study_date)

    # match by integrated dicom folder
    if dicom_included:
        # get the path to the xml file
        path_ = os.path.dirname(xml)
        # get all folders within this path
        included_folders = glob.glob(os.path.join(path_, '*/'))

        # only one folder found, take it
        if len(included_folders) is 1:
            matched = included_folders

        # if there are more folders in this patient folder, match xml file and folder by the view (SAX/LAX) and study date raw
        if len(included_folders) > 1: # make sure to be case insensitive
            matched = [folder for folder in included_folders if view.lower() in folder.lower() and study_date_raw.lower() in folder.lower()]
    else:     # match xml file and dicom folder according to the patient-UID and the study date (the original GCN dicom export has a different formatting for the date)
        matched = [folder for folder in dicom_study_folders if patient_uid in folder and study_date in folder]

    logging.debug(len(matched))
    if matched:
        # if there are two matches, return only the first
        #logging.debug('Found dicom folder: {} for XML file: {}.'.format(matched[0], xml_file))
        matched = matched[0]
    else:
        logging.debug('Found NO dicom folder for XML file: {}.'.format(xml_file))
        logging.debug('Searched for extracted patient-id: {} and date: {}'.format(patient_uid, study_date))
        matched = None
    # return anyway to track the the matched and missmatched
    return xml, matched


# In[15]:
def async_extract(xml_f, folder_n, path_to_export, total_n, patient):
    t1 = time()
    logging.info('-----'*10)
    logging.info('processing patient: {} ({}) of total: {}'.format(patient, os.path.basename(os.path.normpath(folder_n)), total_n))

    args = build_config(xml_f, folder_n, path_to_export, patient)
    convert_xml_to_nrrd(args)
    return patient, time() - t1

def create_volumes(path_to_xml, path_to_dicom=None, path_to_export='export/', max_workers=1, spawn_processes=False):
    """
    Loads all XML files in path_to_xml
    Lists all patients folders in path_to_dicom_folders
    maps the xml file to patient dicom folder
    start contour extraction and dicom filter, sort and 4d building
    """

    xml_extension = '.cvi42wsx'
    dicom_study_folders = list()
    dicom_included = False

    if not path_to_dicom:
        dicom_included = True
        logging.info('no dicom path given, expecting a dicom folder within the xml folder.')

    # get all xml files
    if os.path.isdir(path_to_xml):

        if dicom_included:
            xml_files = glob.glob(os.path.join(path_to_xml, '**/*{}'.format(xml_extension)))
        else:
            # search xml files in patient sub-folders
            xml_files = glob.glob(os.path.join(path_to_xml, '**/*{}'.format(xml_extension)))

            if len(xml_files) == 0: # if no xml files found expect them in a flat folder structure
                xml_files = [os.path.join(path_to_xml, f) for f in os.listdir(path_to_xml) if
                             os.path.isfile(os.path.join(path_to_xml, f))]
                # filter only cvi42wsx files
                xml_files = [file for file in xml_files if os.path.splitext(file)[1] == xml_extension]
        logging.info('Found {} xml-files.'.format(len(xml_files)))
        logging.debug(xml_files)
    else:
        logging.error('Error with path_to_xml: {}'.format(path_to_xml))
        return

    # get all study-folders
    if not dicom_included:
        if os.path.isdir(path_to_dicom):

            dicom_study_folders = [os.path.join(path_to_dicom, f) for f in os.listdir(path_to_dicom) if
                                   os.path.isdir(os.path.join(path_to_dicom, f))]
            logging.info('found {} Patient-Folders.'.format(len(dicom_study_folders)))
            logging.debug(dicom_study_folders)

        else:
            logging.error('Error with path_to_dicom: {}'.format(path_to_dicom))
            return

    # ensure export path exists
    ensure_dir(path_to_export)

    # get a list of (xml-files, study-folder) where we have a segmentation file and a dicom folder
    segmented_studies = [map_xml_to_study(xml, dicom_study_folders, dicom_included) for xml in xml_files]

    mapped_studies = [(xml, folder) for xml, folder in segmented_studies if folder is not None]
    ignored_files = [xml for xml, folder in segmented_studies if folder is None]
    logging.info('Matched: {} xml files to a dicom folder'.format(len(mapped_studies)))
    logging.debug(mapped_studies)
    logging.info('Didn\'t find a dicom folder for: {} xml files'.format(len(ignored_files)))
    [logging.info('No matching for: {}'.format(file)) for file in ignored_files]

    t1 = time()

    if spawn_processes:

        # spawn one process per worker
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            patient = 0
            for xml_f, folder_n in mapped_studies:
            
                patient = patient + 1
                try:
                    future = executor.submit(async_extract, xml_f, folder_n, path_to_export, len(mapped_studies), patient)
                except Exception as e:
                    logging.error('Exception {} - Failed with xml: {} and dicom folder: {}'.format(str(e), str(xml_f), str(folder_n)))
            
                futures.add(future)

            for future in as_completed(futures):
                try:
                    patient, needed_time = future.result()
                    logging.info('process with patient {} finished after {:0.3f} sec.'.format(patient, needed_time))
                except Exception as e:
                    logging.error(' failed to process patient: {} with error: {}'.format(patient, str(e)))
                    print(e)

    else:
        # run in one process, could run with multi thread (# workers = # threads), but slower caused by io
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

            t1 = time()
            # extract contours, filter dicoms, sort slices and build 4d volumes
            patient = 0
            for xml_f, folder_n in mapped_studies:

                patient = patient + 1
                try:
                    executor.submit(async_extract, xml_f, folder_n, path_to_export, len(mapped_studies), patient)
                except Exception as e:
                    logging.error(
                        'Exception {} - Failed with xml: {} and dicom folder: {}'.format(str(e), str(xml_f), str(folder_n)))

    logging.info('All {} patients done in {:0.3f} s'.format(len(mapped_studies), time() - t1))
    

# In[17]:
if __name__ == '__main__':

    """
    Script to extract contours from circle export files (*.cvi42wsx) and map them to dicom images.
    """

    working_dir = os.getcwd()
    path_to_xml = os.path.join(working_dir, 'data/processed/Segmentations/small_xml/')

    path_to_dicom = 'data/raw/ahf_export/'
    path_to_export = os.path.join(working_dir, 'data/processed/sorting_tests/')

    create_volumes(path_to_xml, path_to_dicom, path_to_export)
