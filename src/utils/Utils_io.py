import os, errno
import logging
from time import time
import platform

if not platform.system() == 'Windows':
    import matplotlib as mpl

    #mpl.use('TkAgg')
import matplotlib.pyplot as plt
import json


# define some helper classes
# Define an individual logger
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
        


def save_plot(fig, path, filename='', override=False, tight=True):
    """
    Saves an matplotlib figure to the given path + filename
    If the figure exists, ad a number at the end and increase it
    as long as there is already an image with this name
    :param fig:
    :param path:
    :param filename:
    :return:
    """
    logging.debug('Trying to save to {0}'.format(path))
    ensure_dir(path)
    if tight:
        plt.tight_layout()

    i = 0
    if override:
        newname = '{}.png'.format(filename)
        fig.savefig(os.path.join(path, newname))
    else:
        while True:
            i += 1
            newname = '{}{:d}.png'.format(filename + '_', i)
            if os.path.exists(os.path.join(path, newname)):
                continue
            fig.savefig(os.path.join(path, newname))
            break
    logging.debug('Image saved: {}'.format(os.path.join(path, newname)))
    # free memory, close fig
    plt.close(fig)


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


def init_config(config, save=True):
    """
    Extract all config params (CAPITAL letters) from global or local namespace
    save a serializable version to disk
    make sure all config paths exist

    :param config:
    :param save:
    :return: config (dict) with all training/evaluation params
    """

    # make sure config path and experiment name are set
    exp = config.get('EXPERIMENT', 'UNDEFINED')
    config['CONFIG_PATH'] = config.get('CONFIG_PATH', os.path.join('reports/config/', exp))
    config['TENSORBOARD_LOG_DIR'] = config.get('TENSORBOARD_LOG_DIR', os.path.join('reports/tensorboard_logs/', exp))
    config['MODEL_PATH'] = config.get('MODEL_PATH', os.path.join('models/', exp))
    config['HISTORY_PATH'] = config.get('HISTORY_PATH', os.path.join('reports/history/', exp))

    # make sure all paths exists
    ensure_dir(config['TENSORBOARD_LOG_DIR'])
    ensure_dir(config['MODEL_PATH'])
    ensure_dir(config['CONFIG_PATH'])
    ensure_dir(config['HISTORY_PATH'])

    # Define a config for param injection and save it for usage during evaluation, save all upper key,value pairs from global namespace
    config = dict(((key, value) for key, value in config.items()
                   if key.isupper() and key not in ['HTML', 'K']))

    if save:
        # convert functions to string representations
        try:
            write_config = dict(
                [(key, value.__name__) if callable(value) else (key, value) for key, value in config.items()])
        except:
            write_config = dict(
                [(key, getattr(value, 'name', 'unknownfunction')) if callable(value) else (key, value) for key, value in config.items()])

        # save only simple data types
        write_config = dict(((key, value) for key, value in write_config.items()
                             if type(value) in [bool, int, str, float, list, dict]))

        # save to disk
        with open(os.path.join(write_config['CONFIG_PATH'], 'config.json'), 'w') as fp:
            json.dump(write_config, fp)

        # logging.info('config saved:\n {}'.format(json.dumps(write_config, indent=4, sort_keys=True)))
    return config
