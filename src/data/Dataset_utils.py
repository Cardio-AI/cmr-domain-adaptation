import os
import json
import logging
from src.utils.utils_io import Console_and_file_logger, ensure_dir
from src.models.ModelUtils import load_pretrained_model
import pandas as pd
import platform



def load_config(config_file_path, load=False):

    """
    Load a config file
    Try to
    :param config_file_path: (string) path to the config json file
    :param load: (bool), whether the model and config related files should be loaded
    :return: a dictionary with {'config': cfg, 'model':tf.keras.model}
    """

    # create local namespace object
    glob_ = {}
    
    with open(config_file_path, encoding='utf-8') as data_file:
        config = json.loads(data_file.read())

    # linux / windows paths
    if platform.system() == 'Linux':
        config = dict([(key, value.replace('\\', '/')) if type(value) is str else (key, value) for key, value in config.items()])


    glob_['config'] = config
    Console_and_file_logger(config['EXPERIMENT'], logging.INFO)
    # load all experiment files if user asked for it
    if load:

        try:
            # load trainings history
            logging.info('loading trainings history...')
            glob_['df_history'] = pd.read_csv(os.path.join(config['HISTORY_PATH'], 'history.csv'),index_col=0)
            logging.info('history {} loaded'.format(os.path.join(config['HISTORY_PATH'], 'history.csv')))
        except Exception as e:
            logging.info('No history found! --> {}'.format(os.path.join(config['HISTORY_PATH'], 'history.csv')))
            logging.debug(str(e))

        try:
            # load model
            model = load_pretrained_model(config, comp=False)
            glob_['model'] = model
        except Exception as e:
            logging.info(str(e))
            # make sure we dont use earlier models
            glob_['model'] = None
        

        try:
            # load past evaluations done with that model
            logging.info('loading past evaluation scores...')
            import_path = os.path.join('reports/evaluation', config['EXPERIMENT'])
            #glob_['evaluation_score'] = pd.read_csv(os.path.join(import_path,'evaluation_score.csv')).set_index('Evaluation')
            logging.info('past evaluation scores {} loaded'.format(os.path.join(import_path, 'evaluation_score.csv')))
        except Exception as e:
            # delete the evaluation score object from current namespace
            # if past models & evaluations have been done, to avoid mixing them up
            #glob_['evaluation_score'] = None
            logging.info(str(e))
    
    return glob_
