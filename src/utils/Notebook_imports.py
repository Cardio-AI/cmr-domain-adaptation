# central way to get all standard library available in one import

import logging
import os
import json
import yaml
import glob
import datetime
from ipywidgets import interact
from ipywidgets import interact_manual
import SimpleITK as sitk
import random
from collections import Counter
import seaborn as sb

#from keras.utils import plot_model
#from src.utils.Tensorflow_helper import choose_gpu_by_id
from src.utils.Utils_io import Console_and_file_logger, ensure_dir

# make jupyter able to display multiple lines of variables in one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"

mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 