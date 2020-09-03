# central way to import common libs for the notebooks

import logging
import matplotlib.pyplot as plt
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

from src.utils.Utils_io import Console_and_file_logger, ensure_dir

# make jupyter able to display multiple lines of variables in one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"

# ignore matplotlib info logs
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 