import os
from os.path import join as pjoin
import numpy as np
from pdb import set_trace
import logging

def get_config():
    config = {
        # "debug_mode": logging.DEBUG,
        "debug_mode": logging.WARNING, # Only log warnings or errors
    }
    return config