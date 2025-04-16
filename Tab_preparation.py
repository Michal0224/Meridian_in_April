import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pickle as pkl
#import openpyxl

import max27_func
from max27_func import func_27
import max27_opti_func
from max27_opti_func import opti_func_27

# Data source
file_path = "/home/michal_a_lesniewski/mmm_dump.pkl"
with open(file_path, 'rb') as file:
    mmm = pkl.load(file)

func_27(mmm)
opti_func_27(mmm)   