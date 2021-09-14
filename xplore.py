#%%

import random
import pandas as pd
import argparse
import traceback
import numpy as np
import os, sys
from multiprocessing import Process
import pickle
import tensorflow as tf
# %%
with open('/data/hz2529/zion/MVPContext/combined_feature_2021_v2/ENST00000415083.pickle', 'rb') as infl:
    res = pickle.load(infl)
# %%
