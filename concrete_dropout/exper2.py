# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from evaluation2 import *

#==============================================================================
# BlogFeedback (blog)

data = pd.read_csv('../data/blog.csv')
data = data.dropna()

var = data.columns[0:276]
y = data.columns[-1]

aver_calibrate("blog", data, var, y)
