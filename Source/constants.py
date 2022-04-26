#----------------------------------------------------------------------
# List of constants and some common functions
# Author: Pablo Villanueva Domingo
# Last update: 4/22
#----------------------------------------------------------------------

import numpy as np
import torch
import os
import random

# Random seeds
torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

#--- PARAMETERS AND CONSTANTS ---#

# Reduced Hubble constant
hred = 0.7

# Root path for simulations
simpathroot = "/projects/QUIJOTE/CAMELS/Sims/"

# Box size in comoving kpc/h
boxsize = 25.e3

# Validation and test size
valid_size, test_size = 0.15, 0.15

# Batch size
batch_size = 25

# Number of k bins in the power spectrum
ps_size = 79

#--- FUNCTIONS ---#

# Choose color depending on the CAMELS simulation suite
def colorsuite(suite):
    if suite=="IllustrisTNG":   return "purple"
    elif suite=="SIMBA":            return "dodgerblue"
