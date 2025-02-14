#%pip install msvc-runtime

import os
from spectral import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import tensorflow as tf
#from tensorflow.keras.layers import Dropout, Dense, Input, Lambda
#import scikeras
import scipy
# import keras
# from scikeras.wrappers import KerasClassifier, KerasRegressor
from scipy.stats import reciprocal
# from sklearn.metrics import accuracy_score
# # hyperparameter optimization methods
# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_objective, plot_histogram, plot_evaluations
# from skopt.callbacks import CheckpointSaver
# #import optuna
# #from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
# #import hyperopt
import pickle
import pandas as pd
#import skopt
from sklearn.model_selection import train_test_split, cross_val_score
#import tqdm # nice progress bar
from scipy.signal import find_peaks

print(f'Numpy Version: {np. __version__}')
print(f'TensorFlow Version: {tf. __version__}')
#print(f'Keras Version: {keras. __version__}')

# NOTE:  The 'from lazypredict.Supervised import LazyClassifier' gives this error:
# TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'
# to fix this, go into sklearn and switch
#    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#  to
#    OH_encoder = OneHotEncoder(handle_unknown='ignore')


class NNSpectral():
    def __init__(self, sc, selected_indicies, prediction_class):
        return