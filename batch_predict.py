import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import numpy as np
import shelve
import pickle

import sys
sys.path.append('util/')
import importlib

import batch_predictions_processor as processor

import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
import pandas as pd

print(f"Using TensorFlow version: {tf.__version__}")
# Optional: Configure GPU memory growth if needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

model_path = "data/checkpoints/model_9-25-25_0.keras"

scaler_filename = "data/pkl/scaler_9-25-25_0.pkl"
with open(scaler_filename, 'rb') as file:
    loaded_scaler = pickle.load(file)

model = tf.keras.models.load_model(model_path)

label_maps = {}

with shelve.open('data/shelve/label_maps_store') as s:
    label_maps = s['label_maps']

TASK_NAMES = ['plant', 'age', 'part', 'health', 'lifecycle']

INPUT_DIRECTORY = 'data/Greenhead_aug/temp'
OUTPUT_DIRECTORY = 'output/temp'
INPUT_FILE_PATH = 'data/morven_9-2025/raw_55691_or_ref.hdr'

#(input_dir, output_dir, model, scaler, label_maps, task_names)
ROWS_PER_CHUNK = 512 # Adjust this based on available RAM and image size
# Run the main function
processor.batch_classify(INPUT_FILE_PATH, OUTPUT_DIRECTORY, model, loaded_scaler, label_maps, TASK_NAMES, rows_per_chunk=ROWS_PER_CHUNK)