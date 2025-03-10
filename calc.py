#!/usr/bin/env python3

import sys
sys.path.insert(0, '../optimizer/')

import tensorflow as tf
from tensorflow import keras as K
from auxiliary_functions import collect_data, save_weights_turbomole
from custom_callbacks import PrintMeanPredictionCallback, CustomSaveCallback
from model_creator import ModelCreator

# Set global float precision
tf.keras.backend.set_floatx('float32')

# Training parameters
EPOCHS = 6000
LEARNING_RATE = 0.01
L2_RATIO = 0.0

# Model architecture parameters
HIDDEN_UNITS = 64
NUM_LAYERS = 3
ACTIVATION_HIDDEN = 'gelu'
ACTIVATION_OUTPUT = 'sigmoid'
INPUT_SQUEEZE = 'SignedLogTransform'

# Testset selection
TESTSET_NAMES = ['W417', 'BH76']
TESTSET_PATHS = {
    'W417': "/path/to/data/w417/",
    'BH76': "/path/to/data/bh76/"
}
SCALES = [0.5, 0.5]

#TESTSET_NAMES = ['debugtest']
#TESTSET_PATHS = {
#    'debugtest': "/path/to/data/w417/"
#}
#SCALES = [1.0]

# Input feature configuration (density, density gradient, exact exchange density, tau, laplacian)
FEATURES = [True, True, False, True, False, False]
NUM_INPUTS = 7

# Load input data
x_train, y_train, dict_all, x_features = collect_data(
    testset_names=TESTSET_NAMES,
    testset_paths=TESTSET_PATHS,
    scales=SCALES,
    logical_features=FEATURES,
    MP2=True
)

# Create the model
model_creator = ModelCreator(
    num_inputs=NUM_INPUTS,
    input_squeeze=INPUT_SQUEEZE,
    number_of_layers=NUM_LAYERS,
    hidden_units=HIDDEN_UNITS,
    activation_function_hidden=ACTIVATION_HIDDEN,
    activation_function_output=ACTIVATION_OUTPUT,
    learning_rate=LEARNING_RATE,
    dict_all=dict_all,
    l2_ratio=L2_RATIO,
    x_model='PBE',
    c_model='B97',
    nlx=1.0,
    scal_opp=[0.45332184, -0.15449581, -0.23975305, 1.84252884, 0.33026814, -2.89710851],
    scal_ss=[-0.1459765, -0.61486652, 1.27599001, -1.69344595, 1.35510071, -0.18548735, 1.50881231],
    c_mp2opp=0.577,
    c_mp2par=0.248,
    seed=42,
    c_ss=0.09544,
    c_opp=0.004987,
    corr_train=False,
    rs_model=True,
    testset_names=TESTSET_NAMES,
    print_logical=False
)

model, my_loss = model_creator.create_model()

# Define callbacks
callbacks = [
    PrintMeanPredictionCallback(),
    CustomSaveCallback(base_output_path='weights_')
]

# Train the model
model.fit(
    [x_train, x_features],
    y_train,
    epochs=EPOCHS,
    shuffle=False,
    batch_size=x_train.shape[0],
    callbacks=callbacks
)

# Save weights in TurboMole-friendly format
save_weights_turbomole(model, output_path='final_weights')
