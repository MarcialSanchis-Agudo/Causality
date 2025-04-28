import os
import config_sample as config

prb_def = os.environ.get('MODEL_CNN', None)

if not prb_def:
    app = config.WallRecon
    prb_def = 'WallRecon'
elif prb_def == 'WallRecon':
    app = config.WallRecon
elif prb_def == 'OuterRecon':
    app = config.OuterRecon
elif prb_def == 'PODSuperResolution':
    app = config.PODSuperResolution
else:
    raise ValueError('"MODEL_CNN" environment variable must be defined as "WallRecon", "OuterRecon", or "PODSuperResolution"')

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects

"""
def cnn_model(input_shape, padding, pad_out, pred_fluct=app.FLUCTUATIONS_PRED):
    input_data = layers.Input(shape=input_shape, name='input_data')
    cnv_1 = layers.Conv2D(64, (5, 5), padding=padding)(input_data)
    bch_1 = layers.BatchNormalization()(cnv_1)
    act_1 = layers.Activation('relu')(bch_1)
    cnv_2 = layers.Conv2D(128, (3, 3), padding=padding)(act_1)
    bch_2 = layers.BatchNormalization()(cnv_2)
    act_2 = layers.Activation('relu')(bch_2)
    cnv_3 = layers.Conv2D(256, (3, 3), padding=padding)(act_2)
    bch_3 = layers.BatchNormalization()(cnv_3)
    act_3 = layers.Activation('relu')(bch_3)
    cnv_4 = layers.Conv2D(256, (3, 3), padding=padding)(act_3)
    bch_4 = layers.BatchNormalization()(cnv_4)
    act_4 = layers.Activation('relu')(bch_4)
    cnv_5 = layers.Conv2D(128, (3, 3), padding=padding)(act_4)
    bch_5 = layers.BatchNormalization()(cnv_5)
    act_5 = layers.Activation('relu')(bch_5)
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding)(act_5)
    act_b1 = layers.Activation(thres_relu if pred_fluct else 'relu')(cnv_b1)
    output_b1 = layers.Cropping2D(((pad_out//2, pad_out//2), (pad_out//2, pad_out//2)), name='output_b1')(act_b1)
    outputs_model = output_b1
    losses = {'output_b1': 'mse'}
    CNN_model = tf.keras.models.Model(inputs=input_data, outputs=outputs_model)
    return CNN_model, losses

def thres_relu(x):
    return tf.keras.activations.relu(x, threshold=app.RELU_THRESHOLD)

def get_model(input_shape, output_shape=None):
    return cnn_model(input_shape, padding='same', pad_out=0)

get_custom_objects().update({'thres_relu': layers.Activation(thres_relu)})
"""
def light_cnn_model(input_shape):
    from tensorflow.keras import layers

    input_data = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_data)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(1, (3,3), activation='linear', padding='same')(x)
    
    model = tf.keras.models.Model(inputs=input_data, outputs=x)
    losses = 'mse'
    return model, losses

def get_model(input_shape, output_shape=None):
    import config_sample as config
    prb_def = os.environ.get('MODEL_CNN', None)
    
    if prb_def == 'PODSuperResolution':
        return light_cnn_model(input_shape)
    else:
        from fcn import cnn_model
        return cnn_model(input_shape, padding='same', pad_out=0)
    else:
        from fcn import cnn_model
        return cnn_model(input_shape, padding='same', pad_out=0)
