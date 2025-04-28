import os
import numpy as np
import tensorflow as tf
import importlib
from training_utils import get_model
import config_sample as config

def _parse_function(example_proto):
    features = {
        'x_input': tf.io.FixedLenFeature([], tf.string),
        'x_output': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    x_input = tf.io.decode_raw(parsed_features['x_input'], tf.float32)
    x_output = tf.io.decode_raw(parsed_features['x_output'], tf.float32)
    x_input = tf.reshape(x_input, config.config['input_shape'])
    x_output = tf.reshape(x_output, config.config['output_shape'])
    return x_input, x_output

def get_dataset(filenames, batch_size, shuffle=True):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def main():
    os.environ["MODEL_CNN"] = "PODSuperResolution"
    train_dataset = get_dataset(config.config['train_tfrecords'], config.config['batch_size'])
    val_dataset = get_dataset(config.config['val_tfrecords'], config.config['batch_size'], shuffle=False)
    model, losses = get_model(config.config['input_shape'], config.config['output_shape'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.config['learning_rate']), loss=losses, metrics=['mae'])
    model.fit(train_dataset, validation_data=val_dataset, epochs=config.config['num_epochs'])
    os.makedirs(config.config['save_dir'], exist_ok=True)
    model.save(os.path.join(config.config['save_dir'], 'saved_model.h5'))

if __name__ == "__main__":
    main()