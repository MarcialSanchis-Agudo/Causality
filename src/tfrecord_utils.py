import os
import numpy as np
import tensorflow as tf

X_input_path = '../fan_pod_data/X_input_trunc.npy'
X_output_path = '../fan_pod_data/X_output_full.npy'
tfrecord_path = '../fan_pod_data/pod_data.tfrecord'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(x_input, x_output):
    feature = {
        'x_input': _bytes_feature(x_input.tobytes()),
        'x_output': _bytes_feature(x_output.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def convert_to_tfrecord():
    X_input = np.load(X_input_path).astype(np.float32)
    X_output = np.load(X_output_path).astype(np.float32)

    # Normalize independently
    X_input_min, X_input_max = X_input.min(), X_input.max()
    X_output_min, X_output_max = X_output.min(), X_output.max()

    X_input = (X_input - X_input_min) / (X_input_max - X_input_min)
    X_output = (X_output - X_output_min) / (X_output_max - X_output_min)

    X_input = X_input.reshape((-1, 32, 32, 1))
    X_output = X_output.reshape((-1, 32, 32, 1))

    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for x_in, x_out in zip(X_input, X_output):
            example = serialize_example(x_in, x_out)
            writer.write(example)

    print(f"✅ TFRecord saved at {tfrecord_path}")

if __name__ == "__main__":
    convert_to_tfrecord()