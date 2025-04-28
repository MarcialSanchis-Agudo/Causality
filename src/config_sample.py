class WallRecon:
    N_VARS_OUT = 1
    FLUCTUATIONS_PRED = False
    RELU_THRESHOLD = 0.0

class OuterRecon:
    N_VARS_OUT = 1
    FLUCTUATIONS_PRED = False
    RELU_THRESHOLD = 0.0

class PODSuperResolution:
    N_VARS_OUT = 1
    FLUCTUATIONS_PRED = False
    RELU_THRESHOLD = 0.0

config = {
    "train_tfrecords": ["../fcn_pod_data/pod_data.tfrecord"],
    "val_tfrecords": ["../fcn_pod_data/pod_data.tfrecord"],
    "model_name": "fcn",
    "save_dir": "./checkpoints",
    "num_epochs": 300,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "input_shape": [32, 32, 1],   # channels_last
    "output_shape": [32, 32, 1],
    "log_freq": 50,
    "val_freq": 100
}